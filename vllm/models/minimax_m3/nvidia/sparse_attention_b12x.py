# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B12X MSA-backed MiniMax M3 block-sparse attention.

This path keeps vLLM ownership of scratch memory: decode uses prepared graph
replay plans in both eager prewarm and CUDA graph capture, extend/prefill stays
on the eager planner, and every launch binds runtime device metadata plus the
Triton indexer's q2k indices into caller scratch borrowed from vLLM's workspace
manager.
"""

from __future__ import annotations

import math
import os
from typing import Any, Literal

import torch

from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.models.minimax_m3.common.ops.sparse_attn import (
    SPARSE_BLOCK_SIZE,
    minimax_m3_sparse_attn,
    minimax_m3_sparse_attn_decode,
)
from vllm.models.minimax_m3.common.sparse_attention import (
    MiniMaxM3SparseImpl,
    MiniMaxM3SparseMetadata,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import canonicalize_singleton_dim_strides
from vllm.v1.attention.backend import AttentionLayer
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_logger(__name__)

_B12X_MINIMAX_MSA_TOPK = 16
_B12X_MINIMAX_MSA_PAGE_SIZE = 128
_B12X_FP8_KV_CACHE_DTYPES = ("fp8", "fp8_e4m3")
_B12X_MSA_COMPARE_TRITON = "VLLM_B12X_MINIMAX_M3_MSA_COMPARE_TRITON"


def _is_b12x_fp8_kv_cache(kv_cache_dtype: str) -> bool:
    return kv_cache_dtype in _B12X_FP8_KV_CACHE_DTYPES


def _capture_alloc_forbidden() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except RuntimeError:
        return False


def _ensure_i32_contiguous(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.dtype != torch.int32:
        if _capture_alloc_forbidden():
            raise RuntimeError(
                "B12X MiniMax M3 MSA would convert "
                f"{name} to int32 during CUDA graph capture."
            )
        tensor = tensor.to(torch.int32)
    if not tensor.is_contiguous():
        if _capture_alloc_forbidden():
            raise RuntimeError(
                "B12X MiniMax M3 MSA would make "
                f"{name} contiguous during CUDA graph capture."
            )
        tensor = tensor.contiguous()
    return tensor


def _cu_seqlens_total_q(cu_seqlens_q: torch.Tensor) -> int:
    if _capture_alloc_forbidden():
        raise RuntimeError(
            "B12X MiniMax M3 MSA cannot read cu_seqlens_q on host during "
            "CUDA graph capture."
        )
    return int(cu_seqlens_q[-1].item())


def _kv_dtype_from_cache_config(kv_cache_dtype: str) -> torch.dtype:
    if _is_b12x_fp8_kv_cache(kv_cache_dtype):
        fp8_dtype = current_platform.fp8_dtype()
        if fp8_dtype != torch.float8_e4m3fn:
            raise NotImplementedError(
                "B12X MiniMax M3 MSA supports only fp8_e4m3 KV caches, got "
                f"platform fp8 dtype {fp8_dtype}."
            )
        return fp8_dtype
    if kv_cache_dtype == "bfloat16":
        return torch.bfloat16
    if kv_cache_dtype == "auto":
        return get_current_vllm_config().model_config.dtype
    raise NotImplementedError(
        "B12X MiniMax M3 MSA supports auto, bfloat16, fp8, and fp8_e4m3 "
        f"KV caches, got {kv_cache_dtype!r}."
    )


class MiniMaxM3SparseB12XImpl(MiniMaxM3SparseImpl):
    """B12X paged MSA attend over MiniMax's Triton-selected sparse blocks."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        kv_cache_dtype: str = "auto",
        *,
        topk_blocks: int,
        sparse_block_size: int,
    ) -> None:
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            kv_cache_dtype,
            topk_blocks=topk_blocks,
            sparse_block_size=sparse_block_size,
        )
        if self.head_size != 128:
            raise NotImplementedError(
                "B12X MiniMax M3 MSA requires head_size=128, "
                f"got {self.head_size}."
            )
        if self.block_size != _B12X_MINIMAX_MSA_PAGE_SIZE:
            raise NotImplementedError(
                "B12X MiniMax M3 MSA requires page/block size 128, "
                f"got {self.block_size}."
            )
        if self.block_size != SPARSE_BLOCK_SIZE:
            raise NotImplementedError(
                "B12X MiniMax M3 MSA expects the sparse block size to match "
                f"the KV page size, got {SPARSE_BLOCK_SIZE} vs {self.block_size}."
            )
        if self.topk_blocks != _B12X_MINIMAX_MSA_TOPK:
            raise NotImplementedError(
                "B12X MiniMax M3 MSA requires sparse_topk_blocks=16, "
                f"got {self.topk_blocks}."
            )
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                "B12X MiniMax M3 MSA requires q heads divisible by kv heads."
            )
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        if self.num_queries_per_kv != 16:
            raise NotImplementedError(
                "B12X MiniMax M3 MSA requires GQA group size 16, got "
                f"{self.num_queries_per_kv}."
            )
        expected_scale = self.head_size**-0.5
        if not math.isclose(float(scale), expected_scale, rel_tol=1e-5, abs_tol=1e-7):
            raise NotImplementedError(
                "B12X MiniMax M3 MSA requires canonical softmax scale "
                f"head_dim**-0.5={expected_scale}, got {scale}."
            )

        self.kv_torch_dtype = _kv_dtype_from_cache_config(kv_cache_dtype)
        if self.kv_torch_dtype not in (torch.bfloat16, torch.float8_e4m3fn):
            raise NotImplementedError(
                "B12X MiniMax M3 MSA page-128 plans support only bf16 or "
                f"fp8_e4m3 KV caches, got {self.kv_torch_dtype}."
            )

        self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        self._unit_fp8_descale = torch.ones(
            (), dtype=torch.float32, device=self.device
        )

        from vllm.v1.attention.backends.b12x_attn import (
            _disable_cutlass_memory_debug_snapshot_if_off,
        )

        _disable_cutlass_memory_debug_snapshot_if_off()

        from b12x.attention.paged.api import paged_attention_forward
        from b12x.attention.paged.planner import create_paged_plan
        from b12x.integration.paged_attention_scratch import (
            B12XPagedAttentionScratchCaps,
            plan_paged_attention_scratch,
        )

        self._paged_attention_forward = paged_attention_forward
        self._create_paged_plan = create_paged_plan
        self._scratch_caps_type = B12XPagedAttentionScratchCaps
        self._plan_paged_attention_scratch = plan_paged_attention_scratch
        self._decode_graph_scratch_plans: dict[tuple[Any, ...], Any] = {}
        self._triton_compare_reports = 0
        self._debug_reports = 0

        logger.info_once(
            "Using B12X MiniMax M3 MSA attention: heads=%d kv_heads=%d "
            "head_dim=%d kv_dtype=%s",
            self.num_heads,
            self.num_kv_heads,
            self.head_size,
            self.kv_torch_dtype,
        )

    def _kv_cache_views(
        self,
        kv_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_fp8_kv:
            kv_cache = kv_cache.view(self.kv_cache_fp8_dtype)
        key_cache, value_cache = kv_cache.unbind(1)
        key_cache = canonicalize_singleton_dim_strides(key_cache)
        value_cache = canonicalize_singleton_dim_strides(value_cache)
        if (
            key_cache.dtype != self.kv_torch_dtype
            or value_cache.dtype != self.kv_torch_dtype
        ):
            raise TypeError(
                "B12X MiniMax M3 MSA plan expects KV dtype "
                f"{self.kv_torch_dtype}, got {key_cache.dtype}/{value_cache.dtype}."
            )
        return key_cache, value_cache

    def _prepare_fp8_descales(
        self,
        num_reqs: int,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not _is_b12x_fp8_kv_cache(self.kv_cache_dtype):
            return None, None
        if num_reqs <= 0:
            raise ValueError(
                "B12X MiniMax M3 MSA fp8 descale request count must be positive."
            )
        if self._unit_fp8_descale.device != device:
            raise RuntimeError(
                "B12X MiniMax M3 MSA fp8 descales must be on the query device."
            )
        descale = self._unit_fp8_descale.expand(num_reqs)
        return descale, descale

    def _validate_q2k_indices(
        self,
        q2k_indices: torch.Tensor,
        total_q: int,
    ) -> None:
        if q2k_indices.ndim != 3:
            raise ValueError(
                "B12X MiniMax M3 MSA q2k_indices must have shape "
                "[kv_heads, total_q, 16]."
            )
        if q2k_indices.dtype != torch.int32:
            raise TypeError("B12X MiniMax M3 MSA q2k_indices must be int32.")
        if not q2k_indices.is_contiguous():
            raise ValueError("B12X MiniMax M3 MSA q2k_indices must be contiguous.")
        if (
            int(q2k_indices.shape[0]) != self.num_kv_heads
            or int(q2k_indices.shape[1]) < total_q
            or int(q2k_indices.shape[2]) != _B12X_MINIMAX_MSA_TOPK
        ):
            raise ValueError(
                "B12X MiniMax M3 MSA q2k_indices must have shape "
                f"({self.num_kv_heads}, >={total_q}, 16), got "
                f"{tuple(q2k_indices.shape)}."
            )

    def _make_scratch_caps(
        self,
        *,
        q: torch.Tensor,
        key_cache: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        mode: Literal["decode", "extend"],
        max_work_items: int,
        max_partial_rows: int,
        use_cuda_graph: bool,
    ) -> Any:
        return self._scratch_caps_type(
            device=q.device,
            mode=mode,
            dtype=q.dtype,
            kv_dtype=self.kv_torch_dtype,
            num_q_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim_qk=self.head_size,
            head_dim_vo=self.head_size,
            page_size=_B12X_MINIMAX_MSA_PAGE_SIZE,
            max_total_q=max(int(q.shape[0]), 1),
            max_batch=max(int(cache_seqlens.shape[0]), 1),
            max_page_table_width=max(int(page_table.shape[1]), 1),
            max_work_items=max(int(max_work_items), 1),
            max_partial_rows=max(int(max_partial_rows), 0),
            num_cache_pages=max(int(key_cache.shape[0]), 1),
            use_cuda_graph=use_cuda_graph,
            msa_block_sparse=True,
        )

    def _decode_graph_scratch_key(
        self,
        *,
        q: torch.Tensor,
        key_cache: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
    ) -> tuple[Any, ...]:
        return (
            q.device.index,
            q.dtype,
            self.kv_torch_dtype,
            int(q.shape[0]),
            int(cache_seqlens.shape[0]),
            int(page_table.shape[1]),
            int(key_cache.shape[0]),
            self.num_heads,
            self.num_kv_heads,
            self.head_size,
        )

    def _prepare_decode_graph_scratch_plan(
        self,
        *,
        q: torch.Tensor,
        key_cache: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
    ) -> Any:
        key = self._decode_graph_scratch_key(
            q=q,
            key_cache=key_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
        )
        cached = self._decode_graph_scratch_plans.get(key)
        if cached is not None:
            return cached
        batch = int(cache_seqlens.shape[0])
        page_table_width = int(page_table.shape[1])
        max_decode_work_items = max(batch * 32, 1)
        graph_scratch_plan = self._plan_paged_attention_scratch(
            self._make_scratch_caps(
                q=q,
                key_cache=key_cache,
                page_table=page_table,
                cache_seqlens=cache_seqlens,
                mode="decode",
                max_work_items=max_decode_work_items,
                max_partial_rows=max_decode_work_items,
                use_cuda_graph=True,
            )
        )
        graph_scratch_plan.prepare_decode_graph_replay_state(
            batch=batch,
            max_page_table_width=page_table_width,
            total_q_capacity=max(int(q.shape[0]), 1),
            max_cache_page_count=page_table_width,
            fixed_split_size=-1,
            window_left=-1,
        )
        self._decode_graph_scratch_plans[key] = graph_scratch_plan
        return graph_scratch_plan

    def _select_scratch_plan(
        self,
        *,
        q: torch.Tensor,
        key_cache: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        mode: Literal["decode", "extend"],
        max_work_items: int,
        max_partial_rows: int,
    ) -> Any:
        if mode == "decode":
            return self._prepare_decode_graph_scratch_plan(
                q=q,
                key_cache=key_cache,
                page_table=page_table,
                cache_seqlens=cache_seqlens,
            )
        return self._plan_paged_attention_scratch(
            self._make_scratch_caps(
                q=q,
                key_cache=key_cache,
                page_table=page_table,
                cache_seqlens=cache_seqlens,
                mode=mode,
                max_work_items=max_work_items,
                max_partial_rows=max_partial_rows,
                use_cuda_graph=False,
            )
        )

    def _run_b12x_slice(
        self,
        *,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        q2k_indices: torch.Tensor,
        mode: Literal["decode", "extend"],
        prefix_lens: torch.Tensor | None = None,
        max_query_len: int | None = None,
        layer_name: str,
    ) -> None:
        total_q_capacity = int(q.shape[0])
        if total_q_capacity <= 0:
            return
        if q.dtype not in (torch.float16, torch.bfloat16):
            raise TypeError(
                f"B12X MiniMax M3 MSA does not support q dtype {q.dtype}."
            )
        if output.dtype != q.dtype:
            raise TypeError(
                "B12X MiniMax M3 MSA expects output dtype to match q dtype, "
                f"got {output.dtype} vs {q.dtype}."
            )

        page_table = _ensure_i32_contiguous(page_table, "page_table")
        cache_seqlens = _ensure_i32_contiguous(cache_seqlens, "seq_lens")
        cu_seqlens_q = _ensure_i32_contiguous(cu_seqlens_q, "cu_seqlens_q")

        capturing = _capture_alloc_forbidden()
        if mode == "extend":
            if capturing:
                raise RuntimeError(
                    "B12X MiniMax M3 MSA can only use a pre-planned decode "
                    "graph scratch path during CUDA graph capture."
                )
            total_q = _cu_seqlens_total_q(cu_seqlens_q)
            if total_q < 0 or total_q > total_q_capacity:
                raise ValueError(
                    "B12X MiniMax M3 MSA cu_seqlens_q total query length "
                    f"{total_q} is outside q capacity {total_q_capacity}."
                )
            if total_q < total_q_capacity:
                output[total_q:].zero_()
                if total_q == 0:
                    return
                q = q[:total_q]
                output = output[:total_q]
        else:
            total_q = total_q_capacity

        self._validate_q2k_indices(q2k_indices, total_q)

        if mode == "decode":
            max_work_items = 1
            max_partial_rows = 0
        else:
            plan = self._create_paged_plan(
                q,
                key_cache,
                value_cache,
                page_table,
                cache_seqlens,
                cu_seqlens_q,
                mode=mode,
                msa_block_sparse=True,
            )
            max_work_items = max(
                int(plan.new_batch_size),
                int(plan.padded_batch_size),
                1,
            )
            max_partial_rows = max(int(plan.total_num_partial_rows), 0)
        scratch_plan = self._select_scratch_plan(
            q=q,
            key_cache=key_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            mode=mode,
            max_work_items=max_work_items,
            max_partial_rows=max_partial_rows,
        )
        (scratch_storage,) = current_workspace_manager().get_simultaneous(
            ((int(scratch_plan.layout.nbytes),), torch.uint8),
        )
        num_reqs = int(cache_seqlens.shape[0])
        if (
            os.getenv("VLLM_DEBUG_B12X_MINIMAX_M3_MSA", "0") == "1"
            and self._debug_reports < 12
        ):
            plan = getattr(scratch_plan, "_plan", None)
            logger.warning(
                "B12X MiniMax M3 MSA bind layer=%s mode=%s capturing=%s "
                "q=%s out=%s page_table=%s cache_seqlens=%s cu=%s "
                "q2k=%s scratch=%d plan_total_q=%s plan_split=%s "
                "plan_kv_chunk=%s",
                layer_name,
                mode,
                capturing,
                tuple(q.shape),
                tuple(output.shape),
                tuple(page_table.shape),
                tuple(cache_seqlens.shape),
                tuple(cu_seqlens_q.shape),
                tuple(q2k_indices.shape),
                int(scratch_plan.layout.nbytes),
                getattr(plan, "total_q", None),
                getattr(plan, "split_kv", None),
                getattr(plan, "kv_chunk_size", None),
            )
            self._debug_reports += 1
        k_descale, v_descale = self._prepare_fp8_descales(num_reqs, q.device)
        if mode == "decode" and not capturing:
            scratch_plan._q2k_indices_data_ptr = None
        binding = scratch_plan.bind(
            scratch=scratch_storage,
            q=q,
            k_cache=key_cache,
            v_cache=value_cache,
            output=output,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            active_total_q=total_q,
            q2k_indices=q2k_indices,
            k_descale=k_descale,
            v_descale=v_descale,
        )
        self._paged_attention_forward(binding=binding)
        if mode == "decode" and not capturing:
            scratch_plan._q2k_indices_data_ptr = None
        if (
            os.getenv(_B12X_MSA_COMPARE_TRITON, "0") == "1"
            and not capturing
            and self._triton_compare_reports < 8
        ):
            self._compare_triton_slice(
                q=q,
                kv_cache=kv_cache,
                q2k_indices=q2k_indices,
                b12x_output=output,
                page_table=page_table,
                cache_seqlens=cache_seqlens,
                cu_seqlens_q=cu_seqlens_q,
                mode=mode,
                prefix_lens=prefix_lens,
                max_query_len=max_query_len,
                layer_name=layer_name,
            )
            self._triton_compare_reports += 1

    def _compare_triton_slice(
        self,
        *,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        q2k_indices: torch.Tensor,
        b12x_output: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        mode: Literal["decode", "extend"],
        prefix_lens: torch.Tensor | None,
        max_query_len: int | None,
        layer_name: str,
    ) -> None:
        triton_output = torch.empty_like(b12x_output)
        if mode == "decode":
            num_reqs = int(cache_seqlens.shape[0])
            decode_query_len = max(int(q.shape[0]) // max(num_reqs, 1), 1)
            minimax_m3_sparse_attn_decode(
                q,
                kv_cache,
                q2k_indices,
                page_table,
                cache_seqlens,
                self.num_kv_heads,
                self.scale,
                triton_output,
                decode_query_len,
            )
        else:
            if prefix_lens is None:
                query_lens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
                prefix_lens = cache_seqlens - query_lens
            if max_query_len is None:
                max_query_len = int((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max())
            minimax_m3_sparse_attn(
                q,
                kv_cache,
                q2k_indices,
                page_table,
                cu_seqlens_q,
                cache_seqlens,
                prefix_lens,
                max_query_len,
                self.num_kv_heads,
                self.scale,
                triton_output,
            )
        torch.cuda.synchronize()
        b12x_f32 = b12x_output.float()
        triton_f32 = triton_output.float()
        finite_b12x = bool(torch.isfinite(b12x_f32).all().item())
        finite_triton = bool(torch.isfinite(triton_f32).all().item())
        abs_diff = (b12x_f32 - triton_f32).abs()
        max_abs = float(abs_diff.max().item())
        mean_abs = float(abs_diff.mean().item())
        cos = float(
            torch.nn.functional.cosine_similarity(
                b12x_f32.reshape(-1), triton_f32.reshape(-1), dim=0
            ).item()
        )
        logger.warning(
            "B12X MiniMax M3 MSA compare layer=%s mode=%s q=%s reqs=%d "
            "q2k=%s finite=%s/%s max_abs=%.6g mean_abs=%.6g cos=%.8f",
            layer_name,
            mode,
            tuple(q.shape),
            int(cache_seqlens.shape[0]),
            tuple(q2k_indices.shape),
            finite_b12x,
            finite_triton,
            max_abs,
            mean_abs,
            cos,
        )

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        topk_idx: tuple[torch.Tensor | None, torch.Tensor | None],
        output: torch.Tensor,
    ) -> torch.Tensor:
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return output
        main_md = attn_metadata[layer.layer_name]  # type: ignore[attr-defined]
        assert isinstance(main_md, MiniMaxM3SparseMetadata)
        decode_topk, prefill_topk = topk_idx

        nd = main_md.num_decode_tokens
        num_tokens = main_md.num_actual_tokens
        hd = self.head_size
        q = query[:num_tokens].view(-1, self.num_heads, hd)
        out = output[:num_tokens].view(-1, self.num_heads, hd)
        key_cache, value_cache = self._kv_cache_views(kv_cache)

        if main_md.num_decodes > 0:
            d = main_md.decode
            assert d is not None and decode_topk is not None
            decode_mode: Literal["decode", "extend"] = (
                "decode" if d.decode_query_len == 1 else "extend"
            )
            self._run_b12x_slice(
                q=q[:nd],
                kv_cache=kv_cache,
                key_cache=key_cache,
                value_cache=value_cache,
                output=out[:nd],
                page_table=d.block_table,
                cache_seqlens=d.seq_lens,
                cu_seqlens_q=d.cu_seqlens_q,
                q2k_indices=decode_topk,
                mode=decode_mode,
                layer_name=layer.layer_name,
            )

        if main_md.num_prefills > 0:
            p = main_md.prefill
            assert p is not None and prefill_topk is not None
            self._run_b12x_slice(
                q=q[nd:],
                kv_cache=kv_cache,
                key_cache=key_cache,
                value_cache=value_cache,
                output=out[nd:],
                page_table=p.block_table,
                cache_seqlens=p.seq_lens,
                cu_seqlens_q=p.cu_seqlens_q,
                q2k_indices=prefill_topk,
                mode="extend",
                prefix_lens=p.context_lens,
                max_query_len=p.max_query_len,
                layer_name=layer.layer_name,
            )

        return output
