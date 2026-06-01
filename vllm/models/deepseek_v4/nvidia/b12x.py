# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""b12x compressed sparse-MLA impl for DeepSeek-V4 (consumer Blackwell SM120).

The DSV4 sparse-MLA path uses global top-k slot ids from
``compute_global_topk_indices_and_lens``, a SWA + indexed dual cache, paged
per-chunk prefill, and ``attn_sink``. The leaf call goes through b12x's
``compressed_mla_decode_forward`` binding API (``plan_compressed_mla_scratch``
-> fresh scratch -> ``plan.bind`` in ordinary Python, then one
``compressed_mla_decode_forward`` leaf call), matching the plan/bind style used
by the b12x WO-projection and mHC integrations in this tree. No persistent
workspace object is held.

DSV4 compressed-MLA contract (== upstream/DeepGEMM): q_head_dim = 448 NoPE +
64 RoPE = 512, V = 512; the ``fp8_ds_mla`` 584 B/token page (448 NoPE fp8 +
128 RoPE bf16 + 8-byte UE8M0 footer) is read directly.
"""

from typing import TYPE_CHECKING, ClassVar, Literal, cast

import torch

from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4.common.ops import (
    compute_global_topk_indices_and_lens,
)
from vllm.models.deepseek_v4.nvidia.flashmla import (
    DeepseekV4FlashMLASparseBackend,
    DeepseekV4SparseMLAAttentionImpl,
)
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseMetadata

if TYPE_CHECKING:
    from vllm.models.deepseek_v4.attention import DeepseekV4MLAAttention
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata

# DSV4 compressed-MLA dims (q_head_dim = 448 NoPE + 64 RoPE = 512; V = 512).
_DSV4_HEAD_DIM = 512
_DSV4_V_HEAD_DIM = 512
_DSV4_CACHE_BYTES_PER_TOKEN = 584
_DECODE_SPLIT_TILE = 64


def _cdiv(x: int, y: int) -> int:
    return (int(x) + int(y) - 1) // int(y)


def _b12x_cache_page_view(
    cache: torch.Tensor,
    page_size: int,
    name: str,
) -> torch.Tensor:
    """Return a uint8 ``[pages, padded_page_bytes]`` view for b12x kernels."""
    min_page_nbytes = int(page_size) * _DSV4_CACHE_BYTES_PER_TOKEN
    if min_page_nbytes <= 0:
        raise ValueError(f"{name} page_size must be positive, got {page_size}")

    byte_cache = cache if cache.dtype == torch.uint8 else cache.view(torch.uint8)
    if byte_cache.ndim == 2:
        if int(byte_cache.shape[1]) < min_page_nbytes:
            raise RuntimeError(
                f"{name} page width {int(byte_cache.shape[1])} is smaller than "
                f"DSV4 page payload {min_page_nbytes}"
            )
        if not byte_cache.is_contiguous():
            raise RuntimeError(f"{name} page cache must be contiguous")
        return byte_cache

    if byte_cache.ndim < 2:
        raise RuntimeError(
            f"{name} expected a paged cache tensor, got shape {tuple(cache.shape)}"
        )

    pages = int(byte_cache.shape[0])
    page_stride_nbytes = int(byte_cache.stride(0))
    if page_stride_nbytes < min_page_nbytes:
        raise RuntimeError(
            f"{name} page stride {page_stride_nbytes} is smaller than DSV4 page "
            f"payload {min_page_nbytes}"
        )

    page_view = torch.as_strided(
        byte_cache,
        size=(pages, page_stride_nbytes),
        stride=(page_stride_nbytes, 1),
    )
    if not page_view.is_contiguous():
        raise RuntimeError(f"{name} padded page view must be contiguous")
    return page_view


def _run_compressed_mla(
    *,
    q: torch.Tensor,
    output: torch.Tensor,
    attn_sink: torch.Tensor,
    scale: float,
    swa_k_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    swa_page_size: int,
    indexed_k_cache: torch.Tensor | None,
    indexed_indices: torch.Tensor | None,
    indexed_lens: torch.Tensor | None,
    indexed_page_size: int | None,
    mode: Literal["decode", "extend"] = "decode",
) -> None:
    """Plan, bind, and call b12x compressed MLA in plain eager Python.

    ``q`` is ``[tokens, padded_heads, 512]`` (heads pre-padded to
    {16,32,64,128} by the outer wrapper). Indices are global slot ids, so no
    indexed page table is needed.
    """
    from b12x.integration.compressed_scratch import (
        B12XCompressedMLAScratchCaps,
        plan_compressed_mla_scratch,
    )
    from b12x.attention.mla.compressed_config import (
        compressed_mla_split_config_for_contract,
    )
    from b12x.integration.mla import compressed_mla_decode_forward
    import os

    rows, heads = q.shape[0], q.shape[1]
    q = q.contiguous()
    swa_indices = swa_indices.contiguous()
    swa_lens = swa_lens.contiguous()
    if indexed_indices is not None:
        indexed_indices = indexed_indices.contiguous()
    if indexed_lens is not None:
        indexed_lens = indexed_lens.contiguous()

    # b12x checks total_width = swa_width + indexed_width against scratch.topk,
    # so the scratch must be planned for the combined dual-cache width.
    width = int(swa_indices.shape[-1])
    if indexed_indices is not None:
        width += int(indexed_indices.shape[-1])
    num_splits_cap = max(1, _cdiv(width, _DECODE_SPLIT_TILE))
    max_chunks_override = os.getenv("B12X_COMPRESSED_MLA_MAX_CHUNKS")
    if max_chunks_override:
        num_splits_cap = min(num_splits_cap, max(1, int(max_chunks_override)))
    split_cfg = compressed_mla_split_config_for_contract(
        rows=max(1, int(rows)),
        width=max(1, int(width)),
        max_chunks=num_splits_cap,
    )
    num_splits_cap = max(1, int(split_cfg.num_chunks))

    plan = plan_compressed_mla_scratch(
        B12XCompressedMLAScratchCaps(
            device=q.device,
            num_q_heads=heads,
            max_q_rows=max(1, rows),
            max_width=width,
            head_dim=_DSV4_HEAD_DIM,
            v_head_dim=_DSV4_V_HEAD_DIM,
            page_size=int(swa_page_size),
            max_chunks_per_row=num_splits_cap,
        )
    )
    scratch = tuple(
        torch.empty(shape, dtype=dtype, device=q.device)
        for shape, dtype in plan.shapes_and_dtypes()
    )

    binding = plan.bind(
        scratch=scratch,
        q=q,
        swa_indices=swa_indices,
        swa_lengths=swa_lens,
        indexed_indices=indexed_indices,
        indexed_lengths=indexed_lens,
    )
    binding.scratch.mode = mode

    # attn_sink is sized to the model's padded_heads (max(n_local,64)); b12x
    # wants it at the real local head count (== q heads).
    sink = attn_sink[:heads].contiguous()
    out = compressed_mla_decode_forward(
        binding=binding,
        swa_k_cache=swa_k_cache,
        swa_page_size=int(swa_page_size),
        indexed_k_cache=indexed_k_cache,
        indexed_page_size=indexed_page_size,
        attn_sink=sink,
        sm_scale=scale,
        expected_num_q_heads=heads,
    )
    output.copy_(out)


class DeepseekV4B12xMLASparseBackend(DeepseekV4FlashMLASparseBackend):
    """b12x compressed sparse-MLA backend for DeepSeek-V4 (SM120 / SM121).

    Geometry is identical to the FlashMLA parent (``fp8_ds_mla`` 584 B page,
    head 512, block 64) -- it inherits ``get_kv_cache_shape`` /
    ``get_supported_head_sizes``; only the impl class differs.
    """

    @staticmethod
    def get_name() -> str:
        return "V4_B12X_SPARSE"

    @staticmethod
    def get_impl_cls() -> type["DeepseekV4B12xMLASparseImpl"]:
        return DeepseekV4B12xMLASparseImpl


class DeepseekV4B12xMLASparseImpl(DeepseekV4SparseMLAAttentionImpl):
    """b12x compressed sparse-MLA impl for DeepSeek-V4's custom MLA layer."""

    backend_cls: ClassVar[type[AttentionBackend]] = DeepseekV4B12xMLASparseBackend

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        if num_heads <= 16:
            return 16
        if num_heads <= 32:
            return 32
        if num_heads <= 64:
            return 64
        if num_heads <= 128:
            return 128
        raise ValueError(
            f"DeepseekV4 b12x sparse MLA does not support {num_heads} heads "
            "(kernel requires h_q in {16, 32, 64, 128})."
        )

    @classmethod
    def forward_mqa(  # type: ignore[override]
        cls,
        layer: "DeepseekV4MLAAttention",
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        del kv, positions
        assert output.shape == q.shape, (
            f"output buffer shape {output.shape} must match q shape {q.shape}"
        )
        assert output.dtype == q.dtype, (
            f"output buffer dtype {output.dtype} must match q dtype {q.dtype}"
        )

        prefix = layer.prefix
        swa_cache_prefix = layer.swa_cache_layer.prefix
        compress_ratio = layer.compress_ratio
        compressed_kv_cache = layer.kv_cache
        swa_kv_cache = layer.swa_cache_layer.kv_cache
        topk_indices_buffer = layer.topk_indices_buffer
        attn_sink = layer.attn_sink
        scale = layer.scale

        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            # Warmup dummy run: no metadata; the per-call binding allocates its
            # own scratch, so nothing to pre-reserve. Zero and return.
            output.zero_()
            return

        assert isinstance(attn_metadata, dict)
        flashmla_metadata = cast(
            FlashMLASparseMetadata | None, attn_metadata.get(prefix)
        )
        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(swa_cache_prefix),
        )
        assert swa_metadata is not None

        swa_only = compress_ratio <= 1
        self_kv_cache = compressed_kv_cache if not swa_only else None

        num_decodes = swa_metadata.num_decodes
        num_prefills = swa_metadata.num_prefills
        num_decode_tokens = swa_metadata.num_decode_tokens
        num_prefill_tokens = swa_metadata.num_prefill_tokens

        if num_prefills > 0:
            prefill_end = num_decode_tokens + num_prefill_tokens
            cls._forward_prefill(
                q=q[num_decode_tokens:prefill_end],
                compressed_k_cache=self_kv_cache,
                swa_k_cache=swa_kv_cache,
                output=output[num_decode_tokens:prefill_end],
                attn_metadata=flashmla_metadata,
                swa_metadata=swa_metadata,
                compress_ratio=compress_ratio,
                topk_indices_buffer=topk_indices_buffer,
                attn_sink=attn_sink,
                scale=scale,
            )
        if num_decodes > 0:
            cls._forward_decode(
                q=q[:num_decode_tokens],
                kv_cache=self_kv_cache,
                swa_kv_cache=swa_kv_cache,
                swa_metadata=swa_metadata,
                attn_metadata=flashmla_metadata,
                swa_only=swa_only,
                compress_ratio=compress_ratio,
                topk_indices_buffer=topk_indices_buffer,
                attn_sink=attn_sink,
                scale=scale,
                output=output[:num_decode_tokens],
            )

    @classmethod
    def _forward_decode(
        cls,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,  # only used when compress_ratio > 1
        swa_kv_cache: torch.Tensor,
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: FlashMLASparseMetadata | None,
        swa_only: bool,
        compress_ratio: int,
        topk_indices_buffer: torch.Tensor | None,
        attn_sink: torch.Tensor,
        scale: float,
        output: torch.Tensor,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        # Indexed (compressed) region global top-k.
        topk_indices = None
        topk_lens = None
        indexed_k_cache = None
        indexed_page_size = None
        if not swa_only:
            assert attn_metadata is not None
            assert kv_cache is not None
            assert swa_metadata.is_valid_token is not None
            block_size = attn_metadata.block_size // compress_ratio
            is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
            if compress_ratio == 4:
                assert topk_indices_buffer is not None
                topk_indices, topk_lens = compute_global_topk_indices_and_lens(
                    topk_indices_buffer[:num_decode_tokens],
                    swa_metadata.token_to_req_indices,
                    attn_metadata.block_table[:num_decodes],
                    block_size,
                    is_valid,
                )
            else:
                topk_indices = attn_metadata.c128a_global_decode_topk_indices
                topk_lens = attn_metadata.c128a_decode_topk_lens
            indexed_page_size = block_size
            indexed_k_cache = _b12x_cache_page_view(
                kv_cache,
                indexed_page_size,
                "indexed_k_cache",
            )

        swa_indices = swa_metadata.decode_swa_indices
        swa_lens = swa_metadata.decode_swa_lens
        assert swa_indices is not None
        assert swa_lens is not None
        swa_k_cache = _b12x_cache_page_view(
            swa_kv_cache,
            swa_metadata.block_size,
            "swa_k_cache",
        )

        _run_compressed_mla(
            q=q,
            output=output,
            attn_sink=attn_sink,
            scale=scale,
            swa_k_cache=swa_k_cache,
            swa_indices=swa_indices,
            swa_lens=swa_lens,
            swa_page_size=swa_metadata.block_size,
            indexed_k_cache=indexed_k_cache,
            indexed_indices=topk_indices,
            indexed_lens=topk_lens,
            indexed_page_size=indexed_page_size,
            mode="decode",
        )

    @classmethod
    def _forward_prefill(
        cls,
        q: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: FlashMLASparseMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
        compress_ratio: int,
        topk_indices_buffer: torch.Tensor | None,
        attn_sink: torch.Tensor,
        scale: float,
    ) -> None:
        swa_only = compress_ratio <= 1
        num_prefills = swa_metadata.num_prefills
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens
        num_prefill_tokens = swa_metadata.num_prefill_tokens

        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        assert query_start_loc_cpu is not None
        prefill_token_base = query_start_loc_cpu[num_decodes]

        # Indexed (compressed) region global top-k for the prefill rows.
        extra_topk_indices = None
        extra_topk_lens = None
        indexed_k_cache = None
        indexed_page_size = None
        if not swa_only:
            assert attn_metadata is not None
            assert compressed_k_cache is not None
            if compress_ratio == 4:
                assert topk_indices_buffer is not None
                local_topk_indices = topk_indices_buffer[
                    num_decode_tokens : num_decode_tokens + num_prefill_tokens
                ]
            else:
                local_topk_indices = attn_metadata.c128a_prefill_topk_indices
            assert swa_metadata.token_to_req_indices is not None
            assert swa_metadata.is_valid_token is not None
            prefill_slice = slice(
                num_decode_tokens, num_decode_tokens + num_prefill_tokens
            )
            block_size = attn_metadata.block_size // compress_ratio
            extra_topk_indices, extra_topk_lens = compute_global_topk_indices_and_lens(
                local_topk_indices,
                swa_metadata.token_to_req_indices[prefill_slice],
                attn_metadata.block_table,
                block_size,
                swa_metadata.is_valid_token[prefill_slice],
            )
            indexed_page_size = block_size
            indexed_k_cache = _b12x_cache_page_view(
                compressed_k_cache,
                indexed_page_size,
                "indexed_k_cache",
            )

        assert swa_metadata.prefill_swa_indices is not None
        assert swa_metadata.prefill_swa_lens is not None
        swa_k_cache = _b12x_cache_page_view(
            swa_k_cache,
            swa_metadata.block_size,
            "swa_k_cache",
        )

        num_chunks = (
            num_prefills + cls.PREFILL_CHUNK_SIZE - 1
        ) // cls.PREFILL_CHUNK_SIZE
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * cls.PREFILL_CHUNK_SIZE
            chunk_end = min(chunk_start + cls.PREFILL_CHUNK_SIZE, num_prefills)
            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )

            idx_chunk = (
                extra_topk_indices[query_start:query_end]
                if extra_topk_indices is not None
                else None
            )
            idx_lens_chunk = (
                extra_topk_lens[query_start:query_end]
                if extra_topk_lens is not None
                else None
            )
            _run_compressed_mla(
                q=q[query_start:query_end],
                output=output[query_start:query_end],
                attn_sink=attn_sink,
                scale=scale,
                swa_k_cache=swa_k_cache,
                swa_indices=swa_metadata.prefill_swa_indices[query_start:query_end],
                swa_lens=swa_metadata.prefill_swa_lens[query_start:query_end],
                swa_page_size=swa_metadata.block_size,
                indexed_k_cache=indexed_k_cache,
                indexed_indices=idx_chunk,
                indexed_lens=idx_lens_chunk,
                indexed_page_size=indexed_page_size,
                mode="extend",
            )
