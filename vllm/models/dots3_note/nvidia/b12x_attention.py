# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fail-closed B12X attention backends for Dots3 NOTE on SM120/SM121."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar, cast

import torch

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
    QueryLenSupport,
)
from vllm.model_executor.layers.attention.sparse_mla_attention import (
    SparseMLACommonImpl,
)
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionLayer,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.mla.flashattn_mla_sparse import (
    FlashAttnMLASparseBackend,
    FlashAttnMLASparseMetadata,
    FlashAttnMLASparseMetadataBuilder,
)
from vllm.v1.attention.backends.mla.prefill.base import (
    MLADimensions,
    MLAPrefillBackend,
)
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    is_workspace_manager_initialized,
)

_FP8 = torch.float8_e4m3fn
_BLOCK_SIZE = 64
_PHYSICAL_RECORD_WIDTH = 1088
_DSA_TOTAL_HEADS = 128
_DSA_LOCAL_HEADS = 16
_DSA_QK_DIM = 576
_DSA_VALUE_DIM = 512
_DSA_TOPK = 2048
_DSA_SCALE = 1.0 / math.sqrt(192)
_SWA_TOTAL_HEADS = 64
_SWA_LOCAL_HEADS = 8
_SWA_QK_DIM = 1088
_SWA_VALUE_DIM = 1024
_SWA_WINDOW = 513
_SWA_SCALE = 1.0 / math.sqrt(256)
_QUALIFIED_TP_SIZE = 8
_MAX_PHYSICAL_PAGES = torch.iinfo(torch.int32).max
_MAX_PHYSICAL_BLOCKS = torch.iinfo(torch.int32).max // _BLOCK_SIZE


def _load_strided_sparse_mla() -> Any:
    from b12x.attention.sparse_mla import strided

    return strided


def _load_dense_mla() -> Any:
    from b12x.attention import dense_mla

    return dense_mla


def _page_table_width(max_tokens: int) -> int:
    width = (int(max_tokens) + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    return ((width + 1) // 2) * 2


def _is_dots3_model(vllm_config: VllmConfig) -> bool:
    model_config = vllm_config.model_config
    if model_config is None:
        return False
    model_type = getattr(model_config.hf_text_config, "model_type", None)
    if model_type == "dots3_note":
        return True
    speculative_config = vllm_config.speculative_config
    target = (
        getattr(speculative_config, "target_model_config", None)
        if speculative_config is not None
        else None
    )
    target_type = (
        getattr(target.hf_text_config, "model_type", None)
        if target is not None
        else None
    )
    return model_type == "deepseek_mtp" and target_type == "dots3_note"


def _validate_b12x_hybrid_config(vllm_config: VllmConfig) -> str | None:
    if not _is_dots3_model(vllm_config):
        return "B12X_HYBRID_MLA supports only dots3_note and its MTP draft model"
    model_config = vllm_config.model_config
    assert model_config is not None
    config_model = model_config
    if getattr(model_config.hf_text_config, "model_type", None) == "deepseek_mtp":
        speculative_config = vllm_config.speculative_config
        target = (
            getattr(speculative_config, "target_model_config", None)
            if speculative_config is not None
            else None
        )
        if target is None:
            return "B12X_HYBRID_MLA MTP requires a dots3_note target model config"
        config_model = target
    config = config_model.hf_text_config
    geometry = (
        getattr(config, "num_attention_heads", None),
        getattr(config, "kv_lora_rank", None),
        getattr(config, "qk_nope_head_dim", None),
        getattr(config, "qk_rope_head_dim", None),
        getattr(config, "v_head_dim", None),
        getattr(config, "index_topk", None),
        getattr(config, "swa_num_attention_heads", None),
        getattr(config, "swa_kv_lora_rank", None),
        getattr(config, "swa_qk_nope_head_dim", None),
        getattr(config, "swa_qk_rope_head_dim", None),
        getattr(config, "swa_v_head_dim", None),
        getattr(config, "sliding_window_size", None),
    )
    required = (
        128,
        512,
        128,
        64,
        128,
        2048,
        64,
        1024,
        192,
        64,
        128,
        513,
    )
    if geometry != required:
        return f"B12X_HYBRID_MLA requires the exact DSA/SWA geometry, got {geometry}"
    parallel = vllm_config.parallel_config
    if parallel.tensor_parallel_size != _QUALIFIED_TP_SIZE:
        return (
            f"B12X_HYBRID_MLA is qualified only for TP{_QUALIFIED_TP_SIZE}, got "
            f"TP{parallel.tensor_parallel_size}"
        )
    if parallel.decode_context_parallel_size != 1:
        return "B12X_HYBRID_MLA does not support decode context parallelism"
    if parallel.prefill_context_parallel_size != 1:
        return "B12X_HYBRID_MLA does not support prefill context parallelism"
    cache_config = vllm_config.cache_config
    if cache_config.block_size != _BLOCK_SIZE:
        return (
            f"B12X_HYBRID_MLA requires block size {_BLOCK_SIZE}, got "
            f"{cache_config.block_size}"
        )
    if cache_config.cache_dtype not in ("fp8", "fp8_e4m3"):
        return (
            "B12X_HYBRID_MLA requires E4M3 FP8 KV cache storage, got "
            f"{cache_config.cache_dtype!r}"
        )
    if config_model.dtype != torch.bfloat16:
        return f"B12X_HYBRID_MLA requires BF16 activations, got {config_model.dtype}"
    return None


def _workspace_specs(
    plan: Any,
    *,
    max_rows: int,
    num_heads: int,
    qk_dim: int,
    value_dim: int,
) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
    return (
        *plan.shapes_and_dtypes(),
        ((max_rows, num_heads, qk_dim), torch.bfloat16),
        ((max_rows, num_heads, value_dim), torch.bfloat16),
    )


class _WorkspaceUser:
    _fallback_workspaces: dict[int, tuple[torch.Tensor, ...]]

    def _borrow_workspaces(
        self,
        plan: Any,
        *,
        device: torch.device,
        max_rows: int,
        num_heads: int,
        qk_dim: int,
        value_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        specs = _workspace_specs(
            plan,
            max_rows=max_rows,
            num_heads=num_heads,
            qk_dim=qk_dim,
            value_dim=value_dim,
        )
        buffers: list[torch.Tensor] | tuple[torch.Tensor, ...] | None
        if is_workspace_manager_initialized():
            buffers = current_workspace_manager().get_simultaneous(*specs)
        else:
            buffers = self._fallback_workspaces.get(id(plan))
            if buffers is None:
                buffers = tuple(
                    torch.empty(shape, dtype=dtype, device=device)
                    for shape, dtype in specs
                )
                self._fallback_workspaces[id(plan)] = buffers
        if len(buffers) != 3:
            raise RuntimeError(
                "B12X_HYBRID_MLA expected scratch, query, and output buffers"
            )
        return cast(tuple[torch.Tensor, torch.Tensor, torch.Tensor], buffers)

    @staticmethod
    def _copy_absorbed_query(
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        destination: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(q, tuple):
            q_nope, q_rope = q
            rows = int(q_nope.shape[0])
            result = destination[:rows]
            split = int(q_nope.shape[-1])
            result[..., :split].copy_(q_nope)
            result[..., split:].copy_(q_rope)
            return result
        rows = int(q.shape[0])
        result = destination[:rows]
        result.copy_(q)
        return result


@dataclass
class Dots3NoteB12XSparseMetadata(FlashAttnMLASparseMetadata):
    sparse_mla_plan: Any | None = None
    sparse_cu_seqlens_q: torch.Tensor | None = None


class Dots3NoteB12XSparseMetadataBuilder(FlashAttnMLASparseMetadataBuilder):
    metadata_cls = Dots3NoteB12XSparseMetadata
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    @staticmethod
    def determine_chunked_prefill_workspace_size(vllm_config: VllmConfig) -> int:
        del vllm_config
        return 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        max_rows = int(vllm_config.scheduler_config.max_num_batched_tokens)
        sparse_mla = _load_strided_sparse_mla()
        self._sparse_mla_plan = sparse_mla.plan(
            sparse_mla.Caps(
                device=device,
                num_q_heads=self.model_config.get_num_attention_heads(
                    vllm_config.parallel_config
                ),
                tp_size=vllm_config.parallel_config.tensor_parallel_size,
                max_q_rows=max_rows,
                num_cache_blocks=_MAX_PHYSICAL_BLOCKS,
                max_physical_records=_MAX_PHYSICAL_PAGES,
                block_size=kv_cache_spec.block_size,
                topk=self.topk_tokens,
                kv_dtype=_FP8,
                use_cuda_graph=True,
            )
        )
        self._sparse_cu_seqlens_q = torch.arange(
            max_rows + 1,
            dtype=torch.int32,
            device=device,
        )
        specs = _workspace_specs(
            self._sparse_mla_plan,
            max_rows=max_rows,
            num_heads=_DSA_LOCAL_HEADS,
            qk_dim=_DSA_QK_DIM,
            value_dim=_DSA_VALUE_DIM,
        )
        if is_workspace_manager_initialized():
            current_workspace_manager().get_simultaneous(*specs)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> Dots3NoteB12XSparseMetadata:
        metadata = cast(
            Dots3NoteB12XSparseMetadata,
            super().build(
                common_prefix_len,
                common_attn_metadata,
                fast_build=fast_build,
            ),
        )
        rows = int(metadata.num_actual_tokens)
        metadata.sparse_mla_plan = self._sparse_mla_plan
        metadata.sparse_cu_seqlens_q = self._sparse_cu_seqlens_q[: rows + 1]
        return metadata


class Dots3NoteB12XSparseImpl(
    _WorkspaceUser,
    SparseMLACommonImpl[Dots3NoteB12XSparseMetadata],
):
    supports_dense_mha_prefill: ClassVar[bool] = False
    can_return_lse_for_decode = True
    supports_dcp = False
    supports_pcp = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        **mla_args: Any,
    ) -> None:
        if any(
            value is not None
            for value in (alibi_slopes, sliding_window, logits_soft_cap)
        ):
            raise NotImplementedError(
                "B12X_HYBRID_MLA DSA does not support alibi or sliding windows"
            )
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "B12X_HYBRID_MLA supports decoder self-attention only"
            )
        if kv_sharing_target_layer_name is not None:
            raise NotImplementedError(
                "B12X_HYBRID_MLA does not support KV sharing aliases"
            )
        if kv_cache_dtype not in ("fp8", "fp8_e4m3"):
            raise ValueError("B12X_HYBRID_MLA DSA requires E4M3 FP8 KV cache")
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            **mla_args,
        )
        dims = (
            num_heads,
            head_size,
            self.kv_lora_rank,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
        )
        required = (_DSA_LOCAL_HEADS, 576, 512, 128, 64, 128)
        if dims != required or not math.isclose(
            scale, _DSA_SCALE, rel_tol=0, abs_tol=1e-12
        ):
            raise ValueError(
                f"B12X_HYBRID_MLA DSA received geometry/scale {dims}/{scale}; "
                f"required {required}/{_DSA_SCALE}"
            )
        if num_kv_heads != 1:
            raise ValueError("B12X_HYBRID_MLA DSA requires one compressed KV head")
        if self.topk_indices_buffer is None:
            raise ValueError(
                "B12X_HYBRID_MLA DSA requires the Dots3 indexer output buffer"
            )
        self.masked_mha_available = False
        self.supports_quant_query_input = False
        self._fallback_workspaces = {}
        self._compiled_plans: set[int] = set()

    @staticmethod
    def _logical_cache(kv_cache: torch.Tensor) -> torch.Tensor:
        if kv_cache.shape[-1] != _PHYSICAL_RECORD_WIDTH:
            raise ValueError(
                "B12X_HYBRID_MLA requires 1088-element physical cache records"
            )
        return kv_cache[..., :_DSA_QK_DIM]

    def do_kv_cache_update(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None:
        super().do_kv_cache_update(
            kv_c_normed,
            k_pe,
            self._logical_cache(kv_cache),
            slot_mapping,
            kv_cache_dtype,
            k_scale,
        )

    def forward_mha(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError(
            "B12X_HYBRID_MLA DSA prefill must use sparse compressed-cache extend"
        )

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_cache: torch.Tensor,
        attn_metadata: Dots3NoteB12XSparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        plan = attn_metadata.sparse_mla_plan
        cu_seqlens_q = attn_metadata.sparse_cu_seqlens_q
        if plan is None or cu_seqlens_q is None:
            raise RuntimeError(
                "B12X_HYBRID_MLA DSA metadata is missing its planned state"
            )
        rows = int(q[0].shape[0] if isinstance(q, tuple) else q.shape[0])
        scratch, q_storage, output_storage = self._borrow_workspaces(
            plan,
            device=kv_cache.device,
            max_rows=int(plan.caps.max_q_rows),
            num_heads=_DSA_LOCAL_HEADS,
            qk_dim=_DSA_QK_DIM,
            value_dim=_DSA_VALUE_DIM,
        )
        absorbed_q = self._copy_absorbed_query(q, q_storage)
        output = output_storage[:rows]
        assert self.topk_indices_buffer is not None
        sparse_mla = _load_strided_sparse_mla()
        binding = sparse_mla.bind_indexed(
            plan,
            scratch=scratch,
            q=absorbed_q,
            kv_cache=kv_cache,
            output=output,
            logical_indices=self.topk_indices_buffer[:rows],
            request_ids=attn_metadata.req_id_per_token[:rows],
            block_table=attn_metadata.block_table,
            cu_seqlens_q=cu_seqlens_q,
            kv_scale=layer._k_scale,
            q_scale=layer._q_scale,
        )
        plan_id = id(plan)
        if plan_id not in self._compiled_plans:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "B12X_HYBRID_MLA DSA compile miss during CUDA graph capture; "
                    "eager warmup did not exercise this plan"
                )
            sparse_mla.compile(binding=binding)
            self._compiled_plans.add(plan_id)
        if attn_metadata.num_prefills:
            return sparse_mla.run_extend(binding=binding)
        return sparse_mla.run_decode(binding=binding)


class B12xHybridMLABackend(FlashAttnMLASparseBackend):
    """User-selectable Dots3 backend; the model routes SWA internally."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["fp8", "fp8_e4m3"]

    @staticmethod
    def get_name() -> str:
        return "B12X_HYBRID_MLA"

    @staticmethod
    def get_impl_cls() -> type[Dots3NoteB12XSparseImpl]:
        return Dots3NoteB12XSparseImpl

    @staticmethod
    def get_builder_cls() -> type[Dots3NoteB12XSparseMetadataBuilder]:
        return Dots3NoteB12XSparseMetadataBuilder

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [_DSA_QK_DIM]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [_BLOCK_SIZE]

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        return block_size == _BLOCK_SIZE

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        return (1, 0, 2, 3) if include_num_layers_dimension else (0, 1, 2)

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 12 and capability.minor in (0, 1)

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        del use_mm_prefix
        if not cls.supports_compute_capability(device_capability):
            return "B12X_HYBRID_MLA requires SM120 or SM121"
        if not use_mla or not use_sparse:
            return "B12X_HYBRID_MLA requires Dots3 sparse MLA model selection"
        if has_sink:
            return "B12X_HYBRID_MLA does not support attention sinks"
        if head_size != _DSA_QK_DIM:
            return (
                f"B12X_HYBRID_MLA requires DSA head size {_DSA_QK_DIM}, got {head_size}"
            )
        if dtype != torch.bfloat16:
            return f"B12X_HYBRID_MLA requires BF16 activations, got {dtype}"
        if kv_cache_dtype not in ("fp8", "fp8_e4m3"):
            return "B12X_HYBRID_MLA requires E4M3 FP8 KV cache"
        try:
            sparse_mla = _load_strided_sparse_mla()
            dense_mla = _load_dense_mla()
        except (ImportError, AttributeError):
            return "B12X_HYBRID_MLA requires sparse_mla.strided and dense_mla APIs"
        if not sparse_mla.is_supported() or not dense_mla.is_supported():
            return "B12X_HYBRID_MLA kernels are unavailable on this device"
        vllm_config = get_current_vllm_config()
        configured_block_size = (
            block_size
            if block_size is not None
            else vllm_config.cache_config.block_size
        )
        if configured_block_size != _BLOCK_SIZE:
            return (
                f"B12X_HYBRID_MLA requires block size {_BLOCK_SIZE}, got "
                f"{configured_block_size}"
            )
        return _validate_b12x_hybrid_config(vllm_config)


class B12xHybridMLACompressedPrefillBackend(MLAPrefillBackend):
    """Sentinel backend: B12X SWA prefill is routed through compressed extend."""

    @staticmethod
    def get_name() -> str:
        return "B12X_HYBRID_MLA_COMPRESSED_EXTEND"

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 12 and capability.minor in (0, 1)

    @classmethod
    def supports_mla_dimensions(cls, dims: MLADimensions) -> bool:
        return dims == MLADimensions(
            qk_nope_head_dim=192,
            qk_rope_head_dim=64,
            v_head_dim=128,
        )

    def run_prefill_new_tokens(self, *args: Any, **kwargs: Any):
        raise RuntimeError(
            "B12X_HYBRID_MLA SWA prefill must use compressed-cache extend"
        )

    def run_prefill_context_chunk(self, *args: Any, **kwargs: Any):
        raise RuntimeError(
            "B12X_HYBRID_MLA SWA prefill must use compressed-cache extend"
        )


@dataclass
class Dots3NoteB12XSlidingMetadata(MLACommonMetadata):
    dense_mla_plan: Any | None = None


class Dots3NoteB12XSlidingMetadataBuilder(
    MLACommonMetadataBuilder[Dots3NoteB12XSlidingMetadata]
):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    query_len_support: ClassVar[QueryLenSupport] = QueryLenSupport.VARLEN

    @staticmethod
    def determine_chunked_prefill_workspace_size(vllm_config: VllmConfig) -> int:
        del vllm_config
        return 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(
            kv_cache_spec,
            layer_names,
            vllm_config,
            device,
            Dots3NoteB12XSlidingMetadata,
        )
        max_rows = int(vllm_config.scheduler_config.max_num_batched_tokens)
        max_batch = int(vllm_config.scheduler_config.max_num_seqs)
        self.reorder_batch_threshold = max_rows
        dense_mla = _load_dense_mla()
        self._dense_mla_plan = dense_mla.plan(
            dense_mla.Caps(
                device=device,
                mode="decode",
                dtype=torch.bfloat16,
                q_dtype=torch.bfloat16,
                kv_dtype=_FP8,
                num_q_heads=self.num_heads,
                page_size=kv_cache_spec.block_size,
                max_total_q=max_rows,
                max_batch=max_batch,
                max_cache_tokens=int(vllm_config.model_config.max_model_len),
                max_page_table_width=_page_table_width(
                    vllm_config.model_config.max_model_len
                ),
                num_cache_pages=_MAX_PHYSICAL_PAGES,
                head_dim=_SWA_QK_DIM,
                v_head_dim=_SWA_VALUE_DIM,
                physical_record_width=_PHYSICAL_RECORD_WIDTH,
                window_size=_SWA_WINDOW,
                use_cuda_graph=True,
            )
        )
        specs = _workspace_specs(
            self._dense_mla_plan,
            max_rows=max_rows,
            num_heads=_SWA_LOCAL_HEADS,
            qk_dim=_SWA_QK_DIM,
            value_dim=_SWA_VALUE_DIM,
        )
        if is_workspace_manager_initialized():
            current_workspace_manager().get_simultaneous(*specs)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> Dots3NoteB12XSlidingMetadata:
        metadata = cast(
            Dots3NoteB12XSlidingMetadata,
            super().build(
                common_prefix_len,
                common_attn_metadata,
                fast_build=fast_build,
            ),
        )
        if metadata.num_prefills:
            raise RuntimeError(
                "B12X_HYBRID_MLA SWA metadata escaped compressed-cache extend routing"
            )
        metadata.dense_mla_plan = self._dense_mla_plan
        return metadata


class Dots3NoteB12XSlidingImpl(
    _WorkspaceUser,
    MLACommonImpl[Dots3NoteB12XSlidingMetadata],
):
    can_return_lse_for_decode = True
    supports_dcp = False
    supports_pcp = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        **mla_args: Any,
    ) -> None:
        if alibi_slopes is not None or logits_soft_cap is not None:
            raise NotImplementedError(
                "B12X_HYBRID_MLA SWA does not support alibi or soft caps"
            )
        if sliding_window != _SWA_WINDOW:
            raise ValueError(
                f"B12X_HYBRID_MLA SWA requires a {_SWA_WINDOW}-token window, "
                f"got {sliding_window}"
            )
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "B12X_HYBRID_MLA supports decoder self-attention only"
            )
        if kv_cache_dtype not in ("fp8", "fp8_e4m3"):
            raise ValueError("B12X_HYBRID_MLA SWA requires E4M3 FP8 KV cache")
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            None,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            **mla_args,
        )
        dims = (
            num_heads,
            head_size,
            self.kv_lora_rank,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
        )
        required = (_SWA_LOCAL_HEADS, 1088, 1024, 192, 64, 128)
        if dims != required or not math.isclose(
            scale, _SWA_SCALE, rel_tol=0, abs_tol=1e-12
        ):
            raise ValueError(
                f"B12X_HYBRID_MLA SWA received geometry/scale {dims}/{scale}; "
                f"required {required}/{_SWA_SCALE}"
            )
        if num_kv_heads != 1:
            raise ValueError("B12X_HYBRID_MLA SWA requires one compressed KV head")
        self.sliding_window = _SWA_WINDOW
        self.supports_quant_query_input = False
        self._fallback_workspaces = {}
        self._compiled_plans: set[int] = set()

    def forward_mha(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError(
            "B12X_HYBRID_MLA SWA prefill must use compressed-cache extend"
        )

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_cache: torch.Tensor,
        attn_metadata: Dots3NoteB12XSlidingMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        plan = attn_metadata.dense_mla_plan
        decode = attn_metadata.decode
        if plan is None or decode is None:
            raise RuntimeError(
                "B12X_HYBRID_MLA SWA metadata is missing its planned state"
            )
        rows = int(q[0].shape[0] if isinstance(q, tuple) else q.shape[0])
        scratch, q_storage, output_storage = self._borrow_workspaces(
            plan,
            device=kv_cache.device,
            max_rows=int(plan.caps.max_total_q),
            num_heads=_SWA_LOCAL_HEADS,
            qk_dim=_SWA_QK_DIM,
            value_dim=_SWA_VALUE_DIM,
        )
        absorbed_q = self._copy_absorbed_query(q, q_storage)
        output = output_storage[:rows]
        batch = int(decode.seq_lens.shape[0])
        dense_mla = _load_dense_mla()
        binding = dense_mla.bind(
            plan,
            scratch=scratch,
            q=absorbed_q,
            kv_cache=kv_cache,
            output=output,
            page_table=decode.block_table,
            cache_seqlens=decode.seq_lens,
            cu_seqlens_q=attn_metadata.query_start_loc[: batch + 1],
            kv_scale=layer._k_scale,
            q_scale=layer._q_scale,
            sm_scale=_SWA_SCALE,
        )
        plan_id = id(plan)
        if plan_id not in self._compiled_plans:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "B12X_HYBRID_MLA SWA compile miss during CUDA graph capture; "
                    "eager warmup did not exercise this plan"
                )
            dense_mla.compile(binding=binding)
            self._compiled_plans.add(plan_id)
        return dense_mla.run(binding=binding)


class B12xHybridMLASlidingBackend(MLACommonBackend):
    """Internal SWA route selected by the Dots3 model under B12X_HYBRID_MLA."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["fp8", "fp8_e4m3"]

    @staticmethod
    def get_name() -> str:
        return "B12X_HYBRID_MLA_SWA"

    @staticmethod
    def get_impl_cls() -> type[Dots3NoteB12XSlidingImpl]:
        return Dots3NoteB12XSlidingImpl

    @staticmethod
    def get_builder_cls() -> type[Dots3NoteB12XSlidingMetadataBuilder]:
        return Dots3NoteB12XSlidingMetadataBuilder

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [_SWA_QK_DIM]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [_BLOCK_SIZE]

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        return block_size == _BLOCK_SIZE

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return True

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        return (1, 0, 2, 3) if include_num_layers_dimension else (0, 1, 2)

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 12 and capability.minor in (0, 1)


__all__ = [
    "B12xHybridMLABackend",
    "B12xHybridMLACompressedPrefillBackend",
    "B12xHybridMLASlidingBackend",
]
