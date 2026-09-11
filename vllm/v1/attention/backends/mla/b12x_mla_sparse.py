# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B12x sparse MLA attention backend."""

from dataclasses import dataclass, field, replace
from math import gcd, prod
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable
from weakref import WeakKeyDictionary

import numpy as np
import torch
import torch.distributed as dist

from vllm import _custom_ops as ops
from vllm import envs
from vllm.config import VllmConfig, get_current_vllm_config_or_none
from vllm.config.cache import CacheDType
from vllm.distributed import get_dcp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import MLACommonPrefillMetadata
from vllm.model_executor.layers.attention.sparse_mla_attention import (
    SparseMLACommonImpl,
    SparseMLACommonMetadataBuilder,
)
from vllm.platforms.interface import DeviceCapability
from vllm.triton_utils import tl, triton
from vllm.utils.b12x import get_b12x_sparse_mla
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionLayer,
    AttentionMetadata,
    AttentionType,
    MLAAttentionImpl,
    MultipleOf,
)
from vllm.v1.attention.backends.mla import b12x_topk_sort
from vllm.v1.attention.backends.mla.ckv_prefetch import (
    CKVPrefetchPlan,
    CKVPrefetchRegistry,
    CKVPrefetchState,
    CKVWorkspacePool,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
    triton_filter_and_convert_dcp_index,
)
from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens
from vllm.v1.kv_cache_interface import AttentionSpec, MLAAttentionSpec
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    is_workspace_manager_initialized,
)

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer
    from vllm.v1.attention.backend import CommonAttentionMetadata


_GLM_NEXT_MODEL_TYPES = frozenset(("glm5_next", "glm5_next_text"))
_GLM_DSA_MODEL_TYPES = frozenset(("glm_moe_dsa",))
_GLM_NEXT_CACHE_RECORD_BYTES = 528
_GLM_NEXT_NVFP4_CACHE_RECORD_BYTES = 304
_GLM_NEXT_INDEX_TAIL_BYTES_PER_TOKEN = 132 // 4
_GLM_NEXT_INDEX_PAGE_BYTES = 64 * 132
_GLM_DSA_NVFP4_CACHE_RECORD_BYTES = 368

logger = init_logger(__name__)


@dataclass
class _CKVReservation:
    registry: CKVPrefetchRegistry
    current_local: torch.Tensor
    current_gathered: torch.Tensor
    gather_streams: dict[tuple[int, int], torch.cuda.Stream] = field(
        default_factory=dict
    )
    collectives_warmed: bool = False


_CKV_RESERVATIONS: WeakKeyDictionary = WeakKeyDictionary()


@runtime_checkable
class B12xPhysicalSelectionProvider(Protocol):
    def get_b12x_physical_selection(
        self,
        *,
        num_tokens: int,
        num_prefills: int,
        num_decode_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | None: ...


def _is_glm_next_config(hf_config: object | None) -> bool:
    return getattr(hf_config, "model_type", None) in _GLM_NEXT_MODEL_TYPES


def _is_glm_dsa_config(hf_config: object | None) -> bool:
    return getattr(hf_config, "model_type", None) in _GLM_DSA_MODEL_TYPES


def _current_hf_text_config() -> object | None:
    vllm_config = get_current_vllm_config_or_none()
    if vllm_config is None or vllm_config.model_config is None:
        return None
    return vllm_config.model_config.hf_text_config


def _is_glm_next_spec(spec: AttentionSpec) -> bool:
    if isinstance(spec, MLAAttentionSpec) and spec.model_version == "glm5_next":
        return True
    hf_config = _current_hf_text_config()
    return hf_config is not None and _is_glm_next_config(hf_config)


def _is_glm_dsa_spec(spec: AttentionSpec) -> bool:
    if isinstance(spec, MLAAttentionSpec) and spec.model_version == "glm_moe_dsa":
        return True
    hf_config = _current_hf_text_config()
    return hf_config is not None and _is_glm_dsa_config(hf_config)


def _nvfp4_run_options(*, is_glm_next: bool) -> dict[str, bool | int]:
    if is_glm_next:
        # GLM5Next has no RoPE payload and stores one latent scale per token.
        return {
            "scale_format": 2,
            "fp8_rope": False,
            "latent_scale_per_token": True,
        }
    # GLM-5.2/5.3 DSA stores its 64-dimensional RoPE tail as E4M3.
    return {"scale_format": 2, "fp8_rope": True}


def _glm_next_recipe_error(hf_config: object) -> str | None:
    expected = {
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 256,
        "qk_rope_head_dim": 0,
        "v_head_dim": 256,
        "index_n_heads": 32,
        "index_head_dim": 128,
        "index_topk": 2048,
        "index_kpool": 4,
    }
    mismatches = [
        f"{name}={getattr(hf_config, name, None)!r} (expected {value})"
        for name, value in expected.items()
        if getattr(hf_config, name, None) != value
    ]
    if mismatches:
        return "B12X GLM5Next sparse MLA requires " + ", ".join(mismatches)
    return None


def _glm_next_dcp_error(vllm_config: VllmConfig) -> str | None:
    parallel_config = vllm_config.parallel_config
    dcp_size = int(parallel_config.decode_context_parallel_size)
    if dcp_size <= 1:
        return None
    interleave = int(parallel_config.cp_kv_cache_interleave_size)
    if interleave % 4:
        return (
            "B12X GLM5Next C4 DCP requires cp_kv_cache_interleave_size divisible by 4"
        )
    return None


def _selected_index_block_stride_rows(
    kv_cache: torch.Tensor,
    *,
    block_size: int,
) -> int:
    # B12X selected indices are physical token slots. The kernel applies the
    # cache's byte page stride when it addresses the selected page.
    if int(kv_cache.shape[1]) != block_size:
        raise ValueError(
            "B12X sparse MLA cache page size does not match attention metadata: "
            f"cache={int(kv_cache.shape[1])}, metadata={block_size}"
        )
    return block_size


def _ckv_lane_layer_counts(vllm_config: VllmConfig) -> tuple[int, int]:
    """Attention layer counts of the target lane and the drafter lane."""
    target_layers = max(
        1, int(getattr(vllm_config.model_config.hf_text_config, "num_hidden_layers", 1))
    )
    draft_layers = target_layers
    spec_config = vllm_config.speculative_config
    draft_model_config = getattr(spec_config, "draft_model_config", None)
    if draft_model_config is not None:
        draft_hf = draft_model_config.hf_text_config
        draft_layers = int(
            getattr(draft_hf, "num_nextn_predict_layers", 0)
            or getattr(draft_hf, "num_hidden_layers", 0)
            or 1
        )
    return target_layers, max(1, draft_layers)


def _max_speculative_decode_query_len(vllm_config: VllmConfig) -> int:
    spec_config = vllm_config.speculative_config
    if spec_config is None:
        return 1
    num_speculative_tokens = int(spec_config.num_speculative_tokens or 0)
    multiplier = 2 if spec_config.parallel_drafting else 1
    return 1 + multiplier * num_speculative_tokens


def _is_speculative_decode_batch(
    common_attn_metadata: "CommonAttentionMetadata",
    max_speculative_decode_query_len: int,
) -> bool:
    is_prefilling = getattr(common_attn_metadata, "is_prefilling", None)
    return (
        max_speculative_decode_query_len > 1
        and 1 < common_attn_metadata.max_query_len <= max_speculative_decode_query_len
        and is_prefilling is not None
        and not bool(torch.any(is_prefilling[: common_attn_metadata.num_reqs]))
    )


def _is_native_ckv_source_layout(
    kv_cache: torch.Tensor,
    *,
    page_size: int,
    record_bytes: int,
) -> bool:
    return (
        kv_cache.dtype == torch.uint8
        and kv_cache.ndim == 3
        and tuple(kv_cache.shape[1:]) == (page_size, record_bytes)
        and kv_cache.stride(1) == record_bytes
        and kv_cache.stride(2) == 1
    )


def _use_b12x_full_ckv_gather(
    *,
    enabled: bool,
    is_glm_next: bool,
    dcp_world_size: int,
    max_query_len: int,
    num_tokens: int,
    num_decode_tokens: int,
    min_tokens: int,
    max_tokens: int,
    is_glm_dsa: bool = False,
    is_spec_decode: bool = False,
) -> bool:
    return (
        enabled
        and (is_glm_next or is_glm_dsa)
        and not is_spec_decode
        and dcp_world_size > 1
        and max_query_len > 1
        and num_decode_tokens == 0
        and num_tokens > min_tokens
        and num_tokens <= max_tokens
    )


def _ckv_rank_token_alignment(page_size: int, dcp_world_size: int) -> int:
    """Per-rank padding that makes the gathered CKV page-addressable.

    All-gather concatenates one equally sized token span from every DCP rank.
    Individual rank boundaries do not need to coincide with KV-page boundaries;
    only the concatenated span must contain an integral number of pages.
    """
    if page_size <= 0 or dcp_world_size <= 0:
        raise ValueError("CKV page size and DCP world size must be positive")
    return page_size // gcd(page_size, dcp_world_size)


def _round_up_ckv_rank_tokens(
    token_count: int,
    *,
    page_size: int,
    dcp_world_size: int,
) -> int:
    alignment = _ckv_rank_token_alignment(page_size, dcp_world_size)
    return (token_count + alignment - 1) // alignment * alignment


def _ckv_collective_warmup_sizes(
    capacity: int, *, page_size: int, dcp_world_size: int
) -> tuple[int, ...]:
    """Cover aligned geometric message sizes and the full reserved rank span."""
    alignment = _ckv_rank_token_alignment(page_size, dcp_world_size)
    if capacity < alignment or capacity % alignment:
        raise ValueError("CKV collective capacity must be positive and page-aligned")
    sizes = []
    count = alignment
    while count < capacity:
        sizes.append(count)
        count *= 2
    sizes.append(capacity)
    return tuple(sizes)


def _dcp_all_gather_current_stream(
    group,
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
) -> None:
    if not input_tensor.is_contiguous() or not output_tensor.is_contiguous():
        raise ValueError("CKV all-gather tensors must be contiguous")
    if output_tensor.numel() != input_tensor.numel() * group.world_size:
        raise ValueError("CKV all-gather tensors have incompatible sizes")

    communicator = getattr(group, "device_communicator", None)
    pynccl_comm = getattr(communicator, "pynccl_comm", None)
    if pynccl_comm is not None and not getattr(pynccl_comm, "disabled", False):
        pynccl_comm.all_gather(output_tensor, input_tensor)
        return

    device_group = getattr(group, "device_group", None)
    if device_group is None:
        device_group = getattr(communicator, "device_group", None)
    if device_group is not None:
        dist.all_gather_into_tensor(
            output_tensor,
            input_tensor,
            group=device_group,
            async_op=False,
        )
        return

    output_tensor.copy_(group.all_gather(input_tensor, dim=0))


@triton.jit
def _mask_page_table_after_nsa_len_kernel(
    page_table_ptr,
    nsa_len_ptr,
    page_stride0,
    page_stride1,
    width: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    offs = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    valid = offs < width
    nsa_len = tl.load(nsa_len_ptr + row)
    tl.store(
        page_table_ptr + row * page_stride0 + offs * page_stride1,
        -1,
        mask=valid & (offs >= nsa_len),
    )


def _mask_page_table_after_nsa_len(
    page_table: torch.Tensor,
    nsa_cache_seqlens: torch.Tensor,
) -> None:
    width = page_table.shape[1]
    if width == 0 or page_table.shape[0] == 0:
        return
    block_n = 128
    _mask_page_table_after_nsa_len_kernel[
        (page_table.shape[0], triton.cdiv(width, block_n))
    ](
        page_table,
        nsa_cache_seqlens,
        page_table.stride(0),
        page_table.stride(1),
        width,
        BLOCK_N=block_n,
    )


def _global_causal_lens_for_ckv_gather(
    global_seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    req_id_per_token: torch.Tensor,
    num_actual_tokens: int,
) -> torch.Tensor:
    """Compute each query token's causal length in the gathered global cache.

    Args:
        global_seq_lens: Global cache sequence length for each request.
        query_start_loc: Start offset of each request's query chunk.
        req_id_per_token: Request identifier for each query token.
        num_actual_tokens: Number of active query tokens.

    Returns:
        Causal cache length for each active query token.
    """
    num_reqs = global_seq_lens.shape[0]
    qsl = query_start_loc[: num_reqs + 1].to(torch.int32)
    req_ids = req_id_per_token[:num_actual_tokens].to(torch.int64)
    chunk_start = qsl[:-1][req_ids]
    chunk_len = (qsl[1:] - qsl[:-1])[req_ids]
    full_seq = global_seq_lens[req_ids].to(torch.int32)
    token_idx = torch.arange(
        num_actual_tokens,
        device=global_seq_lens.device,
        dtype=torch.int32,
    )
    return full_seq - chunk_len + (token_idx - chunk_start) + 1


def _map_global_topk_to_gathered_ckv(
    req_ids: torch.Tensor,
    token_indices: torch.Tensor,
    rank_req_starts: torch.Tensor,
    rank_req_lens: torch.Tensor,
    out: torch.Tensor,
    valid_counts: torch.Tensor,
    *,
    dcp_size: int,
    cp_kv_cache_interleave_size: int,
    padded_rank_tokens: int,
) -> None:
    if token_indices.shape != out.shape:
        raise ValueError("CKV gather index output shape does not match top-k input")
    if rank_req_starts.shape != rank_req_lens.shape:
        raise ValueError("CKV gather request starts/lens shapes do not match")
    if rank_req_starts.shape[0] != dcp_size:
        raise ValueError("CKV gather request metadata does not match DCP size")
    if any(
        tensor.dtype != torch.int32
        for tensor in (
            req_ids,
            token_indices,
            rank_req_starts,
            rank_req_lens,
            out,
            valid_counts,
        )
    ):
        raise TypeError("CKV gather index metadata must be int32")

    from b12x.attention._shared.mla.dcp_ckv_mapping import (
        map_global_topk_to_gathered_ckv,
    )

    map_global_topk_to_gathered_ckv(
        req_ids,
        token_indices,
        rank_req_starts,
        rank_req_lens,
        out,
        valid_counts,
        dcp_size=dcp_size,
        cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
        padded_rank_tokens=padded_rank_tokens,
    )


class B12xMLASparseBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "fp8",
        "fp8_e4m3",
        "fp8_ds_mla",
        "nvfp4_ds_mla",
    ]

    @staticmethod
    def get_name() -> str:
        return "B12X"

    @staticmethod
    def get_impl_cls() -> type[MLAAttentionImpl]:
        return B12xMLASparseImpl

    @staticmethod
    def get_builder_cls() -> type["B12xMLASparseMetadataBuilder"]:
        return B12xMLASparseMetadataBuilder

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [512, 576]

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        is_glm_next = _is_glm_next_spec(spec)
        is_glm_dsa_nvfp4 = isinstance(spec, MLAAttentionSpec) and (
            spec.cache_dtype_str == "nvfp4_ds_mla" and _is_glm_dsa_spec(spec)
        )
        if not is_glm_next and not is_glm_dsa_nvfp4:
            return spec
        if not isinstance(spec, MLAAttentionSpec):
            raise TypeError(
                "B12X GLM sparse MLA requires an MLAAttentionSpec, got "
                f"{type(spec).__name__}."
            )
        if is_glm_dsa_nvfp4:
            if spec.head_size != 576:
                raise ValueError(
                    "B12X GLM DSA NVFP4 sparse MLA requires head_size=576, got "
                    f"{spec.head_size}."
                )
            return replace(
                spec,
                state_content_bytes=_GLM_DSA_NVFP4_CACHE_RECORD_BYTES,
                model_version="glm_moe_dsa",
            )
        if spec.head_size != 512:
            raise ValueError(
                "B12X GLM5Next sparse MLA requires head_size=512, got "
                f"{spec.head_size}."
            )
        record_bytes = (
            _GLM_NEXT_NVFP4_CACHE_RECORD_BYTES
            if spec.cache_dtype_str == "nvfp4_ds_mla"
            else _GLM_NEXT_CACHE_RECORD_BYTES
        )
        return replace(
            spec,
            state_content_bytes=record_bytes,
            alignment=_GLM_NEXT_INDEX_PAGE_BYTES,
            page_tail_bytes_per_token=_GLM_NEXT_INDEX_TAIL_BYTES_PER_TOKEN,
            model_version="glm5_next",
        )

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        # The DSA kernels consume one layer-compact, token-major cache view.
        return (KVCacheLayout.LBNHC,)

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [64]

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls) -> bool:
        return False

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return (capability.major, capability.minor) in ((12, 0), (12, 1))

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
        from vllm.config import get_current_vllm_config

        module = get_b12x_sparse_mla()
        if module is None:
            return "B12X sparse MLA requires the optional b12x package"
        vllm_config = get_current_vllm_config()
        if vllm_config.model_config is not None:
            hf_config = vllm_config.model_config.hf_text_config
            if getattr(hf_config, "index_topk", None) is None:
                return "B12X sparse MLA requires a model with index_topk"
            if _is_glm_next_config(hf_config):
                if recipe_error := _glm_next_recipe_error(hf_config):
                    return recipe_error
                if head_size != 512:
                    return "B12X GLM5Next sparse MLA requires head_size=512"
                if dcp_error := _glm_next_dcp_error(vllm_config):
                    return dcp_error
                return None
            if kv_cache_dtype == "nvfp4_ds_mla" and not _is_glm_dsa_config(hf_config):
                return (
                    "B12X nvfp4_ds_mla requires GLM5Next or the "
                    "GLM-5.2/5.3 DSA architecture"
                )
            if head_size != 576:
                return "B12X sparse MLA requires head_size=576"
            if int(getattr(hf_config, "kv_lora_rank", 0)) != 512:
                return "B12X sparse MLA requires kv_lora_rank=512"
            if int(getattr(hf_config, "qk_rope_head_dim", 0)) != 64:
                return "B12X sparse MLA requires qk_rope_head_dim=64"
        return None


class B12xGLMDSAMLASparseBackend(B12xMLASparseBackend):
    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        if not isinstance(spec, MLAAttentionSpec):
            raise TypeError(
                "B12X GLM DSA sparse MLA requires an MLAAttentionSpec, got "
                f"{type(spec).__name__}."
            )
        return super().customize_spec(replace(spec, model_version="glm_moe_dsa"))


class B12xGLM5NextMLASparseBackend(B12xMLASparseBackend):
    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls) -> bool:
        return True

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        # GLM-Next's pooled index tail shares manager blocks with its MLA
        # latent page. Keep the layer dimension inside the manager block so
        # copies and swaps carry both cache regions together.
        return (KVCacheLayout.BLHNC,)

    @staticmethod
    def get_builder_cls() -> type["B12xGLM5NextMLASparseMetadataBuilder"]:
        return B12xGLM5NextMLASparseMetadataBuilder

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        if not isinstance(spec, MLAAttentionSpec):
            raise TypeError(
                "B12X GLM5Next sparse MLA requires an MLAAttentionSpec, got "
                f"{type(spec).__name__}."
            )
        return super().customize_spec(replace(spec, model_version="glm5_next"))

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # Keep the hybrid manager page intact so its FP8 pooled-index tail is
        # copied and recycled with the corresponding MLA page.
        return [MultipleOf(64)]


@dataclass
class B12xMLASparseMetadata(AttentionMetadata):
    num_reqs: int
    max_query_len: int
    max_seq_len: int
    num_actual_tokens: int
    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    req_id_per_token: torch.Tensor
    seq_lens: torch.Tensor
    num_decodes: int
    num_prefills: int
    num_decode_tokens: int
    prefill_max_seq_len: int = 0
    prefill: MLACommonPrefillMetadata | None = None
    prefill_query_lens_cpu: torch.Tensor | None = None
    prefill_seq_lens_cpu: torch.Tensor | None = None
    block_size: int = 64
    topk_tokens: int = 2048
    cp_kv_cache_interleave_size: int = 1
    cache_seq_lens_per_token: torch.Tensor | None = None
    dcp_combine_seq_lens: torch.Tensor | None = None
    dcp_combine_query_start_loc: torch.Tensor | None = None
    selector_state_slot_ids: torch.Tensor | None = None
    selector_state_is_fresh: torch.Tensor | None = None
    selector_num_accepted_tokens: torch.Tensor | None = None
    selector_is_prefilling: torch.Tensor | None = None
    is_spec_decode: bool = False
    ckv_selected_indices: torch.Tensor | None = None
    ckv_active_counts: torch.Tensor | None = None
    dcp_rank_req_starts: torch.Tensor | None = None
    dcp_rank_req_lens: torch.Tensor | None = None
    dcp_local_cu_seq_lens: torch.Tensor | None = None
    global_cache_seq_lens_per_req: torch.Tensor | None = None
    dcp_local_total_tokens: int = 0
    dcp_padded_total_tokens: int = 0
    dcp_ckv_gather_eligible: bool = False
    # False only when CPU sequence bounds prove every prefill has zero history.
    ckv_has_history: bool = True


class B12xMLASparseMetadataBuilder(
    SparseMLACommonMetadataBuilder[B12xMLASparseMetadata]
):
    metadata_cls = B12xMLASparseMetadata
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    requires_glm_next_selector_metadata: bool

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        hf_config = vllm_config.model_config.hf_text_config
        self.requires_glm_next_selector_metadata = _is_glm_next_config(hf_config)
        self._is_glm_dsa = _is_glm_dsa_config(hf_config)
        if self.requires_glm_next_selector_metadata and (
            dcp_error := _glm_next_dcp_error(vllm_config)
        ):
            raise ValueError(dcp_error)
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.supports_draft_decode_metadata_update = (
            self.requires_glm_next_selector_metadata
        )
        self.dcp_rank = get_dcp_group().rank_in_group if self.dcp_world_size > 1 else 0
        scheduler_config = vllm_config.scheduler_config
        max_tokens = scheduler_config.max_num_batched_tokens
        max_reqs = int(scheduler_config.max_num_seqs)
        self._ckv_max_reqs = max_reqs
        self.cache_seq_lens_per_token_buffer = torch.empty(
            (max_tokens,), dtype=torch.int32, device=device
        )
        if self.dcp_world_size > 1 and device.type == "cuda":
            from b12x.comm.pcie._dcp_attention_metadata import (
                precompile_dcp_sequence_lengths,
            )

            precompile_dcp_sequence_lengths(
                self.cache_seq_lens_per_token_buffer.device.index
            )
        self.dcp_combine_query_start_loc_buffer = (
            torch.arange(max_tokens + 1, dtype=torch.int32, device=device)
            if self.dcp_world_size > 1
            else None
        )
        if self.requires_glm_next_selector_metadata:
            self._capture_default_state_slot_ids = torch.arange(
                max_reqs, dtype=torch.int32, device=device
            )
            self._capture_state_slot_ids = torch.empty(
                max_reqs, dtype=torch.int32, device=device
            )
            self._capture_state_is_fresh = torch.ones(
                max_reqs, dtype=torch.bool, device=device
            )
            self._capture_num_accepted_tokens = torch.ones(
                max_reqs, dtype=torch.int32, device=device
            )
            self._capture_is_prefilling = torch.zeros(
                max_reqs, dtype=torch.bool, device=device
            )
        num_q_heads = vllm_config.model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )
        self._ckv_gather_requested = (
            (self.requires_glm_next_selector_metadata or self._is_glm_dsa)
            and self.dcp_world_size > 1
            and envs.VLLM_B12X_MLA_CKV_GATHER
            and (
                self.requires_glm_next_selector_metadata
                or (num_q_heads % 8 == 0 and not vllm_config.parallel_config.enable_dbo)
            )
        )
        if self._ckv_gather_requested:
            hf_config = vllm_config.model_config.hf_text_config
            ckv_topk_tokens = int(hf_config.index_topk)
            if self.requires_glm_next_selector_metadata:
                ckv_topk_tokens += int(hf_config.index_kpool) - 1
            self.ckv_selected_indices_buffer = torch.empty(
                (max_tokens, ckv_topk_tokens), dtype=torch.int32, device=device
            )
            self.ckv_active_counts_buffer = torch.empty(
                (max_tokens,), dtype=torch.int32, device=device
            )
            self.dcp_rank_req_lens_buffer = torch.empty(
                (self.dcp_world_size, max_reqs), dtype=torch.int32, device=device
            )
            self.dcp_rank_req_starts_buffer = torch.empty(
                (self.dcp_world_size, max_reqs), dtype=torch.int32, device=device
            )
            self.dcp_local_cu_seq_lens_buffer = torch.empty(
                (max_reqs + 1,), dtype=torch.int32, device=device
            )
        else:
            self.ckv_selected_indices_buffer = None
            self.ckv_active_counts_buffer = None
            self.dcp_rank_req_lens_buffer = None
            self.dcp_rank_req_starts_buffer = None
            self.dcp_local_cu_seq_lens_buffer = None
        threshold = {8: 128, 16: 128, 32: 128, 64: 256, 128: 1024}.get(
            num_q_heads, 1024
        )
        self._init_reorder_batch_threshold(
            threshold,
            supports_spec_as_decode=True,
            supports_dcp_with_varlen=True,
        )
        self._max_speculative_decode_query_len = _max_speculative_decode_query_len(
            vllm_config
        )

    def _stage_glm_next_selector_metadata(
        self,
        *,
        num_reqs: int,
        for_cudagraph_capture: bool,
        selector_state_slot_ids: torch.Tensor | None,
        selector_state_is_fresh: torch.Tensor | None,
        selector_num_accepted_tokens: torch.Tensor | None,
        selector_is_prefilling: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        values = (
            selector_state_slot_ids,
            selector_state_is_fresh,
            selector_num_accepted_tokens,
            selector_is_prefilling,
        )
        if not self.requires_glm_next_selector_metadata:
            if any(value is not None for value in values):
                raise TypeError(
                    "GLM5Next selector metadata was provided to a non-GLM "
                    "B12X sparse MLA builder"
                )
            return (None, None, None, None)

        capacity = int(self._capture_state_slot_ids.numel())
        if not 0 <= num_reqs <= capacity:
            raise ValueError(
                "GLM5Next selector request count exceeds the metadata buffer "
                f"capacity: num_reqs={num_reqs}, capacity={capacity}"
            )
        if not for_cudagraph_capture and any(value is None for value in values):
            raise RuntimeError(
                "B12X GLM5Next sparse MLA requires selector state slots, fresh "
                "flags, accepted-token counts, and prefill flags"
            )

        if for_cudagraph_capture:
            self._capture_state_slot_ids[:num_reqs].copy_(
                self._capture_default_state_slot_ids[:num_reqs]
            )
            self._capture_state_is_fresh[:num_reqs].fill_(True)
            self._capture_num_accepted_tokens[:num_reqs].fill_(1)
            self._capture_is_prefilling[:num_reqs].fill_(False)
        else:
            typed_values = tuple(value for value in values if value is not None)
            if any(
                value.ndim != 1 or value.numel() < num_reqs for value in typed_values
            ):
                raise ValueError(
                    "GLM5Next selector metadata must be one-dimensional and "
                    "cover every padded request row"
                )
            self._capture_state_slot_ids[:num_reqs].fill_(-1)
            self._capture_state_is_fresh[:num_reqs].fill_(True)
            self._capture_num_accepted_tokens[:num_reqs].fill_(1)
            self._capture_is_prefilling[:num_reqs].fill_(False)
            assert selector_state_slot_ids is not None
            assert selector_state_is_fresh is not None
            assert selector_num_accepted_tokens is not None
            assert selector_is_prefilling is not None
            self._capture_state_slot_ids[:num_reqs].copy_(
                selector_state_slot_ids[:num_reqs]
            )
            self._capture_state_is_fresh[:num_reqs].copy_(
                selector_state_is_fresh[:num_reqs]
            )
            self._capture_num_accepted_tokens[:num_reqs].copy_(
                selector_num_accepted_tokens[:num_reqs]
            )
            self._capture_is_prefilling[:num_reqs].copy_(
                selector_is_prefilling[:num_reqs]
            )

        return (
            self._capture_state_slot_ids[:num_reqs],
            self._capture_state_is_fresh[:num_reqs],
            self._capture_num_accepted_tokens[:num_reqs],
            self._capture_is_prefilling[:num_reqs],
        )

    def _build(
        self,
        common_prefix_len: int,
        common_attn_metadata: "CommonAttentionMetadata",
        fast_build: bool = False,
        *,
        for_cudagraph_capture: bool,
        selector_state_slot_ids: torch.Tensor | None = None,
        selector_state_is_fresh: torch.Tensor | None = None,
        selector_num_accepted_tokens: torch.Tensor | None = None,
        selector_is_prefilling: torch.Tensor | None = None,
    ) -> B12xMLASparseMetadata:
        if (
            self._ckv_gather_requested
            and not for_cudagraph_capture
            and self.cache_seq_lens_per_token_buffer.is_cuda
        ):
            B12xMLASparseImpl.begin_ckv_prefetch_step()
        metadata = super().build(
            common_prefix_len, common_attn_metadata, fast_build=fast_build
        )
        common = common_attn_metadata
        num_tokens = common.num_actual_tokens
        use_dcp = self.dcp_world_size > 1
        seq_lens = common.seq_lens
        if use_dcp:
            seq_lens = common.dcp_local_seq_lens
            if seq_lens is None:
                seq_lens = get_dcp_local_seq_lens(
                    common.seq_lens,
                    self.dcp_world_size,
                    self.dcp_rank,
                    self.cp_kv_cache_interleave_size,
                )
        metadata.seq_lens = seq_lens

        if self.supports_varlen_decode_cudagraph:
            per_token_lens = self.cache_seq_lens_per_token_buffer[:num_tokens]
        elif (
            common.max_query_len <= 1
            and num_tokens == common.num_reqs
            and (not use_dcp or common.positions is None)
        ):
            per_token_lens = seq_lens[:num_tokens]
            if use_dcp:
                self.cache_seq_lens_per_token_buffer[:num_tokens].copy_(per_token_lens)
                per_token_lens = self.cache_seq_lens_per_token_buffer[:num_tokens]
        elif common.positions is not None:
            # The decode kernel binds these lengths, so they must live in the
            # builder's buffer: a FULL CUDA graph captures the buffer address
            # and replays against whatever a later build wrote there.
            per_token_lens = self.cache_seq_lens_per_token_buffer[:num_tokens]
            per_token_lens.copy_(common.positions[:num_tokens])
            per_token_lens += 1
            if use_dcp:
                if per_token_lens.is_cuda:
                    from b12x.comm.pcie._dcp_attention_metadata import (
                        localize_dcp_sequence_lengths,
                    )

                    localize_dcp_sequence_lengths(
                        per_token_lens,
                        per_token_lens,
                        self.dcp_world_size,
                        self.dcp_rank,
                        self.cp_kv_cache_interleave_size,
                    )
                else:
                    per_token_lens.copy_(
                        get_dcp_local_seq_lens(
                            per_token_lens,
                            self.dcp_world_size,
                            self.dcp_rank,
                            self.cp_kv_cache_interleave_size,
                        )
                    )
        else:
            starts = np.asarray(common.query_start_loc_cpu, dtype=np.int32)
            query_lens = np.diff(starts)
            seq_lens_cpu_source = (
                common.seq_lens_cpu_upper_bound
                if common.seq_lens_cpu_upper_bound is not None
                else common.seq_lens_cpu
            )
            seq_lens_cpu = seq_lens_cpu_source.numpy().astype(np.int32, copy=False)
            host_lens = np.zeros((num_tokens,), dtype=np.int32)
            for req_id, query_len in enumerate(query_lens):
                if query_len <= 0:
                    continue
                start = int(starts[req_id])
                end = int(starts[req_id + 1])
                context_len = int(seq_lens_cpu[req_id]) - int(query_len)
                request_lens = torch.arange(
                    context_len + 1,
                    context_len + int(query_len) + 1,
                    dtype=torch.int32,
                )
                if use_dcp:
                    request_lens = get_dcp_local_seq_lens(
                        request_lens,
                        self.dcp_world_size,
                        self.dcp_rank,
                        self.cp_kv_cache_interleave_size,
                    )
                host_lens[start:end] = request_lens.numpy()
            host_tensor = torch.from_numpy(host_lens).pin_memory()
            self.cache_seq_lens_per_token_buffer[:num_tokens].copy_(
                host_tensor, non_blocking=True
            )
            per_token_lens = self.cache_seq_lens_per_token_buffer[:num_tokens]

        metadata.cache_seq_lens_per_token = per_token_lens
        if use_dcp:
            # Each causal query row can have a different empty-shard mask,
            # including rows belonging to the same speculative request.
            assert self.dcp_combine_query_start_loc_buffer is not None
            metadata.dcp_combine_seq_lens = per_token_lens
            metadata.dcp_combine_query_start_loc = (
                self.dcp_combine_query_start_loc_buffer[: num_tokens + 1]
            )
        metadata.is_spec_decode = _is_speculative_decode_batch(
            common,
            self._max_speculative_decode_query_len,
        )
        if metadata.num_prefills:
            prefill_start = metadata.num_decodes
            prefill_end = prefill_start + metadata.num_prefills + 1
            metadata.prefill_query_lens_cpu = torch.diff(
                common.query_start_loc_cpu[prefill_start:prefill_end]
            )
            seq_lens_cpu_source = (
                common.seq_lens_cpu_upper_bound
                if common.seq_lens_cpu_upper_bound is not None
                else common.seq_lens_cpu
            )
            metadata.prefill_seq_lens_cpu = seq_lens_cpu_source[
                prefill_start : prefill_start + metadata.num_prefills
            ].clone()
            if use_dcp and self._ckv_gather_requested:
                metadata.ckv_has_history = not torch.equal(
                    metadata.prefill_seq_lens_cpu, metadata.prefill_query_lens_cpu
                )
        if _use_b12x_full_ckv_gather(
            enabled=self._ckv_gather_requested and not for_cudagraph_capture,
            is_glm_next=self.requires_glm_next_selector_metadata,
            is_glm_dsa=getattr(self, "_is_glm_dsa", False),
            is_spec_decode=metadata.is_spec_decode,
            dcp_world_size=self.dcp_world_size,
            max_query_len=common.max_query_len,
            num_tokens=num_tokens,
            num_decode_tokens=metadata.num_decode_tokens,
            min_tokens=envs.VLLM_B12X_MLA_CKV_GATHER_MIN_TOKENS,
            max_tokens=envs.VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS,
        ):
            assert self.ckv_selected_indices_buffer is not None
            assert self.ckv_active_counts_buffer is not None
            assert self.dcp_rank_req_lens_buffer is not None
            assert self.dcp_rank_req_starts_buffer is not None
            assert self.dcp_local_cu_seq_lens_buffer is not None
            global_seq_lens = common.seq_lens[: common.num_reqs]
            all_rank_lens = get_dcp_local_seq_lens(
                global_seq_lens,
                self.dcp_world_size,
                dcp_rank=None,
                cp_kv_cache_interleave_size=self.cp_kv_cache_interleave_size,
            ).transpose(0, 1)
            rank_req_lens = self.dcp_rank_req_lens_buffer[
                : self.dcp_world_size, : common.num_reqs
            ]
            rank_req_lens.copy_(all_rank_lens)
            rank_req_starts = self.dcp_rank_req_starts_buffer[
                : self.dcp_world_size, : common.num_reqs
            ]
            rank_req_starts[:, 0].zero_()
            if common.num_reqs > 1:
                torch.cumsum(rank_req_lens[:, :-1], dim=1, out=rank_req_starts[:, 1:])
            local_cu_seq_lens = self.dcp_local_cu_seq_lens_buffer[: common.num_reqs + 1]
            local_cu_seq_lens[0].zero_()
            torch.cumsum(
                rank_req_lens[self.dcp_rank],
                dim=0,
                out=local_cu_seq_lens[1:],
            )
            rank_totals = rank_req_lens.sum(dim=1).tolist()
            local_total_tokens = int(rank_totals[self.dcp_rank])
            page_size = int(self.kv_cache_spec.block_size)
            padded_total_tokens = _round_up_ckv_rank_tokens(
                max(int(total) for total in rank_totals),
                page_size=page_size,
                dcp_world_size=self.dcp_world_size,
            )
            max_local_capacity = _round_up_ckv_rank_tokens(
                (envs.VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS + self.dcp_world_size - 1)
                // self.dcp_world_size
                + self._ckv_max_reqs * self.cp_kv_cache_interleave_size,
                page_size=page_size,
                dcp_world_size=self.dcp_world_size,
            )
            if 0 < padded_total_tokens <= max_local_capacity:
                metadata.ckv_selected_indices = self.ckv_selected_indices_buffer[
                    :num_tokens
                ]
                metadata.ckv_active_counts = self.ckv_active_counts_buffer[:num_tokens]
                metadata.dcp_rank_req_lens = rank_req_lens
                metadata.dcp_rank_req_starts = rank_req_starts
                metadata.dcp_local_cu_seq_lens = local_cu_seq_lens
                metadata.global_cache_seq_lens_per_req = global_seq_lens
                metadata.dcp_local_total_tokens = local_total_tokens
                metadata.dcp_padded_total_tokens = padded_total_tokens
                metadata.dcp_ckv_gather_eligible = True
        (
            metadata.selector_state_slot_ids,
            metadata.selector_state_is_fresh,
            metadata.selector_num_accepted_tokens,
            metadata.selector_is_prefilling,
        ) = self._stage_glm_next_selector_metadata(
            num_reqs=common.num_reqs,
            for_cudagraph_capture=for_cudagraph_capture,
            selector_state_slot_ids=selector_state_slot_ids,
            selector_state_is_fresh=selector_state_is_fresh,
            selector_num_accepted_tokens=selector_num_accepted_tokens,
            selector_is_prefilling=selector_is_prefilling,
        )
        return metadata

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: "CommonAttentionMetadata",
        fast_build: bool = False,
        selector_state_slot_ids: torch.Tensor | None = None,
        selector_state_is_fresh: torch.Tensor | None = None,
        selector_num_accepted_tokens: torch.Tensor | None = None,
        selector_is_prefilling: torch.Tensor | None = None,
    ) -> B12xMLASparseMetadata:
        return self._build(
            common_prefix_len,
            common_attn_metadata,
            fast_build,
            for_cudagraph_capture=False,
            selector_state_slot_ids=selector_state_slot_ids,
            selector_state_is_fresh=selector_state_is_fresh,
            selector_num_accepted_tokens=selector_num_accepted_tokens,
            selector_is_prefilling=selector_is_prefilling,
        )

    def build_for_cudagraph_capture(
        self,
        common_attn_metadata: "CommonAttentionMetadata",
    ) -> B12xMLASparseMetadata:
        return self._build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
            for_cudagraph_capture=True,
        )

    def update_draft_decode_metadata(
        self,
        metadata: B12xMLASparseMetadata,
    ) -> None:
        accepted = metadata.selector_num_accepted_tokens
        if not self.requires_glm_next_selector_metadata or accepted is None:
            raise RuntimeError(
                "GLM5Next draft decode metadata requires accepted-token counts"
            )
        accepted.fill_(1)


@triton.jit
def _glm_device_token_metadata_kernel(
    query_start_loc,
    seq_lens,
    request_ids,
    causal_lens,
    NUM_REQS: tl.constexpr,
    NUM_TOKENS: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
    DCP_SIZE: tl.constexpr,
    DCP_RANK: tl.constexpr,
    INTERLEAVE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    lo = tl.full((BLOCK,), 0, tl.int32)
    hi = tl.full((BLOCK,), NUM_REQS, tl.int32)
    for _ in range(SEARCH_STEPS):
        mid = (lo + hi) // 2
        end = tl.load(query_start_loc + tl.minimum(mid + 1, NUM_REQS))
        advance = (lo < hi) & (end <= token)
        hi = tl.where((lo < hi) & ~advance, mid, hi)
        lo = tl.where(advance, mid + 1, lo)
    valid = (token < NUM_TOKENS) & (lo < NUM_REQS)
    start = tl.load(query_start_loc + lo, mask=valid, other=0)
    end = tl.load(query_start_loc + lo + 1, mask=valid, other=0)
    length = tl.load(seq_lens + lo, mask=valid, other=0)
    length = tl.maximum(length - (end - start) + token - start + 1, 0)
    if DCP_SIZE > 1:
        base = length // (INTERLEAVE * DCP_SIZE) * INTERLEAVE
        length = base + tl.minimum(
            tl.maximum(length - base * DCP_SIZE - DCP_RANK * INTERLEAVE, 0),
            INTERLEAVE,
        )
    tl.store(request_ids + token, tl.where(valid, lo, 0), token < NUM_TOKENS)
    tl.store(causal_lens + token, tl.where(valid, length, 0), token < NUM_TOKENS)


class B12xGLM5NextMLASparseMetadataBuilder(B12xMLASparseMetadataBuilder):
    # The pooled selector must commit every fresh or extended prompt row
    # through run_prefill; decode commits only accepted prior rows.
    treat_short_extends_as_decodes: ClassVar[bool] = False
    supports_varlen_decode_cudagraph: ClassVar[bool] = True

    def _build_req_id_per_token(
        self,
        common_attn_metadata: "CommonAttentionMetadata",
    ) -> torch.Tensor:
        common = common_attn_metadata
        num_tokens = common.num_actual_tokens
        # Adaptive verification redistributes only the decode prefix. Its total
        # length and the CPU prefill boundaries remain exact.
        if num_tokens:
            _glm_device_token_metadata_kernel[(triton.cdiv(num_tokens, 128),)](
                common.query_start_loc,
                common.seq_lens,
                self.req_id_per_token_buffer,
                self.cache_seq_lens_per_token_buffer,
                NUM_REQS=common.num_reqs,
                NUM_TOKENS=num_tokens,
                SEARCH_STEPS=common.num_reqs.bit_length(),
                DCP_SIZE=self.dcp_world_size,
                DCP_RANK=self.dcp_rank,
                INTERLEAVE=self.cp_kv_cache_interleave_size,
                BLOCK=128,
            )
        return self.req_id_per_token_buffer[:num_tokens]


class B12xMLASparseImpl(SparseMLACommonImpl[B12xMLASparseMetadata]):
    can_return_lse_for_decode = True
    lse_base_on_e = True
    supports_dense_mha_prefill = False
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
        topk_indices_buffer: torch.Tensor | None = None,
        indexer: "Indexer | None" = None,
        **mla_args,
    ) -> None:
        if any((alibi_slopes, sliding_window, logits_soft_cap)):
            raise NotImplementedError(
                "B12X sparse MLA does not support ALiBi, sliding window, or "
                "logit soft caps."
            )
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "B12X sparse MLA supports decoder self-attention only."
            )

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
            indexer=indexer,
            topk_indices_buffer=topk_indices_buffer,
            **mla_args,
        )
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        hf_config = vllm_config.model_config.hf_text_config
        self._is_glm_next = _is_glm_next_config(hf_config)
        self._is_glm_dsa = _is_glm_dsa_config(hf_config)
        self._physical_selection_provider = (
            indexer if isinstance(indexer, B12xPhysicalSelectionProvider) else None
        )
        self.supports_mtp_with_cp_non_trivial_interleave_size = self._is_glm_next
        if self._is_glm_next:
            if recipe_error := _glm_next_recipe_error(hf_config):
                raise ValueError(recipe_error)
            if dcp_error := _glm_next_dcp_error(vllm_config):
                raise ValueError(dcp_error)
            if head_size != 512:
                raise ValueError("B12X GLM5Next sparse MLA requires head_size=512.")
        else:
            if self.kv_lora_rank != 512 or self.qk_rope_head_dim != 64:
                raise ValueError(
                    "B12X sparse MLA requires kv_lora_rank=512 and qk_rope_head_dim=64."
                )
            if head_size != 576:
                raise ValueError("B12X sparse MLA requires head_size=576.")
        if self.topk_indices_buffer is None:
            raise ValueError("B12X sparse MLA requires a top-k index buffer.")
        uses_nvfp4_cache = kv_cache_dtype == "nvfp4_ds_mla"
        self._uses_glm_dsa_nvfp4_cache = self._is_glm_dsa and uses_nvfp4_cache
        if uses_nvfp4_cache and not (
            self._is_glm_next or self._uses_glm_dsa_nvfp4_cache
        ):
            raise ValueError(
                "B12X nvfp4_ds_mla requires GLM5Next or the "
                "GLM-5.2/5.3 DSA architecture."
            )
        if kv_cache_dtype not in ("fp8_ds_mla", "nvfp4_ds_mla"):
            raise ValueError(
                "B12X sparse MLA requires a packed fp8_ds_mla or "
                "nvfp4_ds_mla KV cache; "
                f"got kv_cache_dtype={kv_cache_dtype!r}."
            )
        self._uses_nvfp4_cache = uses_nvfp4_cache

        module = get_b12x_sparse_mla()
        if module is None:
            raise RuntimeError("B12X sparse MLA requires `pip install vllm[b12x]`.")
        if not module.is_supported():
            raise RuntimeError("B12X sparse MLA is not supported on this device.")
        required_symbols = ["Caps", "bind", "plan", "run"]
        if self._is_glm_next:
            required_symbols.append(
                "concat_and_cache_glm_next_mla_nvfp4"
                if uses_nvfp4_cache
                else "concat_and_cache_glm_next_mla_fp8"
            )
        if self._uses_glm_dsa_nvfp4_cache:
            required_symbols.append("concat_and_cache_nvfp4_mla_fp8_rope")
        for name in required_symbols:
            getattr(module, name)
        self._bind = module.bind
        self._run = module.run
        self._model_type: int | None = None
        self._concat_and_cache_nvfp4_mla_fp8_rope = None
        self._concat_and_cache_glm_next_mla = None
        if self._is_glm_next:
            self._model_type = int(module.ModelType.GLM_NEXT)
            self._concat_and_cache_glm_next_mla = (
                module.concat_and_cache_glm_next_mla_nvfp4
                if self._uses_nvfp4_cache
                else module.concat_and_cache_glm_next_mla_fp8
            )
        elif self._uses_glm_dsa_nvfp4_cache:
            self._model_type = int(module.ModelType.GLM_NSA)
            self._concat_and_cache_nvfp4_mla_fp8_rope = (
                module.concat_and_cache_nvfp4_mla_fp8_rope
            )

        scheduler_config = vllm_config.scheduler_config
        max_tokens = int(scheduler_config.max_num_batched_tokens)
        max_seqs = int(scheduler_config.max_num_seqs)
        self._input_num_heads = self.num_heads * self.dcp_world_size
        self._q_head_dim = self.kv_lora_rank + self.qk_rope_head_dim
        self._topk_tokens = int(self.topk_indices_buffer.shape[-1])
        if self._is_glm_next or self._is_glm_dsa:
            expected_width = int(hf_config.index_topk)
            if self._is_glm_next:
                expected_width += int(hf_config.index_kpool) - 1
            if self._topk_tokens != expected_width:
                raise ValueError(
                    "B12X GLM sparse MLA requires a selector output width "
                    f"of {expected_width}, got {self._topk_tokens}."
                )
        self._max_tokens = max_tokens
        self._max_seqs = max_seqs
        # Model lane 0 executes the target model; lane 1, when the runner
        # reserves it, executes the drafter. Lookahead cannot exceed a lane's
        # layer count minus one, so the single-layer drafter keeps depth zero.
        self._ckv_lane_layer_counts = _ckv_lane_layer_counts(vllm_config)
        self._max_speculative_decode_query_len = _max_speculative_decode_query_len(
            vllm_config
        )
        self._decode_max_rows = min(
            max_tokens,
            max_seqs * self._max_speculative_decode_query_len,
        )
        self._kv_dtype = torch.uint8
        kernel_page_size = (
            int(vllm_config.cache_config.block_size) if self._is_glm_next else 64
        )
        self._ckv_gather_enabled = (
            (self._is_glm_next or self._is_glm_dsa)
            and self.dcp_world_size > 1
            and envs.VLLM_B12X_MLA_CKV_GATHER
            and (
                self._is_glm_next
                or (
                    self.num_heads % 8 == 0
                    and not vllm_config.parallel_config.enable_dbo
                )
            )
        )
        if envs.VLLM_B12X_MLA_CKV_GATHER:
            logger.info_once(
                "Full-CKV prefill configuration: model=%s DCP=%d local_heads=%d "
                "enabled=%s",
                getattr(hf_config, "model_type", None),
                self.dcp_world_size,
                self.num_heads,
                self._ckv_gather_enabled,
            )
        max_ckv_tokens = envs.VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS
        cp_kv_cache_interleave_size = int(
            vllm_config.parallel_config.cp_kv_cache_interleave_size
        )
        self._ckv_interleave = cp_kv_cache_interleave_size
        self._ckv_capacity_tokens = (
            max_ckv_tokens + self.dcp_world_size - 1
        ) // self.dcp_world_size + max_seqs * cp_kv_cache_interleave_size
        self._ckv_local_capacity = 0
        self._ckv_reservation: _CKVReservation | None = None
        self._ckv_current_cache: torch.Tensor | None = None

        self._module = module
        self._kernel_page_size = 0
        self._kernel_page_size_finalized = not self._is_glm_next
        self._set_kernel_page_size(kernel_page_size)
        self.supports_quant_query_input = False
        if self.dcp_world_size > 1:
            from .prefill_diagnostics import get_prefill_trace, install_backend_trace

            trace = get_prefill_trace()
            if trace is not None:
                install_backend_trace(self, trace)

    def _set_kernel_page_size(self, kernel_page_size: int) -> None:
        if kernel_page_size <= 0 or kernel_page_size % 64:
            raise ValueError(
                "B12X sparse MLA kernel page size must be a positive multiple "
                f"of 64, got {kernel_page_size}."
            )
        if kernel_page_size == self._kernel_page_size:
            return

        def make_plan(mode: str, num_q_heads: int = self._input_num_heads):
            max_rows = self._decode_max_rows if mode == "decode" else self._max_tokens
            caps_kwargs = dict(
                device=torch.device("cuda", torch.accelerator.current_device_index()),
                num_q_heads=num_q_heads,
                max_q_rows=max_rows,
                max_width=self._topk_tokens,
                softmax_scale=self.scale,
                dtype=torch.bfloat16,
                kv_dtype=self._kv_dtype,
                head_dim=self._q_head_dim,
                v_head_dim=self.kv_lora_rank,
                mode=mode,
                max_batch=max_rows,
                max_chunks_per_row=max(1, (self._topk_tokens + 63) // 64),
                page_size=kernel_page_size,
                return_lse=self.need_to_return_lse_for_decode,
                lse_scale="natural",
            )
            if self._model_type is not None:
                caps_kwargs["model_type"] = self._model_type
            if self._uses_nvfp4_cache:
                caps_kwargs.update(_nvfp4_run_options(is_glm_next=self._is_glm_next))
            return self._module.plan(self._module.Caps(**caps_kwargs))

        decode_plan = make_plan("decode")
        extend_plan = make_plan("extend")
        self._cache_record_bytes = int(decode_plan.caps.cache_record_bytes)
        self._decode_plan = decode_plan
        self._extend_plan = extend_plan
        self._ckv_extend_plan = (
            make_plan("extend", self.num_heads) if self._ckv_gather_enabled else None
        )
        plans = [decode_plan, extend_plan]
        if self._ckv_extend_plan is not None:
            plans.append(self._ckv_extend_plan)
        self._scratch_nbytes = max(int(plan.layout.nbytes) for plan in plans)
        self._ckv_local_capacity = _round_up_ckv_rank_tokens(
            self._ckv_capacity_tokens,
            page_size=kernel_page_size,
            dcp_world_size=self.dcp_world_size,
        )
        self._ckv_current_capacity = _round_up_ckv_rank_tokens(
            (self._max_tokens + self.dcp_world_size - 1) // self.dcp_world_size
            + self._max_seqs * self._ckv_interleave,
            page_size=kernel_page_size,
            dcp_world_size=self.dcp_world_size,
        )
        self._kernel_page_size = kernel_page_size
        self._reserve_planned_workspaces()

    def _base_workspace_specs(
        self, plan
    ) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
        del plan
        q_spec = (
            (self._max_tokens, self._input_num_heads, self._q_head_dim),
            torch.bfloat16,
        )
        scratch_spec = ((self._scratch_nbytes,), torch.uint8)
        return (q_spec, scratch_spec)

    @staticmethod
    def _workspace_nbytes(
        specs: tuple[tuple[tuple[int, ...], torch.dtype], ...],
    ) -> int:
        return sum(
            ((prod(shape) * dtype.itemsize + 255) // 256) * 256
            for shape, dtype in specs
        )

    def _reserve_planned_workspaces(self) -> None:
        if not is_workspace_manager_initialized():
            return
        plan_specs = (
            self._base_workspace_specs(self._decode_plan),
            self._base_workspace_specs(self._extend_plan),
        )
        largest_specs = max(plan_specs, key=self._workspace_nbytes)
        current_workspace_manager().get_simultaneous(*largest_specs)
        if self._ckv_gather_enabled and self._kernel_page_size_finalized:
            self._reserve_ckv_prefetch()

    def get_dcp_prefill_workspace_specs(
        self,
    ) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
        """Return the sequential prefill scratch prefix, excluding persistent KV."""
        return self._base_workspace_specs(self._extend_plan)

    def _reserve_ckv_prefetch(self) -> None:
        manager = current_workspace_manager()
        num_ubatches, num_lanes = manager.execution_lane_shape()
        budget_bytes = envs.VLLM_B12X_MLA_CKV_PREFETCH_WORKSPACE_MIB * 1024**2
        current_bytes = (
            (1 + self.dcp_world_size)
            * self._ckv_current_capacity
            * self._cache_record_bytes
        )
        requested_depth = (
            envs.VLLM_B12X_MLA_CKV_PREFETCH_DEPTH if self._is_glm_dsa else 0
        )
        if requested_depth < 0:
            raise ValueError("CKV prefetch depth must be non-negative")
        lane_layer_counts: tuple[int, ...] | None = None
        recorded_counts = getattr(self, "_ckv_lane_layer_counts", None)
        if recorded_counts is not None:
            # Lanes beyond the recorded target/drafter pair keep the target depth.
            lane_layer_counts = tuple(
                recorded_counts[lane]
                if lane < len(recorded_counts)
                else recorded_counts[0]
                for lane in range(num_lanes)
            )
        plan = CKVPrefetchPlan.create(
            requested_depth=0,
            budget_bytes=budget_bytes,
            dcp_world_size=self.dcp_world_size,
            local_capacity=self._ckv_local_capacity,
            record_bytes=self._cache_record_bytes,
            num_ubatches=num_ubatches,
            num_lanes=num_lanes,
            lane_layer_counts=lane_layer_counts,
        )
        if requested_depth and (
            budget_bytes == 0 or budget_bytes >= current_bytes + plan.lane_nbytes
        ):
            plan = CKVPrefetchPlan.create(
                requested_depth=requested_depth,
                budget_bytes=budget_bytes - current_bytes if budget_bytes else 0,
                dcp_world_size=self.dcp_world_size,
                local_capacity=self._ckv_local_capacity,
                record_bytes=self._cache_record_bytes,
                num_ubatches=num_ubatches,
                num_lanes=num_lanes,
                lane_layer_counts=lane_layer_counts,
            )
        current_capacity = self._ckv_current_capacity if plan.effective_depth else 0
        key = (plan, current_capacity)
        reservations = _CKV_RESERVATIONS.setdefault(manager, {})
        reservation = reservations.get(key)
        if reservation is None:
            if plan.effective_depth:
                from vllm.distributed.parallel_state import (
                    ensure_dcp_ckv_prefetch_group,
                )

                ensure_dcp_ckv_prefetch_group()
            device = self._decode_plan.caps.device
            pool = CKVWorkspacePool(plan, device)
            current_local = torch.empty(
                (
                    plan.num_ubatches,
                    plan.num_lanes,
                    current_capacity,
                    self._cache_record_bytes,
                ),
                dtype=torch.uint8,
                device=device,
            )
            current_gathered = torch.empty(
                (
                    plan.num_ubatches,
                    plan.num_lanes,
                    self.dcp_world_size * current_capacity,
                    self._cache_record_bytes,
                ),
                dtype=torch.uint8,
                device=device,
            )
            reservation = _CKVReservation(
                CKVPrefetchRegistry(pool), current_local, current_gathered
            )
            reservations[key] = reservation
            logger.info(
                "Reserved CKV prefill storage: depth=%d lane_depths=%s lanes=%d "
                "bytes=%d including native current-chunk exchange",
                plan.effective_depth,
                list(plan.lane_depths),
                plan.num_ubatches * plan.num_lanes,
                plan.total_nbytes + current_local.numel() + current_gathered.numel(),
            )
        self._ckv_reservation = reservation

    @staticmethod
    def reset_kv_cache_binding_state() -> None:
        """Finish ring users and discard cache identities before cache rebinding."""
        for reservations in _CKV_RESERVATIONS.values():
            for reservation in reservations.values():
                reservation.registry.clear()

    @staticmethod
    def begin_ckv_prefetch_step() -> None:
        """Join interrupted gathers before this lane's native-cache updates."""
        from vllm.v1.worker.ubatching import dbo_current_ubatch_id
        from vllm.v1.worker.workspace import current_workspace_lane

        lane = (dbo_current_ubatch_id(), current_workspace_lane())
        for reservations in _CKV_RESERVATIONS.values():
            for reservation in reservations.values():
                state = reservation.registry.states.get(lane)
                if state is not None and state.pending:
                    state.begin_step(torch.cuda.current_stream())

    def set_ckv_current_cache(self, original_cache: torch.Tensor) -> None:
        """Supply stable cache identity after this layer's native producer runs."""
        self._ckv_current_cache = original_cache

    def _append_ckv_current_chunk(
        self,
        gathered: torch.Tensor,
        metadata: B12xMLASparseMetadata,
        original_cache: torch.Tensor,
        lane: tuple[int, int],
    ) -> None:
        from b12x.attention._shared.mla.kv_cache import (
            gather_ckv_current_chunk,
            insert_ckv_current_chunk,
        )

        reservation = self._ckv_reservation
        assert reservation is not None
        assert metadata.global_cache_seq_lens_per_req is not None
        assert metadata.dcp_rank_req_starts is not None
        local = reservation.current_local[lane[0], lane[1]]
        current = reservation.current_gathered[lane[0], lane[1]]
        trace = getattr(self, "_prefill_trace", None)
        exchange = _dcp_all_gather_current_stream
        group = get_dcp_group()
        if trace is not None and trace.ownership_active:
            gather_ckv_current_chunk = trace.wrap(
                "current_pack", gather_ckv_current_chunk
            )
            exchange = trace.wrap(
                "current_exchange", exchange, **trace.group_fields(group)
            )
            insert_ckv_current_chunk = trace.wrap(
                "current_insert", insert_ckv_current_chunk
            )
        gather_ckv_current_chunk(
            original_cache.view(torch.uint8),
            local,
            metadata.block_table,
            metadata.global_cache_seq_lens_per_req,
            metadata.query_start_loc,
            dcp_rank=self.dcp_rank,
            dcp_world_size=self.dcp_world_size,
            interleave=metadata.cp_kv_cache_interleave_size,
            num_reqs=metadata.num_reqs,
            current_capacity=self._ckv_current_capacity,
        )
        exchange(group, local, current)
        insert_ckv_current_chunk(
            current,
            gathered,
            metadata.dcp_rank_req_starts,
            metadata.global_cache_seq_lens_per_req,
            metadata.query_start_loc,
            dcp_world_size=self.dcp_world_size,
            interleave=metadata.cp_kv_cache_interleave_size,
            num_reqs=metadata.num_reqs,
            current_capacity=self._ckv_current_capacity,
            padded_tokens=metadata.dcp_padded_total_tokens,
        )

    def _queue_ckv_gather(
        self,
        state: CKVPrefetchState,
        layer_idx: int,
        cache: torch.Tensor,
        metadata: B12xMLASparseMetadata,
        stream: torch.cuda.Stream,
        producer: torch.cuda.Stream,
        *,
        asynchronous: bool,
    ) -> None:
        from vllm.distributed.parallel_state import get_dcp_ckv_prefetch_group

        local, gathered = state.prepare_gather(
            layer_idx, stream, producer_stream=producer
        )
        try:
            with torch.cuda.stream(stream):
                self._gather_full_ckv(
                    cache,
                    metadata,
                    local,
                    gathered,
                    group=get_dcp_ckv_prefetch_group() if asynchronous else None,
                    history_only=asynchronous,
                )
        finally:
            completion = torch.cuda.Event()
            trace = getattr(self, "_prefill_trace", None)
            if trace is None:
                completion.record(stream)
            else:
                trace.record(completion, stream)
            state.finish_gather(layer_idx, completion)

    def _consume_ckv(
        self,
        cache: torch.Tensor,
        metadata: B12xMLASparseMetadata,
        layer: AttentionLayer,
        original_cache: torch.Tensor | None,
    ) -> tuple[torch.Tensor, CKVPrefetchState, int]:
        from vllm.model_executor.models.utils import extract_layer_index
        from vllm.v1.worker.ubatching import dbo_current_ubatch_id
        from vllm.v1.worker.workspace import current_workspace_lane

        reservation = self._ckv_reservation
        if reservation is None:
            raise RuntimeError(
                "CKV prefill storage was not reserved before KV admission"
            )
        if not reservation.collectives_warmed:
            raise RuntimeError("CKV collective warmup must finish before KV admission")
        lane = (dbo_current_ubatch_id(), current_workspace_lane())
        state = reservation.registry.for_workspace(
            reservation.registry.pool.storage, lane=lane
        )
        main_stream = torch.cuda.current_stream()
        try:
            layer_idx = extract_layer_index(getattr(layer, "layer_name", ""))
        except (AttributeError, ValueError, AssertionError, IndexError):
            layer_idx = 0
            original_cache = None
        cache_identity_known = original_cache is not None
        can_prefetch = cache_identity_known and getattr(
            metadata, "ckv_has_history", True
        )
        original_cache = cache if original_cache is None else original_cache
        state.register_cache(layer_idx, original_cache)
        state.enter_layer(layer_idx, main_stream)
        prefetched = layer_idx in state.pending
        if state.pending and not can_prefetch:
            state.begin_step(main_stream)
            prefetched = False
        trace = getattr(self, "_prefill_trace", None)
        if trace is not None and trace.ownership_active:
            with trace.scope(
                "ckv_selection",
                prefetched=int(prefetched),
                cache_identity_known=int(cache_identity_known),
                has_history=int(getattr(metadata, "ckv_has_history", True)),
            ):
                pass
        if not prefetched:
            self._queue_ckv_gather(
                state,
                layer_idx,
                original_cache,
                metadata,
                main_stream,
                main_stream,
                asynchronous=False,
            )
        gathered = state.consume(layer_idx, main_stream).view(
            -1, self._kernel_page_size, self._cache_record_bytes
        )
        try:
            if prefetched:
                self._append_ckv_current_chunk(gathered, metadata, original_cache, lane)
            if can_prefetch and reservation.registry.pool.plan.effective_depth:
                targets = state.targets(layer_idx)
                if targets:
                    stream = state.get_gather_stream(
                        lambda: reservation.gather_streams[lane]
                    )
                    for target in targets:
                        self._queue_ckv_gather(
                            state,
                            target,
                            state.layer_caches[target],
                            metadata,
                            stream,
                            main_stream,
                            asynchronous=True,
                        )
        except BaseException:
            completion = torch.cuda.Event()
            trace = getattr(self, "_prefill_trace", None)
            if trace is None:
                completion.record(main_stream)
            else:
                trace.record(completion, main_stream)
            state.finish_consumer(layer_idx, completion)
            raise
        return gathered, state, layer_idx

    def _use_decode_plan(
        self,
        attn_metadata: B12xMLASparseMetadata,
        num_tokens: int,
    ) -> bool:
        if attn_metadata.max_query_len <= 1:
            return True
        return (
            attn_metadata.is_spec_decode
            and attn_metadata.max_query_len <= self._max_speculative_decode_query_len
            and num_tokens
            <= attn_metadata.num_reqs * self._max_speculative_decode_query_len
            and num_tokens <= self._decode_max_rows
        )

    def _workspace_specs(
        self,
        plan: Any,
        *,
        input_num_heads: int,
        include_ckv: bool,
    ) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
        del plan
        q_spec = (
            (self._max_tokens, input_num_heads, self._q_head_dim),
            torch.bfloat16,
        )
        scratch_spec = ((self._scratch_nbytes,), torch.uint8)
        ckv_specs = (
            (
                (self._ckv_local_capacity, self._cache_record_bytes),
                torch.uint8,
            ),
            (
                (
                    self.dcp_world_size * self._ckv_local_capacity,
                    self._cache_record_bytes,
                ),
                torch.uint8,
            ),
        )
        return (
            q_spec,
            scratch_spec,
            *(ckv_specs if include_ckv else ()),
        )

    def _reserve_attention_workspaces(self) -> None:
        if not self._ckv_gather_enabled:
            return
        assert self._ckv_extend_plan is not None
        manager = current_workspace_manager()
        for plan, input_num_heads, include_ckv in (
            (self._decode_plan, self._input_num_heads, False),
            (self._extend_plan, self._input_num_heads, False),
            (self._ckv_extend_plan, self.num_heads, False),
        ):
            manager.reserve_all(
                *self._workspace_specs(
                    plan,
                    input_num_heads=input_num_heads,
                    include_ckv=include_ckv,
                )
            )

    def _borrow_workspaces(
        self,
        *,
        input_num_heads: int | None = None,
        include_ckv: bool = False,
    ) -> list[torch.Tensor]:
        input_num_heads = (
            self._input_num_heads if input_num_heads is None else input_num_heads
        )
        specs = self._workspace_specs(
            self._decode_plan,
            input_num_heads=input_num_heads,
            include_ckv=include_ckv,
        )
        return current_workspace_manager().get_simultaneous(*specs)

    def supports_fused_mla_query_output(
        self,
        num_heads: int,
        output_dtype: torch.dtype,
    ) -> bool:
        return bool(
            self.dcp_world_size == 1
            and output_dtype == torch.bfloat16
            and num_heads == self._input_num_heads
            and self._q_head_dim == 576
        )

    def get_fused_mla_query_output(
        self,
        num_tokens: int,
        num_heads: int,
        output_dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if (
            not self.supports_fused_mla_query_output(num_heads, output_dtype)
            or num_tokens <= 0
            or num_tokens > self._max_tokens
        ):
            return None
        q_buffer = self._borrow_workspaces()[0]
        output = q_buffer[:num_tokens, :num_heads]
        if not output.is_contiguous():
            raise RuntimeError("B12X fused MLA query output must be contiguous.")
        return output

    def b12x_warmup_key(self) -> tuple[object, ...]:
        return (
            type(self),
            self._decode_plan.caps.device,
            self._input_num_heads,
            self._q_head_dim,
            self._topk_tokens,
            self._max_tokens,
            self._decode_plan.caps.max_q_rows,
            self.need_to_return_lse_for_decode,
            self._model_type,
            self._kernel_page_size,
            self._ckv_gather_enabled,
        )

    def prepare_profile_collectives(self) -> None:
        """Prepare native-cache collectives before the attention memory profile."""
        self._warmup_ckv_collectives()

    def _warmup_ckv_collectives(self) -> None:
        """Initialize actual collective paths and lane streams before KV admission."""
        reservation = self._ckv_reservation
        if reservation is None or reservation.collectives_warmed:
            return
        from vllm.distributed.parallel_state import get_dcp_ckv_prefetch_group

        plan = reservation.registry.pool.plan
        sizes = _ckv_collective_warmup_sizes(
            plan.local_capacity,
            page_size=self._kernel_page_size,
            dcp_world_size=plan.dcp_world_size,
        )
        default_group = get_dcp_group()
        prefetch_group = get_dcp_ckv_prefetch_group() if plan.effective_depth else None
        main = torch.cuda.current_stream()
        completions = []
        unit = plan.local_capacity * plan.record_bytes
        trace = getattr(self, "_prefill_trace", None)
        exchange = _dcp_all_gather_current_stream
        if trace is not None and trace.ownership_active:

            def observed_exchange(group, source, output):
                with trace.scope(
                    "startup_collective",
                    ubatch=ubatch,
                    lane=model_lane,
                    slot=slot,
                    stream_role=stream_role,
                    **trace.group_fields(group),
                ):
                    return _dcp_all_gather_current_stream(group, source, output)

            exchange = observed_exchange
        for ubatch in range(plan.num_ubatches):
            for model_lane in range(plan.num_lanes):
                lane = (ubatch, model_lane)
                storage = reservation.registry.pool.storage.narrow(
                    0,
                    plan.lane_offset(ubatch, model_lane),
                    plan.lane_nbytes_for(model_lane),
                )
                local = storage[:unit].view(plan.local_capacity, plan.record_bytes)
                local.zero_()
                targets = [
                    storage.narrow(
                        0,
                        unit + slot * unit * plan.dcp_world_size,
                        unit * plan.dcp_world_size,
                    ).view(plan.dcp_world_size * plan.local_capacity, plan.record_bytes)
                    for slot in range(plan.lane_ring_slots(model_lane))
                ]
                stream_role = 0
                for slot, gathered in enumerate(targets):
                    for count in sizes:
                        exchange(
                            default_group,
                            local[:count].view(-1),
                            gathered[: plan.dcp_world_size * count].view(-1),
                        )
                # A lane without lookahead never gathers on the prefetch stream.
                if prefetch_group is not None and plan.lane_depth(model_lane):
                    slot = -1
                    current_local = reservation.current_local[ubatch, model_lane]
                    current_local.zero_()
                    exchange(
                        default_group,
                        current_local,
                        reservation.current_gathered[ubatch, model_lane],
                    )
                    stream = reservation.gather_streams.get(lane)
                    if stream is None:
                        stream = torch.cuda.Stream(device=local.device)
                        reservation.gather_streams[lane] = stream
                    if trace is not None and trace.ownership_active:
                        with trace.scope(
                            "startup_stream_wait",
                            ubatch=ubatch,
                            lane=model_lane,
                            event=-1,
                        ):
                            stream.wait_stream(main)
                    else:
                        stream.wait_stream(main)
                    stream_role = 1
                    with torch.cuda.stream(stream):
                        for slot, gathered in enumerate(targets):
                            for count in sizes:
                                exchange(
                                    prefetch_group,
                                    local[:count].view(-1),
                                    gathered[: plan.dcp_world_size * count].view(-1),
                                )
                        completion = torch.cuda.Event()
                        if trace is not None and trace.ownership_active:
                            trace.record(
                                completion,
                                stream,
                                ubatch=ubatch,
                                lane=model_lane,
                                slot=-1,
                            )
                        else:
                            completion.record(stream)
                    if trace is not None and trace.ownership_active:
                        with trace.scope(
                            "startup_join", ubatch=ubatch, lane=model_lane
                        ):
                            trace.wait(main, completion, reason=5)
                    else:
                        main.wait_event(completion)
                    completions.append(completion)
        main.synchronize()
        reservation.collectives_warmed = True
        logger.info(
            "Warmed CKV all-gather paths: rank_spans=%s lane_ring_slots=%s lanes=%d "
            "prefetch_streams=%d current_rank_capacity=%d",
            sizes,
            [plan.lane_ring_slots(lane) for lane in range(plan.num_lanes)],
            plan.num_ubatches * plan.num_lanes,
            len(reservation.gather_streams),
            reservation.current_local.shape[-2],
        )

    def warmup(self, token_counts: tuple[int, ...]) -> None:
        decode_capacity = int(self._decode_plan.caps.max_q_rows)
        decode_rows = {
            int(rows) for rows in token_counts if 0 < int(rows) <= decode_capacity
        }
        decode_rows.add(1)
        extend_rows = {1, 2, 4, self._max_tokens}
        kv_cache = torch.zeros(
            (1, self._kernel_page_size, self._cache_record_bytes),
            dtype=torch.uint8,
            device=self._decode_plan.caps.device,
        )

        plans: list[tuple[Any, list[int], int]] = [
            (self._decode_plan, sorted(decode_rows), self._input_num_heads),
            (self._extend_plan, sorted(extend_rows), self._input_num_heads),
        ]
        if self._ckv_extend_plan is not None:
            plans.append((self._ckv_extend_plan, sorted(extend_rows), self.num_heads))

        for plan, rows_to_warm, input_num_heads in plans:
            q_buffer, scratch = self._borrow_workspaces(
                input_num_heads=input_num_heads
            )[:2]
            for rows in rows_to_warm:
                if rows > int(plan.caps.max_q_rows):
                    continue
                q = q_buffer[:rows]
                q.zero_()
                selected_indices = torch.zeros(
                    (rows, self._topk_tokens),
                    dtype=torch.int32,
                    device=q.device,
                )
                cache_lengths = torch.full(
                    (rows if plan is self._decode_plan else 1,),
                    self._kernel_page_size,
                    dtype=torch.int32,
                    device=q.device,
                )
                selected_lengths = torch.ones(
                    (rows,), dtype=torch.int32, device=q.device
                )
                binding = self._bind(
                    plan,
                    scratch=scratch,
                    q=q,
                    kv_cache=kv_cache,
                    selected_indices=selected_indices,
                    cache_lengths=cache_lengths,
                    selected_lengths=selected_lengths,
                )
                self._run(binding)

        reservation = self._ckv_reservation
        if reservation is not None and reservation.registry.pool.plan.effective_depth:
            from b12x.attention._shared.mla.kv_cache import (
                gather_ckv_current_chunk,
                gather_ckv_history,
                insert_ckv_current_chunk,
            )

            # Empty request spans compile native-copy paths without reading KV.
            device = kv_cache.device
            starts = torch.zeros(2, dtype=torch.int32, device=device)
            lengths = torch.zeros(1, dtype=torch.int32, device=device)
            rank_spans = torch.zeros(
                (self.dcp_world_size, 1), dtype=torch.int32, device=device
            )
            blocks = torch.full((1, 1), -1, dtype=torch.int32, device=device)
            unit = self._ckv_local_capacity * self._cache_record_bytes
            storage = reservation.registry.pool.storage
            local = storage[:unit].view(
                self._ckv_local_capacity, self._cache_record_bytes
            )
            gathered = storage[unit : unit * (1 + self.dcp_world_size)].view(
                self.dcp_world_size * self._ckv_local_capacity, self._cache_record_bytes
            )
            gather_ckv_history(
                kv_cache,
                local,
                blocks,
                rank_spans,
                rank_spans,
                lengths,
                starts,
                dcp_rank=self.dcp_rank,
                dcp_world_size=self.dcp_world_size,
                interleave=self._ckv_interleave,
                num_reqs=1,
                padded_tokens=1,
            )
            current_local = reservation.current_local[0, 0]
            current_gathered = reservation.current_gathered[0, 0]
            gather_ckv_current_chunk(
                kv_cache,
                current_local,
                blocks,
                lengths,
                starts,
                dcp_rank=self.dcp_rank,
                dcp_world_size=self.dcp_world_size,
                interleave=self._ckv_interleave,
                num_reqs=1,
                current_capacity=self._ckv_current_capacity,
            )
            insert_ckv_current_chunk(
                current_gathered,
                gathered,
                rank_spans,
                lengths,
                starts,
                dcp_world_size=self.dcp_world_size,
                interleave=self._ckv_interleave,
                num_reqs=1,
                current_capacity=self._ckv_current_capacity,
                padded_tokens=1,
            )
        self._warmup_ckv_collectives()

    def finalize_kv_cache_geometry(self, kernel_page_size: int) -> None:
        """Finalize kernel plans and workspace memory before KV profiling.

        GLM5Next hybrid cache alignment resolves the physical page size after
        model construction. Full-CKV gather workspace depends on that value,
        so every execution slot must be sized while the memory profiler can
        still subtract the allocation from the KV-cache budget.

        Args:
            kernel_page_size: Resolved physical KV-cache page size in tokens.

        Raises:
            RuntimeError: If an established page size is changed.
        """
        if not self._is_glm_next:
            return
        if self._kernel_page_size_finalized:
            if kernel_page_size != self._kernel_page_size:
                raise RuntimeError(
                    "B12X GLM5Next KV-cache page size is immutable after "
                    f"finalization: {self._kernel_page_size} != {kernel_page_size}."
                )
            return
        self._set_kernel_page_size(kernel_page_size)
        self._reserve_attention_workspaces()
        self._kernel_page_size_finalized = True
        if self._ckv_gather_enabled:
            self._reserve_ckv_prefetch()

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        if getattr(self, "_uses_glm_dsa_nvfp4_cache", False) and (
            kv_cache.ndim != 3
            or int(kv_cache.shape[-1]) != _GLM_DSA_NVFP4_CACHE_RECORD_BYTES
            or kv_cache.dtype != torch.uint8
        ):
            raise ValueError(
                "B12X GLM DSA NVFP4 cache must have shape "
                f"[pages, page_size, {_GLM_DSA_NVFP4_CACHE_RECORD_BYTES}] "
                "uint8, got "
                f"shape={tuple(kv_cache.shape)}, dtype={kv_cache.dtype}."
            )
        if self._is_glm_next:
            if (
                kv_cache.ndim != 3
                or int(kv_cache.shape[-1]) != self._cache_record_bytes
            ):
                raise ValueError(
                    "B12X GLM5Next cache must have shape "
                    f"[pages, page_size, {self._cache_record_bytes}], got "
                    f"shape={tuple(kv_cache.shape)}, stride={kv_cache.stride()}, "
                    f"dtype={kv_cache.dtype}"
                )
            cache_page_size = int(kv_cache.shape[1])
            if getattr(self, "_kernel_page_size_finalized", False):
                if cache_page_size != self._kernel_page_size:
                    raise RuntimeError(
                        "B12X GLM5Next bound cache does not match the finalized "
                        f"page size: {cache_page_size} != {self._kernel_page_size}."
                    )
                return
            if self._ckv_gather_enabled:
                raise RuntimeError(
                    "B12X GLM5Next full-CKV gather requires page geometry "
                    "finalization before KV-cache memory profiling."
                )
            self._set_kernel_page_size(cache_page_size)

    def do_kv_cache_update(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None:
        if getattr(self, "_uses_glm_dsa_nvfp4_cache", False):
            if kv_cache.numel() == 0:
                return
            assert self._concat_and_cache_nvfp4_mla_fp8_rope is not None
            self._concat_and_cache_nvfp4_mla_fp8_rope(
                kv_c_normed,
                k_pe.squeeze(1),
                kv_cache,
                slot_mapping.flatten(),
                k_scale,
            )
            return
        if not self._is_glm_next:
            return super().do_kv_cache_update(
                kv_c_normed,
                k_pe,
                kv_cache,
                slot_mapping,
                kv_cache_dtype,
                k_scale,
            )
        del k_scale
        if kv_cache.numel() == 0:
            return
        if int(k_pe.shape[-1]) != 0:
            raise ValueError(
                "B12X GLM5Next cache updates require a zero-width RoPE tensor, "
                f"got shape={tuple(k_pe.shape)}."
            )
        assert self._concat_and_cache_glm_next_mla is not None
        self._concat_and_cache_glm_next_mla(
            kv_c_normed,
            kv_cache,
            slot_mapping.flatten(),
        )

    def uses_full_ckv_dcp(
        self,
        attn_metadata: B12xMLASparseMetadata,
        num_tokens: int,
    ) -> bool:
        if torch.cuda.is_current_stream_capturing():
            return False
        return (
            self._ckv_gather_enabled
            and getattr(self, "_kernel_page_size_finalized", False)
            and attn_metadata.dcp_ckv_gather_eligible
            and attn_metadata.num_decode_tokens == 0
            and not attn_metadata.is_spec_decode
            and num_tokens == attn_metadata.num_actual_tokens
            and 0 < attn_metadata.dcp_padded_total_tokens <= self._ckv_local_capacity
            and attn_metadata.dcp_local_total_tokens
            <= attn_metadata.dcp_padded_total_tokens
            and all(
                value is not None
                for value in (
                    attn_metadata.ckv_selected_indices,
                    attn_metadata.ckv_active_counts,
                    attn_metadata.dcp_rank_req_starts,
                    attn_metadata.dcp_rank_req_lens,
                    attn_metadata.dcp_local_cu_seq_lens,
                    attn_metadata.global_cache_seq_lens_per_req,
                )
            )
        )

    def _gather_full_ckv(
        self,
        kv_cache: torch.Tensor,
        attn_metadata: B12xMLASparseMetadata,
        local_buffer: torch.Tensor,
        gathered_buffer: torch.Tensor,
        *,
        group: Any | None = None,
        history_only: bool = False,
    ) -> torch.Tensor:
        if not self.uses_full_ckv_dcp(attn_metadata, attn_metadata.num_actual_tokens):
            raise RuntimeError("full CKV gather called for an ineligible batch")
        # The model layer exposes packed FP8 records through an E4M3 view.
        # Gathering copies the native bytes, including scales and the RoPE tail.
        if kv_cache.dtype == torch.float8_e4m3fn:
            kv_cache = kv_cache.view(torch.uint8)
        if not _is_native_ckv_source_layout(
            kv_cache,
            page_size=self._kernel_page_size,
            record_bytes=self._cache_record_bytes,
        ):
            raise ValueError(
                "CKV gather requires native "
                f"{self._cache_record_bytes}-byte records; "
                f"got shape={tuple(kv_cache.shape)}, stride={kv_cache.stride()}"
            )
        expected_local_shape = (
            self._ckv_local_capacity,
            self._cache_record_bytes,
        )
        expected_gathered_shape = (
            self.dcp_world_size * self._ckv_local_capacity,
            self._cache_record_bytes,
        )
        if tuple(local_buffer.shape) != expected_local_shape:
            raise RuntimeError("CKV local workspace has an invalid shape")
        if tuple(gathered_buffer.shape) != expected_gathered_shape:
            raise RuntimeError("CKV gathered workspace has an invalid shape")

        assert attn_metadata.dcp_local_cu_seq_lens is not None
        local_tokens = attn_metadata.dcp_local_total_tokens
        padded_tokens = attn_metadata.dcp_padded_total_tokens
        if history_only:
            from b12x.attention._shared.mla.kv_cache import gather_ckv_history

            assert attn_metadata.dcp_rank_req_starts is not None
            assert attn_metadata.dcp_rank_req_lens is not None
            assert attn_metadata.global_cache_seq_lens_per_req is not None
            gather_ckv_history(
                kv_cache,
                local_buffer,
                attn_metadata.block_table,
                attn_metadata.dcp_rank_req_starts,
                attn_metadata.dcp_rank_req_lens,
                attn_metadata.global_cache_seq_lens_per_req,
                attn_metadata.query_start_loc,
                dcp_rank=self.dcp_rank,
                dcp_world_size=self.dcp_world_size,
                interleave=attn_metadata.cp_kv_cache_interleave_size,
                num_reqs=attn_metadata.num_reqs,
                padded_tokens=padded_tokens,
            )
        elif local_tokens:
            ops.cp_gather_cache(
                src_cache=kv_cache,
                dst=local_buffer[:local_tokens],
                block_table=attn_metadata.block_table,
                cu_seq_lens=attn_metadata.dcp_local_cu_seq_lens,
                batch_size=attn_metadata.num_reqs,
            )
        if not history_only and local_tokens < padded_tokens:
            local_buffer[local_tokens:padded_tokens].zero_()
        _dcp_all_gather_current_stream(
            get_dcp_group() if group is None else group,
            local_buffer[:padded_tokens].view(-1),
            gathered_buffer[: self.dcp_world_size * padded_tokens].view(-1),
        )
        return gathered_buffer.view(
            -1, self._kernel_page_size, self._cache_record_bytes
        )

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: B12xMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        current_cache = getattr(self, "_ckv_current_cache", None)
        self._ckv_current_cache = None
        cache_page_size = int(kv_c_and_k_pe_cache.shape[1])
        metadata_page_size = int(attn_metadata.block_size)
        if self._is_glm_next and (
            cache_page_size != self._kernel_page_size
            or metadata_page_size != self._kernel_page_size
        ):
            raise RuntimeError(
                "B12X GLM5Next page geometry does not match the bound plan: "
                f"cache={cache_page_size}, metadata={metadata_page_size}, "
                f"plan={self._kernel_page_size}"
            )
        num_tokens = int(q[0].shape[0] if isinstance(q, tuple) else q.shape[0])
        use_ckv_gather = self.uses_full_ckv_dcp(attn_metadata, num_tokens)
        if use_ckv_gather:
            assert self._ckv_extend_plan is not None
            plan = self._ckv_extend_plan
            logger.info_once("Using full-CKV gather for B12X DCP prefill")
        else:
            plan = (
                self._decode_plan
                if self._use_decode_plan(attn_metadata, num_tokens)
                else self._extend_plan
            )
        input_num_heads = self.num_heads if use_ckv_gather else self._input_num_heads
        workspace_specs = self._workspace_specs(
            plan,
            input_num_heads=input_num_heads,
            include_ckv=False,
        )
        workspaces = current_workspace_manager().get_simultaneous(*workspace_specs)
        q_buffer = workspaces[0]
        scratch = workspaces[1]

        if isinstance(q, tuple):
            q_nope, q_pe = q
            q_all = q_buffer[:num_tokens]
            if int(q_pe.shape[-1]) == 0:
                q_all.copy_(q_nope)
            else:
                ops.concat_mla_q(q_nope, q_pe, q_all)
        else:
            q_all = q_buffer[:num_tokens]
            exact_workspace_alias = (
                tuple(q.shape) == tuple(q_all.shape)
                and tuple(q.stride()) == tuple(q_all.stride())
                and q.dtype == q_all.dtype
                and q.device == q_all.device
                and q.untyped_storage().data_ptr() == q_all.untyped_storage().data_ptr()
                and q.storage_offset() == q_all.storage_offset()
            )
            if not exact_workspace_alias:
                q_all.copy_(q)

        if int(q_all.shape[1]) != input_num_heads:
            raise ValueError(
                "B12X sparse MLA query heads do not match the planned head "
                f"count: {q_all.shape[1]} != {input_num_heads}."
            )

        assert self.topk_indices_buffer is not None
        # Inside full CUDA graphs the indexer sorts its selection on a side
        # stream (b12x_topk_sort) while the latent query projection and the
        # query concatenation above run on this stream; the selection is read
        # from this point on.
        b12x_topk_sort.join(self.topk_indices_buffer.device)
        topk_indices = self.topk_indices_buffer[:num_tokens]
        kv_cache_for_run = kv_c_and_k_pe_cache
        ckv_state = None
        ckv_layer_idx = 0
        try:
            if use_ckv_gather:
                kv_cache_for_run, ckv_state, ckv_layer_idx = self._consume_ckv(
                    kv_c_and_k_pe_cache,
                    attn_metadata,
                    layer,
                    current_cache,
                )
                assert attn_metadata.ckv_selected_indices is not None
                assert attn_metadata.ckv_active_counts is not None
                assert attn_metadata.dcp_rank_req_starts is not None
                assert attn_metadata.dcp_rank_req_lens is not None
                selected_indices = attn_metadata.ckv_selected_indices[
                    :num_tokens, : topk_indices.shape[1]
                ]
                active_counts = attn_metadata.ckv_active_counts[:num_tokens]
                _map_global_topk_to_gathered_ckv(
                    attn_metadata.req_id_per_token[:num_tokens],
                    topk_indices,
                    attn_metadata.dcp_rank_req_starts,
                    attn_metadata.dcp_rank_req_lens,
                    selected_indices,
                    active_counts,
                    dcp_size=self.dcp_world_size,
                    cp_kv_cache_interleave_size=(
                        attn_metadata.cp_kv_cache_interleave_size
                    ),
                    padded_rank_tokens=attn_metadata.dcp_padded_total_tokens,
                )
                assert attn_metadata.global_cache_seq_lens_per_req is not None
                cache_seq_lens = _global_causal_lens_for_ckv_gather(
                    attn_metadata.global_cache_seq_lens_per_req,
                    attn_metadata.query_start_loc,
                    attn_metadata.req_id_per_token,
                    num_tokens,
                ).contiguous()
                torch.minimum(active_counts, cache_seq_lens, out=active_counts)
                _mask_page_table_after_nsa_len(selected_indices, active_counts)
            else:
                physical_selection = (
                    self._physical_selection_provider.get_b12x_physical_selection(
                        num_tokens=num_tokens,
                        num_prefills=int(attn_metadata.num_prefills),
                        num_decode_tokens=int(attn_metadata.num_decode_tokens),
                    )
                    if self._physical_selection_provider is not None
                    else None
                )
                if physical_selection is not None:
                    selected_indices, active_counts = physical_selection
                elif self.dcp_world_size > 1:
                    block_stride_rows = _selected_index_block_stride_rows(
                        kv_c_and_k_pe_cache,
                        block_size=attn_metadata.block_size,
                    )
                    selected_indices, active_counts = (
                        triton_filter_and_convert_dcp_index(
                            attn_metadata.req_id_per_token[:num_tokens],
                            attn_metadata.block_table,
                            topk_indices,
                            dcp_size=self.dcp_world_size,
                            dcp_rank=self.dcp_rank,
                            cp_kv_cache_interleave_size=(
                                attn_metadata.cp_kv_cache_interleave_size
                            ),
                            BLOCK_SIZE=attn_metadata.block_size,
                            BLOCK_STRIDE_ROWS=block_stride_rows,
                            NUM_TOPK_TOKENS=topk_indices.shape[1],
                            return_valid_counts=True,
                        )
                    )
                elif not self._is_glm_next:
                    selected_indices = topk_indices
                    cache_seq_lens_per_token = attn_metadata.cache_seq_lens_per_token
                    assert cache_seq_lens_per_token is not None
                    active_counts = cache_seq_lens_per_token[:num_tokens]
                else:
                    block_stride_rows = _selected_index_block_stride_rows(
                        kv_c_and_k_pe_cache,
                        block_size=attn_metadata.block_size,
                    )
                    selected_indices, active_counts = (
                        triton_convert_req_index_to_global_index(
                            attn_metadata.req_id_per_token[:num_tokens],
                            attn_metadata.block_table,
                            topk_indices,
                            BLOCK_SIZE=attn_metadata.block_size,
                            BLOCK_STRIDE_ROWS=block_stride_rows,
                            NUM_TOPK_TOKENS=topk_indices.shape[1],
                            return_valid_counts=True,
                        )
                    )

            if not use_ckv_gather:
                if self._is_glm_next:
                    cache_seq_lens = attn_metadata.cache_seq_lens_per_token
                    assert cache_seq_lens is not None
                    cache_seq_lens = cache_seq_lens[:num_tokens].contiguous()
                else:
                    cache_seq_lens = attn_metadata.seq_lens[
                        : attn_metadata.num_reqs
                    ].contiguous()
            binding = self._bind(
                plan,
                scratch=scratch,
                q=q_all,
                kv_cache=kv_cache_for_run,
                selected_indices=selected_indices,
                cache_lengths=cache_seq_lens,
                selected_lengths=active_counts,
            )
            result = self._run(binding)
            if self.need_to_return_lse_for_decode:
                output, lse = result
                return output, lse
            assert isinstance(result, torch.Tensor)
            return result, None
        finally:
            if ckv_state is not None:
                completion = torch.cuda.Event()
                trace = getattr(self, "_prefill_trace", None)
                if trace is None:
                    completion.record(torch.cuda.current_stream())
                else:
                    trace.record(completion, torch.cuda.current_stream())
                ckv_state.finish_consumer(ckv_layer_idx, completion)
