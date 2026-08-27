# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B12x QSA owner for Qwen3.8-Flash-Next on NVIDIA SM12x."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, ClassVar, cast

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention.attention import set_default_quant_scales
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.rotary_embedding.mrope import triton_mrope
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.b12x import get_b12x_qsa
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    canonicalize_singleton_dim_strides,
    direct_register_custom_op,
    kv_cache_dtype_str_to_dtype,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.b12x import (
    B12xPagedAttentionBackend,
    B12xPagedMetadata,
    B12xPagedMetadataBuilder,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    FullAttentionSpec,
    KVCacheLayout,
    KVCacheSpec,
    get_kv_quant_mode,
)

from ..common.qsa_cache import (
    canonical_qsa_rope_positions,
    qsa_compressed_cache_view,
    qsa_compressed_slot_mapping,
    qsa_logical_positions,
    qsa_padded_page_size_bytes,
)
from ..config import Qwen3_8FlashNextTextConfig
from .indexer_qsa import QSAIndexer

_QSA_COMPRESS_RATIO = 4
_QSA_INDEX_HEAD_DIM = 128
# The supported speculative envelope is zero to four draft tokens. Its raw
# ring is either 4 or 8 rows, so eight is the static manager-page alignment
# that is valid for every supported runtime configuration.
_QSA_MANAGER_BLOCK_ALIGNMENT = 8
_QSA_MAX_SPECULATIVE_TOKENS = 4
_QSA_SPLITTING_OP = "vllm::qwen3_8_flash_next_qsa_with_output"


def _register_qsa_compilation_context(
    compilation_config: Any,
    layer_name: str,
    layer: nn.Module,
) -> None:
    static_context = compilation_config.static_forward_context
    if layer_name in static_context:
        raise ValueError(f"duplicate layer name: {layer_name}")
    static_context[layer_name] = layer
    splitting_ops = compilation_config.splitting_ops
    if splitting_ops is not None and _QSA_SPLITTING_OP not in splitting_ops:
        splitting_ops.append(_QSA_SPLITTING_OP)


def _without_modelopt_fp4(
    quant_config: QuantizationConfig | None,
) -> QuantizationConfig | None:
    if quant_config is not None and quant_config.get_name() == "modelopt_fp4":
        return None
    return quant_config


@dataclass
class Qwen3_8FlashNextQSAMetadata(B12xPagedMetadata):
    """Main-cache metadata plus persistent selector-state ownership."""

    request_ids: torch.Tensor | None = None
    is_prefilling: torch.Tensor | None = None
    has_prefill: bool = False
    qsa_state_slot_ids: torch.Tensor | None = None
    qsa_state_is_fresh: torch.Tensor | None = None
    qsa_num_accepted_tokens: torch.Tensor | None = None


def _qsa_batch_has_prefill(
    common_metadata: CommonAttentionMetadata,
    *,
    max_speculative_tokens: int,
) -> bool:
    num_reqs = int(common_metadata.seq_lens.shape[0])
    query_lengths = (
        common_metadata.query_start_loc_cpu[1 : num_reqs + 1]
        - common_metadata.query_start_loc_cpu[:num_reqs]
    )
    active = query_lengths > 0
    if common_metadata.is_prefilling is None:
        flags = torch.full_like(
            active,
            common_metadata.max_query_len > 1,
            dtype=torch.bool,
        )
    else:
        flags = common_metadata.is_prefilling[:num_reqs].to(device="cpu")
    flags = flags | (query_lengths > 1 + max_speculative_tokens)
    return bool((active & flags).any().item())


def _qsa_prefill_requests(
    metadata: Qwen3_8FlashNextQSAMetadata,
    *,
    max_speculative_tokens: int,
) -> torch.Tensor:
    num_reqs = int(metadata.seq_lens.shape[0])
    query_lengths = (
        metadata.query_start_loc[1 : num_reqs + 1] - metadata.query_start_loc[:num_reqs]
    )
    active = query_lengths > 0
    if metadata.is_prefilling is None:
        flags = torch.full(
            (num_reqs,),
            metadata.max_query_len > 1,
            dtype=torch.bool,
            device=active.device,
        )
    else:
        flags = metadata.is_prefilling[:num_reqs]
        if flags.device != active.device:
            raise RuntimeError("QSA prefill flags were not staged on the GPU")
    # PIECEWISE capture constructs synthetic requests that can exceed the
    # configured speculative decode envelope while marking them non-prefill.
    return flags | (query_lengths > 1 + max_speculative_tokens)


class Qwen3_8FlashNextQSAMetadataBuilder(B12xPagedMetadataBuilder):
    """Build QSA metadata without introducing another KV-cache owner."""

    requires_qsa_metadata: ClassVar[bool] = True
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    supports_draft_decode_metadata_update = True

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        scheduler = vllm_config.scheduler_config
        max_tokens = int(scheduler.max_num_batched_tokens)
        max_reqs = int(scheduler.max_num_seqs)
        self.max_speculative_tokens = int(vllm_config.num_speculative_tokens)
        self._request_ids = torch.empty(max_tokens, dtype=torch.int32, device=device)
        self._capture_state_slot_ids = torch.arange(
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

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        qsa_state_slot_ids: torch.Tensor | None = None,
        qsa_state_is_fresh: torch.Tensor | None = None,
        qsa_num_accepted_tokens: torch.Tensor | None = None,
        qsa_is_prefilling: torch.Tensor | None = None,
    ) -> Qwen3_8FlashNextQSAMetadata:
        del common_prefix_len, fast_build
        cm = common_attn_metadata
        num_reqs = int(cm.seq_lens.shape[0])
        has_prefill = _qsa_batch_has_prefill(
            cm,
            max_speculative_tokens=self.max_speculative_tokens,
        )
        if qsa_state_slot_ids is not None:
            self._capture_state_slot_ids[:num_reqs].copy_(qsa_state_slot_ids[:num_reqs])
        if qsa_state_is_fresh is not None:
            self._capture_state_is_fresh[:num_reqs].copy_(qsa_state_is_fresh[:num_reqs])
        if qsa_num_accepted_tokens is not None:
            self._capture_num_accepted_tokens[:num_reqs].copy_(
                qsa_num_accepted_tokens[:num_reqs]
            )
        qsa_state_slot_ids = self._capture_state_slot_ids[:num_reqs]
        qsa_state_is_fresh = self._capture_state_is_fresh[:num_reqs]
        qsa_num_accepted_tokens = self._capture_num_accepted_tokens[:num_reqs]
        is_prefilling = qsa_is_prefilling
        if is_prefilling is None:
            is_prefilling = cm.is_prefilling
        if is_prefilling is not None:
            self._capture_is_prefilling[:num_reqs].copy_(is_prefilling[:num_reqs])
            is_prefilling = self._capture_is_prefilling[:num_reqs]
        request_ids = cm.token_to_req_indices(self._request_ids)
        num_mapped_tokens = int(cm.query_start_loc_cpu[-1])
        if num_mapped_tokens < cm.num_actual_tokens:
            request_ids[num_mapped_tokens:].fill_(-1)
        return Qwen3_8FlashNextQSAMetadata(
            num_actual_tokens=cm.num_actual_tokens,
            max_query_len=cm.max_query_len,
            query_start_loc=cm.query_start_loc,
            max_seq_len=cm.max_seq_len,
            seq_lens=cm.seq_lens,
            block_table=cm.block_table_tensor,
            slot_mapping=cm.slot_mapping,
            causal=cast(bool, cm.causal),
            request_ids=request_ids,
            is_prefilling=is_prefilling,
            has_prefill=has_prefill,
            qsa_state_slot_ids=qsa_state_slot_ids,
            qsa_state_is_fresh=qsa_state_is_fresh,
            qsa_num_accepted_tokens=qsa_num_accepted_tokens,
        )

    def update_draft_decode_metadata(
        self,
        metadata: B12xPagedMetadata,
    ) -> None:
        qsa_metadata = cast(Qwen3_8FlashNextQSAMetadata, metadata)
        if qsa_metadata.qsa_num_accepted_tokens is None:
            raise RuntimeError(
                "QSA draft decode metadata requires accepted-token counts"
            )
        qsa_metadata.qsa_num_accepted_tokens.fill_(1)

    def update_block_table(
        self,
        metadata: B12xPagedMetadata,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> Qwen3_8FlashNextQSAMetadata:
        if not isinstance(metadata, Qwen3_8FlashNextQSAMetadata):
            raise TypeError(f"expected QSA metadata, got {type(metadata)!r}")
        num_reqs = metadata.seq_lens.shape[0]
        num_tokens = metadata.num_actual_tokens
        assert metadata.request_ids is not None
        assert metadata.qsa_state_slot_ids is not None
        assert metadata.qsa_state_is_fresh is not None
        assert metadata.qsa_num_accepted_tokens is not None
        self._request_ids[:num_tokens].copy_(metadata.request_ids[:num_tokens])
        is_prefilling = None
        if metadata.is_prefilling is not None:
            self._capture_is_prefilling[:num_reqs].copy_(
                metadata.is_prefilling[:num_reqs]
            )
            is_prefilling = self._capture_is_prefilling[:num_reqs]
        self._capture_state_slot_ids[:num_reqs].copy_(
            metadata.qsa_state_slot_ids[:num_reqs]
        )
        self._capture_state_is_fresh[:num_reqs].copy_(
            metadata.qsa_state_is_fresh[:num_reqs]
        )
        self._capture_num_accepted_tokens[:num_reqs].copy_(
            metadata.qsa_num_accepted_tokens[:num_reqs]
        )
        return replace(
            metadata,
            block_table=blk_table,
            slot_mapping=slot_mapping,
            request_ids=self._request_ids[:num_tokens],
            is_prefilling=is_prefilling,
            qsa_state_slot_ids=self._capture_state_slot_ids[:num_reqs],
            qsa_state_is_fresh=self._capture_state_is_fresh[:num_reqs],
            qsa_num_accepted_tokens=self._capture_num_accepted_tokens[:num_reqs],
        )


class Qwen3_8FlashNextQSABackend(B12xPagedAttentionBackend):
    """Sparse QSA backend using one padded, unsplit BLHNC manager page."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto", "bfloat16"]
    forward_includes_kv_cache_update: bool = True

    @staticmethod
    def get_name() -> str:
        return "QWEN38_FLASH_NEXT_B12X_QSA"

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        packed = B12xPagedAttentionBackend.customize_spec(spec)
        return replace(
            packed,
            page_size_padded=qsa_padded_page_size_bytes(
                packed,
                compress_ratio=_QSA_COMPRESS_RATIO,
                index_head_dim=_QSA_INDEX_HEAD_DIM,
            ),
        )

    @classmethod
    def get_impl_cls(cls) -> type[Qwen3_8FlashNextQSAImpl]:
        return Qwen3_8FlashNextQSAImpl

    @staticmethod
    def get_builder_cls() -> type[Qwen3_8FlashNextQSAMetadataBuilder]:
        return Qwen3_8FlashNextQSAMetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(_QSA_MANAGER_BLOCK_ALIGNMENT)]

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        return block_size is None or int(block_size) % _QSA_MANAGER_BLOCK_ALIGNMENT == 0

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        return (
            (int(default_block_size) + _QSA_MANAGER_BLOCK_ALIGNMENT - 1)
            // _QSA_MANAGER_BLOCK_ALIGNMENT
            * _QSA_MANAGER_BLOCK_ALIGNMENT
        )

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [256]

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        # Block-copy copies page padding only when the raw allocation is block
        # outer. LBHNC would copy the logical main view and lose selector tails.
        return (KVCacheLayout.BLHNC,)

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_kv_connector(cls) -> bool:
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
        del use_sparse
        if dtype != torch.bfloat16:
            return "Qwen3.8-Flash-Next QSA requires BF16 queries"
        if kv_cache_dtype not in (None, "auto", "bfloat16"):
            return "Qwen3.8-Flash-Next QSA requires a BF16 KV cache"
        if use_mla or has_sink or use_mm_prefix:
            return "QSA does not support MLA, attention sinks, or MM-prefix attention"
        if not cls.supports_block_size(block_size):
            return (
                "QSA manager block size must be a multiple of "
                f"{_QSA_MANAGER_BLOCK_ALIGNMENT}"
            )
        qsa = get_b12x_qsa()
        if qsa is None:
            return "Install the b12x backend with `pip install vllm[b12x]`"
        if not qsa.is_supported():
            return "b12x QSA is not supported on the current device"
        if (device_capability.major, device_capability.minor) not in (
            (12, 0),
            (12, 1),
        ):
            return "b12x QSA requires SM120 or SM121"
        return None


class Qwen3_8FlashNextQSAImpl(AttentionImpl[Qwen3_8FlashNextQSAMetadata]):
    """Native vLLM main-K/V writer for the merged QSA owner."""

    is_sparse: ClassVar[bool] = True
    supports_dense_mha_prefill: ClassVar[bool] = False
    supports_dcp: bool = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        if num_kv_heads is None:
            num_kv_heads = num_heads
        if alibi_slopes is not None or sliding_window is not None or sinks is not None:
            raise NotImplementedError(
                "QSA does not support ALiBi, sliding window, or sinks"
            )
        if logits_soft_cap not in (None, 0):
            raise NotImplementedError("QSA does not support logits soft cap")
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("QSA supports causal decoder attention only")
        if kv_cache_dtype not in ("auto", "bfloat16"):
            raise NotImplementedError("QSA requires a BF16 main KV cache")
        if head_size != 256 or num_heads % num_kv_heads:
            raise ValueError("QSA requires head_dim=256 and valid grouped-query heads")
        if not math.isclose(scale, head_size**-0.5, rel_tol=1e-5, abs_tol=1e-7):
            raise ValueError("QSA requires canonical head_dim**-0.5 scaling")
        if self.total_cp_world_size > 1:
            raise NotImplementedError("QSA does not support context parallelism")
        self.num_heads = int(num_heads)
        self.head_size = int(head_size)
        self.output_head_size = self.head_size
        self.scale = float(scale)
        self.num_kv_heads = int(num_kv_heads)
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.supports_quant_query_input = False

    def _kv_cache_views(
        self, kv_cache: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key_cache, value_cache = kv_cache.unbind(1)
        key_cache = key_cache.unflatten(-1, (self.num_kv_heads, self.head_size))
        value_cache = value_cache.unflatten(-1, (self.num_kv_heads, self.head_size))
        key_cache = canonicalize_singleton_dim_strides(key_cache)
        value_cache = canonicalize_singleton_dim_strides(value_cache)
        if key_cache.dtype != torch.bfloat16 or value_cache.dtype != torch.bfloat16:
            raise TypeError("QSA main K/V cache views must be BF16")
        return key_cache, value_cache

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        torch.ops._C_cache_ops.reshape_and_cache_flash(
            key,
            value,
            *self._kv_cache_views(kv_cache),
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: Qwen3_8FlashNextQSAMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del (
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale,
            output_block_scale,
        )
        raise RuntimeError("QSA must run through its merged write-before-read owner")


def apply_qsa_rope(
    rotary_emb: nn.Module,
    positions: torch.Tensor,
    tensor: torch.Tensor,
) -> torch.Tensor:
    """Apply the main attention's exact scalar/MRoPE composition to QSA."""

    num_tokens, _, head_dim = tensor.shape
    rotary_dim = int(rotary_emb.rotary_dim)
    cache = rotary_emb._match_cos_sin_cache_dtype(tensor)  # noqa: SLF001
    cos_sin = cache[positions]
    cos, sin = cos_sin.chunk(2, dim=-1)
    if positions.ndim == 2:
        shape = tensor.shape
        rotated, _ = triton_mrope(
            tensor.reshape(num_tokens, -1),
            tensor.new_empty((num_tokens, head_dim)),
            cos,
            sin,
            rotary_emb.mrope_section,
            head_dim,
            rotary_dim,
            rotary_emb.mrope_interleaved,
            rotary_emb.is_neox_style,
        )
        return rotated.reshape(shape)
    rotated = rotary_emb.apply_rotary_emb.forward_cuda(
        tensor[..., :rotary_dim], cos, sin
    )
    return torch.cat((rotated, tensor[..., rotary_dim:]), dim=-1)


@triton.jit(do_not_specialize=["num_reqs"])
def _reset_fresh_qsa_state_kernel(
    state_slot_ids,
    state_is_fresh,
    sequence_lengths,
    query_start_loc,
    num_accepted_tokens,
    raw_k_ring,
    raw_logical_positions,
    raw_rope_positions,
    raw_interval_start_positions,
    raw_k_slot_stride,
    raw_k_ring_stride,
    raw_tag_slot_stride,
    raw_rope_slot_stride,
    raw_rope_ring_stride,
    raw_interval_stride,
    num_reqs,
    MAX_STATE_SLOTS: tl.constexpr,
    RING_CAPACITY: tl.constexpr,
    INDEX_HEAD_DIM: tl.constexpr,
    POSITION_AXES: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    request = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    active = request < num_reqs
    fresh = tl.load(state_is_fresh + request, mask=active, other=0)
    state_slot = tl.load(state_slot_ids + request, mask=active, other=-1).to(tl.int64)
    query_start = tl.load(query_start_loc + request, mask=active, other=0)
    query_end = tl.load(query_start_loc + request + 1, mask=active, other=0)
    valid = (
        active
        & fresh
        & (query_end > query_start)
        & (state_slot >= 0)
        & (state_slot < MAX_STATE_SLOTS)
    )

    key_rows = offsets // INDEX_HEAD_DIM
    key_dims = offsets % INDEX_HEAD_DIM
    key_mask = valid & (key_rows < RING_CAPACITY)
    tl.store(
        raw_k_ring
        + state_slot * raw_k_slot_stride
        + key_rows * raw_k_ring_stride
        + key_dims,
        0.0,
        mask=key_mask,
    )
    tag_mask = valid & (offsets < RING_CAPACITY)
    tl.store(
        raw_logical_positions + state_slot * raw_tag_slot_stride + offsets,
        -1,
        mask=tag_mask,
    )
    rope_rows = offsets // POSITION_AXES
    rope_axes = offsets % POSITION_AXES
    rope_mask = valid & (rope_rows < RING_CAPACITY)
    tl.store(
        raw_rope_positions
        + state_slot * raw_rope_slot_stride
        + rope_rows * raw_rope_ring_stride
        + rope_axes,
        -1,
        mask=rope_mask,
    )
    if tl.program_id(1) == 0:
        sequence_length = tl.load(sequence_lengths + request, mask=valid, other=0)
        accepted = tl.load(num_accepted_tokens + request, mask=valid, other=1)
        first_position = sequence_length - (query_end - query_start)
        anchor = tl.where(query_end > query_start, first_position - accepted, -1)
        tl.store(
            raw_interval_start_positions + state_slot * raw_interval_stride,
            anchor.to(tl.int64),
            mask=valid,
        )


@triton.jit
def _stage_qsa_request_state_kernel(
    source_state_slot_ids,
    source_state_is_fresh,
    source_num_accepted_tokens,
    query_start_loc,
    state_slot_ids,
    state_is_fresh,
    num_accepted_tokens,
    num_reqs,
) -> None:
    request = tl.program_id(0)
    in_range = request < num_reqs
    query_start = tl.load(query_start_loc + request, mask=in_range, other=0)
    query_end = tl.load(query_start_loc + request + 1, mask=in_range, other=0)
    active = in_range & (query_end > query_start)
    state_slot = tl.load(source_state_slot_ids + request, mask=active, other=-1)
    fresh = tl.load(source_state_is_fresh + request, mask=active, other=0)
    accepted = tl.load(source_num_accepted_tokens + request, mask=active, other=0)
    tl.store(state_slot_ids + request, tl.where(active, state_slot, -1))
    tl.store(state_is_fresh + request, tl.where(active, fresh, 0))
    tl.store(num_accepted_tokens + request, tl.where(active, accepted, 0))


@triton.jit
def _stage_qsa_rope_positions_kernel(
    source,
    request_ids,
    output,
    source_row_stride,
    source_axis_stride,
    output_row_stride,
    rows,
    POSITION_AXES: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    axes = tl.arange(0, 4)
    axis_mask = axes < POSITION_AXES
    active = row < rows
    request_id = tl.load(request_ids + row, mask=active, other=-1)
    positions = tl.load(
        source + row * source_row_stride + axes * source_axis_stride,
        mask=active & axis_mask,
        other=-1,
    )
    positions = tl.where(request_id >= 0, positions, -1)
    tl.store(
        output + row * output_row_stride + axes,
        positions,
        mask=active & axis_mask,
    )


@triton.jit
def _commit_prefill_qsa_state_kernel(
    raw_index_key,
    rope_positions,
    is_prefilling,
    sequence_lengths,
    query_start_loc,
    state_slot_ids,
    raw_k_ring,
    raw_logical_positions,
    raw_rope_positions,
    raw_interval_start_positions,
    raw_key_row_stride,
    rope_row_stride,
    rope_axis_stride,
    raw_k_slot_stride,
    raw_k_ring_stride,
    raw_tag_slot_stride,
    raw_rope_slot_stride,
    raw_rope_ring_stride,
    raw_interval_stride,
    num_reqs,
    MAX_STATE_SLOTS: tl.constexpr,
    RING_CAPACITY: tl.constexpr,
    INDEX_HEAD_DIM: tl.constexpr,
    POSITION_AXES: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    """Commit each request's final ring-capacity rows without slot races."""

    request = tl.program_id(0)
    suffix_offset = tl.program_id(1)
    active_request = request < num_reqs
    query_start = tl.load(query_start_loc + request, mask=active_request, other=0)
    query_end = tl.load(query_start_loc + request + 1, mask=active_request, other=0)
    query_length = query_end - query_start
    suffix_length = tl.minimum(query_length, RING_CAPACITY)
    row = query_end - suffix_length + suffix_offset
    state_slot = tl.load(state_slot_ids + request, mask=active_request, other=-1).to(
        tl.int64
    )
    valid = (
        active_request
        & (suffix_offset < suffix_length)
        & (state_slot >= 0)
        & (state_slot < MAX_STATE_SLOTS)
    )
    sequence_length = tl.load(
        sequence_lengths + request, mask=active_request, other=0
    ).to(tl.int64)
    request_is_prefilling = tl.load(
        is_prefilling + request, mask=active_request, other=0
    )
    logical_position = sequence_length - (query_end - row)
    ring_slot = logical_position % RING_CAPACITY

    dims = tl.arange(0, BLOCK_D)
    key = tl.load(
        raw_index_key + row * raw_key_row_stride + dims,
        mask=valid & (dims < INDEX_HEAD_DIM),
        other=0.0,
    )
    tl.store(
        raw_k_ring
        + state_slot * raw_k_slot_stride
        + ring_slot * raw_k_ring_stride
        + dims,
        key,
        mask=valid & (dims < INDEX_HEAD_DIM),
    )
    tl.store(
        raw_logical_positions + state_slot * raw_tag_slot_stride + ring_slot,
        logical_position,
        mask=valid,
    )

    axes = tl.arange(0, 4)
    rope = tl.load(
        rope_positions + row * rope_row_stride + axes * rope_axis_stride,
        mask=valid & (axes < POSITION_AXES),
        other=-1,
    )
    tl.store(
        raw_rope_positions
        + state_slot * raw_rope_slot_stride
        + ring_slot * raw_rope_ring_stride
        + axes,
        rope,
        mask=valid & (axes < POSITION_AXES),
    )
    if suffix_offset == 0:
        interval_anchor = tl.where(
            request_is_prefilling,
            sequence_length - 1,
            sequence_length - query_length,
        )
        tl.store(
            raw_interval_start_positions + state_slot * raw_interval_stride,
            interval_anchor,
            mask=valid,
        )


class Qwen3_8FlashNextQSAAttention(nn.Module, AttentionLayerBase):
    """Merged main attention, selector state, and b12x decode transaction."""

    supports_dcp = False

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config: Qwen3_8FlashNextTextConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config
        if cache_config is None:
            raise ValueError("QSA requires a paged KV cache")
        if model_config.dtype != torch.bfloat16:
            raise NotImplementedError("QSA currently requires BF16 activations")
        if cache_config.cache_dtype not in ("auto", "bfloat16"):
            raise NotImplementedError("QSA requires a BF16 main KV cache")
        if getattr(quant_config, "kv_cache_scheme", None) is not None:
            raise NotImplementedError("QSA does not support KV-cache quantization")
        parallel = vllm_config.parallel_config
        if (
            parallel.prefill_context_parallel_size > 1
            or parallel.decode_context_parallel_size > 1
        ):
            raise NotImplementedError("QSA does not support context parallelism")
        if not getattr(config, "is_causal", True):
            raise NotImplementedError("QSA requires causal decoder attention")
        if getattr(config, "dual_chunk_attention_config", None) is not None:
            raise NotImplementedError("QSA does not support dual-chunk RoPE")
        if (
            int(config.indexer_compress_ratio) != _QSA_COMPRESS_RATIO
            or int(config.indexer_head_dim) != _QSA_INDEX_HEAD_DIM
        ):
            raise NotImplementedError(
                "The SM12x QSA integration requires compress_ratio=4 and "
                "index_head_dim=128"
            )

        self.config = config
        self.layer_id = int(layer_id)
        self.hidden_size = int(config.hidden_size)
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = int(config.num_attention_heads)
        if self.total_num_heads % tp_size:
            raise ValueError("QSA query heads must divide across TP ranks")
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = int(config.num_key_value_heads)
        if self.total_num_kv_heads >= tp_size:
            if self.total_num_kv_heads % tp_size:
                raise ValueError("QSA KV heads must divide across TP ranks")
        elif tp_size % self.total_num_kv_heads:
            raise ValueError("TP size must divide replicated QSA KV heads")
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = int(config.head_dim)
        if self.head_dim != 256:
            raise NotImplementedError("The SM12x QSA integration requires head_dim=256")
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.attn_output_gate = True

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads * 2,
            self.total_num_kv_heads,
            bias=False,
            quant_config=_without_modelopt_fp4(quant_config),
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters,
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=float(config.rms_norm_eps))
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=float(config.rms_norm_eps))
        self.indexer = QSAIndexer(
            vllm_config=vllm_config,
            config=config,
            quant_config=quant_config,
            prefix=f"{prefix}.indexer",
        )

        self.layer_name = f"{prefix}.attn"
        self.attn_type = AttentionType.DECODER
        self.kv_cache_dtype = cache_config.cache_dtype
        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
            self.kv_cache_dtype, model_config
        )
        if self.kv_cache_torch_dtype != torch.bfloat16:
            raise NotImplementedError("QSA cache storage must resolve to BF16")
        self.kv_sharing_target_layer_name = None
        self.kv_cache = torch.tensor([])
        set_default_quant_scales(self, register_buffer=True)

        self.attn_backend = Qwen3_8FlashNextQSABackend
        self.impl = Qwen3_8FlashNextQSAImpl(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            None,
            None,
            self.kv_cache_dtype,
            None,
            AttentionType.DECODER,
            None,
        )

        scheduler = vllm_config.scheduler_config
        self.max_tokens = int(scheduler.max_num_batched_tokens)
        self.max_seqs = int(scheduler.max_num_seqs)
        self.max_seq_len = int(model_config.max_model_len)
        self.max_speculative_tokens = int(vllm_config.num_speculative_tokens)
        if self.max_speculative_tokens > _QSA_MAX_SPECULATIVE_TOKENS:
            raise NotImplementedError(
                "QSA currently supports at most four speculative tokens"
            )
        self.compress_ratio = int(config.indexer_compress_ratio)
        self.budget = int(config.indexer_budget)
        self.index_heads = int(config.indexer_n_heads)
        self.index_head_dim = int(config.indexer_head_dim)
        self.position_axes = 3 if model_config.uses_mrope else 1
        self.raw_ring_capacity = self.compress_ratio * math.ceil(
            (self.compress_ratio + self.max_speculative_tokens) / self.compress_ratio
        )
        self.max_decode_rows = self.max_seqs * (1 + self.max_speculative_tokens)
        device = torch.device("cuda", torch.accelerator.current_device_index())
        self.device = device

        self.register_buffer(
            "_raw_k_ring",
            torch.zeros(
                self.max_seqs,
                self.raw_ring_capacity,
                self.index_head_dim,
                dtype=torch.bfloat16,
                device=device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_raw_logical_positions",
            torch.full(
                (self.max_seqs, self.raw_ring_capacity),
                -1,
                dtype=torch.int64,
                device=device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_raw_rope_positions",
            torch.full(
                (self.max_seqs, self.raw_ring_capacity, self.position_axes),
                -1,
                dtype=torch.int64,
                device=device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_raw_interval_start_positions",
            torch.full((self.max_seqs,), -1, dtype=torch.int64, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_raw_interval_start_snapshot",
            torch.empty(self.max_seqs, dtype=torch.int64, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_decode_output",
            torch.empty(
                self.max_decode_rows,
                self.num_heads,
                self.head_dim,
                dtype=torch.bfloat16,
                device=device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_query_input",
            torch.empty(
                self.max_decode_rows,
                self.num_heads,
                self.head_dim,
                dtype=torch.bfloat16,
                device=device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_rope_position_input",
            torch.empty(
                self.max_decode_rows,
                self.position_axes,
                dtype=torch.int64,
                device=device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_selected_positions",
            torch.empty(
                self.max_tokens,
                self.budget + self.compress_ratio - 1,
                dtype=torch.int32,
                device=device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_index_query_input",
            torch.empty(
                self.max_tokens,
                self.index_heads,
                self.index_head_dim,
                dtype=torch.bfloat16,
                device=device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_raw_index_key_input",
            torch.empty(
                self.max_tokens,
                self.index_head_dim,
                dtype=torch.bfloat16,
                device=device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_sequence_lengths",
            torch.zeros(self.max_seqs, dtype=torch.int32, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_query_start_loc",
            torch.zeros(self.max_seqs + 1, dtype=torch.int32, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_raw_state_slot_ids",
            torch.full((self.max_seqs,), -1, dtype=torch.int32, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_state_is_fresh",
            torch.zeros(self.max_seqs, dtype=torch.bool, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_num_accepted_tokens",
            torch.ones(self.max_seqs, dtype=torch.int32, device=device),
            persistent=False,
        )
        self._main_block_table: torch.Tensor | None = None
        self._compressed_cache: torch.Tensor | None = None
        self._qsa_plan: Any | None = None
        self._qsa_binding: Any | None = None
        self._qsa_scratch: torch.Tensor | None = None
        self._b12x_diagnostic_request_ids: torch.Tensor | None = None

        _register_qsa_compilation_context(
            vllm_config.compilation_config,
            self.layer_name,
            self,
        )

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.attn_backend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        base = FullAttentionSpec(
            block_size=int(vllm_config.cache_config.block_size),
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            head_size_v=self.head_dim,
            dtype=self.kv_cache_torch_dtype,
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
        )
        return self.attn_backend.customize_spec(base)

    def snapshot_speculative_interval_starts(self) -> None:
        self._raw_interval_start_snapshot.copy_(self._raw_interval_start_positions)

    def restore_speculative_interval_starts(self) -> None:
        self._raw_interval_start_positions.copy_(self._raw_interval_start_snapshot)

    def _project_qkv_gate(
        self,
        qkv: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q_gate, key, value = qkv.split(
            (self.q_size * 2, self.kv_size, self.kv_size), dim=-1
        )
        q_gate = q_gate.unflatten(-1, (self.num_heads, self.head_dim * 2))
        query, gate = q_gate.chunk(2, dim=-1)
        query = self.q_norm(query).flatten(-2)
        key = self.k_norm(
            key.unflatten(-1, (self.num_kv_heads, self.head_dim))
        ).flatten(-2)
        query, key = self.rotary_emb(positions, query, key)
        return query, key, value, gate.flatten(-2)

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        super().bind_kv_cache(kv_cache)
        if kv_cache.ndim != 4 or int(kv_cache.shape[1]) != 2:
            raise ValueError(
                "QSA requires a native B12x [pages,2,page,packed_kv] cache"
            )
        main_page_size = int(kv_cache.shape[2])
        if main_page_size % self.raw_ring_capacity:
            raise ValueError(
                "QSA manager block must be divisible by raw_ring_capacity: "
                f"{main_page_size} % {self.raw_ring_capacity} != 0"
            )
        if main_page_size % _QSA_MANAGER_BLOCK_ALIGNMENT:
            raise ValueError("QSA manager block does not satisfy the backend alignment")
        if self.max_decode_rows > self.max_tokens:
            raise ValueError(
                "QSA max_num_batched_tokens must cover the largest verifier batch"
            )

        impl = cast(Qwen3_8FlashNextQSAImpl, self.impl)
        main_k_cache, main_v_cache = impl._kv_cache_views(kv_cache)
        compressed_cache = qsa_compressed_cache_view(
            kv_cache,
            compress_ratio=self.compress_ratio,
            index_head_dim=self.index_head_dim,
        )
        self._compressed_cache = compressed_cache
        compressed_page_size = int(compressed_cache.shape[1])
        qsa_max_seq_len = (
            (self.max_seq_len + self.compress_ratio - 1)
            // self.compress_ratio
            * self.compress_ratio
        )
        table_width = math.ceil(qsa_max_seq_len / main_page_size)
        compressed_table_width = math.ceil(
            (qsa_max_seq_len // self.compress_ratio) / compressed_page_size
        )
        if compressed_table_width != table_width:
            raise RuntimeError(
                "QSA packed main/compressed caches require one shared table width"
            )
        # CUDA-graph memory profiling binds a deliberately small throwaway
        # cache. Caps describe the maximum cache page counts admitted by the
        # plan; bind() separately validates actual tensors against those maxima.
        planned_main_cache_pages = max(int(main_k_cache.shape[0]), table_width)
        planned_compressed_cache_pages = max(
            int(compressed_cache.shape[0]), compressed_table_width
        )
        self._main_block_table = torch.full(
            (self.max_seqs, table_width),
            -1,
            dtype=torch.int32,
            device=kv_cache.device,
        )

        qsa = get_b12x_qsa()
        if qsa is None or not qsa.is_supported():
            raise RuntimeError("b12x QSA is unavailable on the current device")
        sections = None
        if self.position_axes == 3:
            mrope_section = getattr(self.rotary_emb, "mrope_section", None)
            if mrope_section is None:
                raise RuntimeError("QSA M-RoPE requires mrope_section")
            sections = tuple(map(int, mrope_section))
        plan = qsa.plan(
            qsa.Caps(
                device=kv_cache.device,
                max_batch=self.max_seqs,
                max_raw_state_slots=self.max_seqs,
                max_q_rows=self.max_decode_rows,
                max_seq_len=qsa_max_seq_len,
                num_main_cache_pages=planned_main_cache_pages,
                num_compressed_cache_pages=planned_compressed_cache_pages,
                main_page_size=main_page_size,
                compressed_page_size=compressed_page_size,
                max_speculative_tokens=self.max_speculative_tokens,
                q_heads=self.num_heads,
                kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                index_heads=self.index_heads,
                index_kv_heads=1,
                index_head_dim=self.index_head_dim,
                index_rotary_dim=int(self.rotary_emb.rotary_dim),
                compress_ratio=self.compress_ratio,
                budget=self.budget,
                position_axes=self.position_axes,
                mrope_sections=sections,
                mrope_interleaved=bool(
                    getattr(self.rotary_emb, "mrope_interleaved", False)
                ),
                rms_norm_eps=float(self.indexer.q_layernorm.variance_epsilon),
                dtype=torch.bfloat16,
            )
        )
        scratch_spec = plan.scratch_specs()[0]
        scratch = torch.empty(
            scratch_spec.shape,
            dtype=scratch_spec.dtype,
            device=kv_cache.device,
        )
        cos_sin = self.rotary_emb.cos_sin_cache
        rope_cos, rope_sin = cos_sin.chunk(2, dim=-1)
        expected_rope_width = int(self.rotary_emb.rotary_dim) // 2
        if (
            int(rope_cos.shape[1]) != expected_rope_width
            or rope_sin.shape != rope_cos.shape
        ):
            raise RuntimeError("QSA received an unexpected main RoPE cache layout")
        binding = plan.bind(
            scratch=scratch,
            main_k_cache=main_k_cache,
            main_v_cache=main_v_cache,
            main_block_table=self._main_block_table,
            compressed_k_cache=compressed_cache,
            compressed_block_table=self._main_block_table,
            raw_k_ring=self._raw_k_ring,
            raw_logical_positions=self._raw_logical_positions,
            raw_rope_positions=self._raw_rope_positions,
            raw_interval_start_positions=self._raw_interval_start_positions,
            raw_state_slot_ids=self._raw_state_slot_ids,
            index_q_norm_weight=self.indexer.q_layernorm.weight,
            index_k_norm_weight=self.indexer.k_layernorm.weight,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            output=self._decode_output,
            selected_positions=self._selected_positions[: self.max_decode_rows],
        )
        self._qsa_plan = plan
        self._qsa_binding = binding
        self._qsa_scratch = scratch

    def unbind_kv_cache(self) -> None:
        self._qsa_binding = None
        self._qsa_plan = None
        self._qsa_scratch = None
        self._compressed_cache = None
        self._main_block_table = None
        super().unbind_kv_cache()

    @staticmethod
    def _active_positions(positions: torch.Tensor, rows: int) -> torch.Tensor:
        return positions[:rows] if positions.ndim == 1 else positions[:, :rows]

    @staticmethod
    def _require_qsa_metadata(
        metadata: Qwen3_8FlashNextQSAMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fields = (
            metadata.request_ids,
            metadata.qsa_state_slot_ids,
            metadata.qsa_state_is_fresh,
            metadata.qsa_num_accepted_tokens,
        )
        if any(tensor is None for tensor in fields):
            raise RuntimeError("QSA runtime metadata is incomplete")
        return cast(
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], fields
        )

    def _reset_fresh_selector_state(
        self,
        *,
        state_slot_ids: torch.Tensor,
        state_is_fresh: torch.Tensor,
        sequence_lengths: torch.Tensor,
        query_start_loc: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        num_reqs: int,
    ) -> None:
        if num_reqs <= 0:
            return
        if not HAS_TRITON:
            raise RuntimeError("QSA state reset requires Triton")
        block = 256
        elements = self.raw_ring_capacity * self.index_head_dim
        _reset_fresh_qsa_state_kernel[(num_reqs, triton.cdiv(elements, block))](
            state_slot_ids,
            state_is_fresh,
            sequence_lengths,
            query_start_loc,
            num_accepted_tokens,
            self._raw_k_ring,
            self._raw_logical_positions,
            self._raw_rope_positions,
            self._raw_interval_start_positions,
            self._raw_k_ring.stride(0),
            self._raw_k_ring.stride(1),
            self._raw_logical_positions.stride(0),
            self._raw_rope_positions.stride(0),
            self._raw_rope_positions.stride(1),
            self._raw_interval_start_positions.stride(0),
            num_reqs,
            MAX_STATE_SLOTS=self.max_seqs,
            RING_CAPACITY=self.raw_ring_capacity,
            INDEX_HEAD_DIM=self.index_head_dim,
            POSITION_AXES=self.position_axes,
            BLOCK=block,
            num_warps=4,
        )

    def _stage_runtime_metadata(
        self,
        metadata: Qwen3_8FlashNextQSAMetadata,
        rows: int,
        *,
        row_capacity: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
    ]:
        if self._main_block_table is None:
            raise RuntimeError("QSA main cache is not bound")
        request_ids, state_slots, state_is_fresh, accepted = self._require_qsa_metadata(
            metadata
        )
        num_reqs = int(metadata.seq_lens.shape[0])
        if rows > row_capacity or num_reqs > self.max_seqs:
            raise ValueError("QSA batch exceeds its planned capacity")
        table_width = int(self._main_block_table.shape[1])
        if int(metadata.block_table.shape[1]) < table_width:
            raise ValueError("QSA block table is narrower than the planned context")

        self._main_block_table.fill_(-1)
        self._main_block_table[:num_reqs].copy_(
            metadata.block_table[:num_reqs, :table_width]
        )
        self._sequence_lengths.zero_()
        self._sequence_lengths[:num_reqs].copy_(metadata.seq_lens[:num_reqs])
        self._query_start_loc.fill_(rows)
        self._query_start_loc[: num_reqs + 1].copy_(
            metadata.query_start_loc[: num_reqs + 1]
        )
        self._raw_state_slot_ids.fill_(-1)
        self._state_is_fresh.zero_()
        self._num_accepted_tokens.zero_()
        if num_reqs:
            _stage_qsa_request_state_kernel[(num_reqs,)](
                state_slots,
                state_is_fresh,
                accepted,
                self._query_start_loc,
                self._raw_state_slot_ids,
                self._state_is_fresh,
                self._num_accepted_tokens,
                num_reqs,
                num_warps=1,
            )
        active_request_ids = request_ids[:rows]
        logical_positions = qsa_logical_positions(
            sequence_lengths=metadata.seq_lens[:num_reqs],
            query_start_loc=metadata.query_start_loc[: num_reqs + 1],
            request_ids=active_request_ids,
        )
        return (
            active_request_ids,
            logical_positions,
            self._raw_state_slot_ids[:num_reqs],
            self._state_is_fresh[:num_reqs],
            self._num_accepted_tokens[:num_reqs],
            self._query_start_loc[: num_reqs + 1],
            num_reqs,
        )

    def _prepare_decode_metadata(
        self,
        metadata: Qwen3_8FlashNextQSAMetadata,
        rows: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        (
            request_ids,
            logical_positions,
            state_slots,
            state_is_fresh,
            accepted,
            query_start_loc,
            num_reqs,
        ) = self._stage_runtime_metadata(
            metadata,
            rows,
            row_capacity=self.max_decode_rows,
        )
        self._reset_fresh_selector_state(
            state_slot_ids=state_slots,
            state_is_fresh=state_is_fresh,
            sequence_lengths=self._sequence_lengths,
            query_start_loc=query_start_loc,
            num_accepted_tokens=accepted,
            num_reqs=num_reqs,
        )
        return request_ids, logical_positions

    @staticmethod
    def _require_prefill_invariant(
        condition: torch.Tensor,
        message: str,
    ) -> None:
        # The portable prefill is deliberately an eager reference path. A
        # host-visible validation barrier keeps every cache mutation after all
        # continuity and page-table checks.
        if not bool(condition.all().item()):
            raise RuntimeError(message)

    def _validate_portable_prefill(
        self,
        metadata: Qwen3_8FlashNextQSAMetadata,
        *,
        request_ids: torch.Tensor,
        logical_positions: torch.Tensor,
        state_slots: torch.Tensor,
        state_is_fresh: torch.Tensor,
        accepted: torch.Tensor,
        is_prefilling: torch.Tensor,
        query_start_loc: torch.Tensor,
        num_reqs: int,
        rows: int,
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        if self._main_block_table is None:
            raise RuntimeError("QSA main cache is not bound")
        mapped_rows = int(query_start_loc[num_reqs].item())
        if not 0 < mapped_rows <= rows:
            raise RuntimeError("QSA prefill must contain at least one mapped row")

        active_request_ids = request_ids[:mapped_rows]
        active_logical_positions = logical_positions[:mapped_rows]
        query_lengths = query_start_loc[1 : num_reqs + 1] - query_start_loc[:num_reqs]
        active_requests = query_lengths > 0
        expected_request_ids = torch.repeat_interleave(
            torch.arange(
                num_reqs,
                dtype=torch.int32,
                device=request_ids.device,
            ),
            query_lengths,
            output_size=mapped_rows,
        )
        self._require_prefill_invariant(
            active_request_ids == expected_request_ids,
            "QSA prefill rows are not dense request intervals",
        )
        if mapped_rows < rows:
            self._require_prefill_invariant(
                request_ids[mapped_rows:rows] == -1,
                "QSA padded rows must carry request ID -1",
            )

        active_slots = state_slots[active_requests]
        self._require_prefill_invariant(
            (active_slots >= 0) & (active_slots < self.max_seqs),
            "QSA prefill received an invalid persistent state slot",
        )
        if int(torch.unique(active_slots).numel()) != int(active_slots.numel()):
            raise RuntimeError("QSA prefill state slots must be unique")

        active_lengths = query_lengths[active_requests].to(torch.int64)
        sequence_lengths = self._sequence_lengths[:num_reqs][active_requests].to(
            torch.int64
        )
        first_positions = sequence_lengths - active_lengths
        active_fresh = state_is_fresh[active_requests]
        active_accepted = accepted[active_requests].to(torch.int64)
        self._require_prefill_invariant(
            (first_positions >= 0) & (sequence_lengths <= self.max_seq_len),
            "QSA prefill sequence positions exceed the planned context",
        )
        self._require_prefill_invariant(
            (active_accepted >= 1)
            & (active_accepted <= self.max_speculative_tokens + 1),
            "QSA prefill received an invalid accepted-token count",
        )
        active_prefilling = is_prefilling[:num_reqs][active_requests].to(
            device=active_accepted.device
        )
        self._require_prefill_invariant(
            ~active_prefilling | (active_accepted == 1),
            "QSA prefill requests must commit exactly one token",
        )
        self._require_prefill_invariant(
            ~active_fresh | (first_positions.remainder(self.compress_ratio) == 0),
            "A fresh QSA prefix must end on a complete compression group",
        )
        anchors = self._raw_interval_start_positions.index_select(
            0, active_slots.to(torch.long)
        )
        continues_decode_interval = (~active_prefilling) & (
            first_positions == anchors + active_accepted
        )
        continues_prefill_interval = (
            active_prefilling
            & (active_accepted == 1)
            & (first_positions == anchors + 1)
        )
        valid_continuation = (
            active_fresh | continues_decode_interval | continues_prefill_interval
        )
        if not bool(valid_continuation.all().item()):
            raise RuntimeError(
                "QSA prefill does not continue the committed selector interval: "
                f"slots={active_slots.tolist()}, "
                f"first_positions={first_positions.tolist()}, "
                f"anchors={anchors.tolist()}, "
                f"accepted={active_accepted.tolist()}, "
                f"prefilling={active_prefilling.tolist()}, "
                f"fresh={active_fresh.tolist()}"
            )
        state_reset_mask = state_is_fresh[:num_reqs].clone()

        compressed_slots = qsa_compressed_slot_mapping(
            block_table=self._main_block_table,
            request_ids=active_request_ids,
            logical_positions=active_logical_positions,
            main_page_size=int(self.kv_cache.shape[2]),
            compress_ratio=self.compress_ratio,
        )
        completed = (active_logical_positions >= self.compress_ratio - 1) & (
            (active_logical_positions + 1).remainder(self.compress_ratio) == 0
        )
        self._require_prefill_invariant(
            ~completed | (compressed_slots >= 0),
            "QSA prefill cannot map a completed group into the packed cache",
        )
        self._require_prefill_invariant(
            metadata.slot_mapping[:mapped_rows] >= 0,
            "QSA prefill cannot map a main K/V row into the cache",
        )

        row_requests = active_request_ids.to(torch.long)
        request_first_positions = self._sequence_lengths[:num_reqs].to(
            torch.int64
        ) - query_lengths.to(torch.int64)
        row_first_positions = request_first_positions.index_select(0, row_requests)
        group_offsets = torch.arange(
            self.compress_ratio,
            dtype=torch.int64,
            device=active_logical_positions.device,
        )
        source_positions = (
            active_logical_positions[:, None] - self.compress_ratio + 1 + group_offsets
        )
        needs_history = completed[:, None] & (
            source_positions < row_first_positions[:, None]
        )
        row_state_slots = state_slots.index_select(0, row_requests).to(torch.long)
        observed_tags = self._raw_logical_positions[
            row_state_slots[:, None],
            source_positions.remainder(self.raw_ring_capacity),
        ]
        self._require_prefill_invariant(
            ~needs_history | (observed_tags == source_positions),
            "QSA prefill cannot complete a group from the committed raw ring",
        )
        if self.position_axes == 1:
            observed_rope = self._raw_rope_positions[
                row_state_slots[:, None],
                source_positions.remainder(self.raw_ring_capacity),
                0,
            ]
            self._require_prefill_invariant(
                ~needs_history | (observed_rope == source_positions),
                "QSA scalar-RoPE history does not match its logical tags",
            )
        return mapped_rows, compressed_slots, state_reset_mask

    def _commit_portable_prefill_state(
        self,
        *,
        raw_index_key: torch.Tensor,
        rope_positions: torch.Tensor,
        is_prefilling: torch.Tensor,
        state_slots: torch.Tensor,
        query_start_loc: torch.Tensor,
        num_reqs: int,
    ) -> None:
        if not HAS_TRITON:
            raise RuntimeError("QSA portable prefill state updates require Triton")
        _commit_prefill_qsa_state_kernel[(num_reqs, self.raw_ring_capacity)](
            raw_index_key,
            rope_positions,
            is_prefilling,
            self._sequence_lengths,
            query_start_loc,
            state_slots,
            self._raw_k_ring,
            self._raw_logical_positions,
            self._raw_rope_positions,
            self._raw_interval_start_positions,
            raw_index_key.stride(0),
            rope_positions.stride(0),
            rope_positions.stride(1),
            self._raw_k_ring.stride(0),
            self._raw_k_ring.stride(1),
            self._raw_logical_positions.stride(0),
            self._raw_rope_positions.stride(0),
            self._raw_rope_positions.stride(1),
            self._raw_interval_start_positions.stride(0),
            num_reqs,
            MAX_STATE_SLOTS=self.max_seqs,
            RING_CAPACITY=self.raw_ring_capacity,
            INDEX_HEAD_DIM=self.index_head_dim,
            POSITION_AXES=self.position_axes,
            BLOCK_D=triton.next_power_of_2(self.index_head_dim),
            num_warps=4,
        )

    def _run_portable_qsa(
        self,
        *,
        metadata: Qwen3_8FlashNextQSAMetadata,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        index_query: torch.Tensor,
        raw_index_key: torch.Tensor,
        is_prefilling: torch.Tensor,
        output: torch.Tensor,
        rows: int,
    ) -> None:
        if self._compressed_cache is None or self._main_block_table is None:
            raise RuntimeError("QSA caches are not bound")
        # CUDA-graph profiling executes split ops with every synthetic token
        # marked as padding. Native K/V writes already treat slot -1 as a
        # no-op; keep the portable selector path equally side-effect free.
        if bool((metadata.slot_mapping[:rows] < 0).all().item()):
            output.zero_()
            return
        (
            request_ids,
            logical_positions,
            state_slots,
            state_is_fresh,
            accepted,
            query_start_loc,
            num_reqs,
        ) = self._stage_runtime_metadata(
            metadata,
            rows,
            row_capacity=self.max_tokens,
        )
        mapped_rows, compressed_slots, state_reset_mask = (
            self._validate_portable_prefill(
                metadata,
                request_ids=request_ids,
                logical_positions=logical_positions,
                state_slots=state_slots,
                state_is_fresh=state_is_fresh,
                accepted=accepted,
                is_prefilling=is_prefilling,
                query_start_loc=query_start_loc,
                num_reqs=num_reqs,
                rows=rows,
            )
        )
        request_ids = request_ids[:mapped_rows]
        logical_positions = logical_positions[:mapped_rows]
        compressed_slots = compressed_slots[:mapped_rows]
        active_positions = self._active_positions(positions, mapped_rows)
        rope_positions = canonical_qsa_rope_positions(
            active_positions,
            num_rows=mapped_rows,
            position_axes=self.position_axes,
        )

        self._reset_fresh_selector_state(
            state_slot_ids=state_slots,
            state_is_fresh=state_reset_mask,
            sequence_lengths=self._sequence_lengths,
            query_start_loc=query_start_loc,
            num_accepted_tokens=accepted,
            num_reqs=num_reqs,
        )

        from .ops.qsa import (
            qsa_compress_groups_with_ratio,
            qsa_select_paged_tokens,
            qsa_sparse_paged_attention,
            qsa_store_cache_rows,
        )

        active_raw_key = raw_index_key[:mapped_rows]
        if self.position_axes == 3:
            raw_positions = rope_positions[:, None, :]
            rope_cache = self._raw_rope_positions.unsqueeze(2)
        else:
            raw_positions = logical_positions[:, None, None].expand(-1, 1, 3)
            rope_cache = None
        pooled, first_rope_positions = qsa_compress_groups_with_ratio(
            active_raw_key.unsqueeze(1),
            raw_positions,
            self._raw_k_ring.unsqueeze(2),
            state_slots[:, None],
            request_ids,
            query_start_loc,
            logical_positions,
            compressed_slots,
            self.compress_ratio,
            rope_cache,
        )
        completed = compressed_slots >= 0
        pooled = torch.where(completed[:, None, None], pooled, 0.0)
        first_rope_positions = torch.where(completed[:, None], first_rope_positions, 0)
        compressed_keys = cast(torch.Tensor, self.indexer.k_layernorm(pooled))
        compressed_key_positions = (
            first_rope_positions.transpose(0, 1)
            if self.position_axes == 3
            else first_rope_positions[:, 0]
        )
        compressed_keys = apply_qsa_rope(
            self.rotary_emb,
            compressed_key_positions,
            compressed_keys,
        )
        qsa_store_cache_rows(
            self._compressed_cache.unsqueeze(2),
            compressed_slots,
            compressed_keys,
        )
        self._commit_portable_prefill_state(
            raw_index_key=active_raw_key,
            rope_positions=rope_positions,
            is_prefilling=is_prefilling,
            state_slots=state_slots,
            query_start_loc=query_start_loc,
            num_reqs=num_reqs,
        )

        impl = cast(Qwen3_8FlashNextQSAImpl, self.impl)
        impl.do_kv_cache_update(
            self,
            key[:mapped_rows],
            value[:mapped_rows],
            self.kv_cache,
            metadata.slot_mapping[:mapped_rows],
        )
        prepared_index_query = cast(
            torch.Tensor,
            self.indexer.q_layernorm(index_query[:mapped_rows]),
        )
        prepared_index_query = apply_qsa_rope(
            self.rotary_emb,
            active_positions,
            prepared_index_query,
        )
        selected = self._selected_positions[:mapped_rows]
        qsa_select_paged_tokens(
            prepared_index_query,
            self._compressed_cache.unsqueeze(2),
            self._main_block_table,
            request_ids,
            logical_positions,
            self._sequence_lengths,
            self.budget,
            self.compress_ratio,
            selected,
        )
        main_k_cache, main_v_cache = impl._kv_cache_views(self.kv_cache)
        output.zero_()
        qsa_sparse_paged_attention(
            query[:mapped_rows],
            main_k_cache,
            main_v_cache,
            selected,
            self._main_block_table,
            request_ids,
            output[:mapped_rows],
        )

    def _run_b12x_decode(
        self,
        *,
        metadata: Qwen3_8FlashNextQSAMetadata,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        index_query: torch.Tensor,
        raw_index_key: torch.Tensor,
        output: torch.Tensor,
        rows: int,
    ) -> None:
        if self._qsa_binding is None:
            raise RuntimeError("b12x QSA decode was not bound to its cache")
        request_ids, logical_positions = self._prepare_decode_metadata(metadata, rows)
        # The model-state diagnostic reads this stable staged view after CUDA
        # graph replay, when the custom-op body itself does not execute in Python.
        self._b12x_diagnostic_request_ids = request_ids
        rope_positions = canonical_qsa_rope_positions(
            self._active_positions(positions, rows),
            num_rows=rows,
            position_axes=self.position_axes,
        )
        _stage_qsa_rope_positions_kernel[(rows,)](
            rope_positions,
            request_ids,
            self._rope_position_input,
            rope_positions.stride(0),
            rope_positions.stride(1),
            self._rope_position_input.stride(0),
            rows,
            POSITION_AXES=self.position_axes,
            num_warps=1,
        )
        rope_positions = self._rope_position_input[:rows]
        self._query_input[:rows].copy_(query[:rows])
        self._index_query_input[:rows].copy_(index_query[:rows])
        self._raw_index_key_input[:rows].copy_(raw_index_key[:rows])

        impl = cast(Qwen3_8FlashNextQSAImpl, self.impl)
        impl.do_kv_cache_update(
            self,
            key[:rows],
            value[:rows],
            self.kv_cache,
            metadata.slot_mapping[:rows],
        )
        qsa = get_b12x_qsa()
        if qsa is None:
            raise RuntimeError("b12x QSA disappeared after cache binding")
        result = qsa.run(
            self._qsa_binding,
            query=self._query_input[:rows],
            index_query=self._index_query_input[:rows],
            raw_index_key=self._raw_index_key_input[:rows],
            request_ids=request_ids,
            query_positions=logical_positions,
            rope_positions=rope_positions,
            sequence_lengths=self._sequence_lengths,
            query_start_loc=self._query_start_loc,
            num_accepted_tokens=self._num_accepted_tokens,
        )
        output.zero_()
        output[:rows].copy_(result)

    def _run_qsa(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        index_query: torch.Tensor,
        raw_index_key: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        raw_metadata = get_forward_context().attn_metadata
        if isinstance(raw_metadata, list):
            raw_metadata = raw_metadata[0]
        if not isinstance(raw_metadata, dict):
            output.zero_()
            return
        metadata = raw_metadata.get(self.layer_name)
        if not isinstance(metadata, Qwen3_8FlashNextQSAMetadata):
            raise TypeError(
                f"{self.layer_name} expected QSA metadata, got "
                f"{type(metadata).__name__}"
            )
        if self.kv_cache.numel() == 0:
            raise RuntimeError("QSA main K/V cache is not bound")
        rows = int(metadata.num_actual_tokens)
        if rows < 0 or rows > min(
            query.shape[0],
            key.shape[0],
            value.shape[0],
            index_query.shape[0],
            raw_index_key.shape[0],
            output.shape[0],
        ):
            raise ValueError("QSA metadata exceeds the projected token rows")
        if rows == 0:
            output.zero_()
            return
        if metadata.causal is not True:
            raise NotImplementedError("QSA supports causal attention only")

        if metadata.has_prefill:
            prefill_requests = _qsa_prefill_requests(
                metadata,
                max_speculative_tokens=self.max_speculative_tokens,
            )
            self._run_portable_qsa(
                metadata=metadata,
                positions=positions,
                query=query,
                key=key,
                value=value,
                index_query=index_query,
                raw_index_key=raw_index_key,
                is_prefilling=prefill_requests,
                output=output,
                rows=rows,
            )
        else:
            self._run_b12x_decode(
                metadata=metadata,
                positions=positions,
                query=query,
                key=key,
                value=value,
                index_query=index_query,
                raw_index_key=raw_index_key,
                output=output,
                rows=rows,
            )
        if rows < output.shape[0]:
            output[rows:].zero_()

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        query, key, value, gate = self._project_qkv_gate(qkv, positions)
        index_query, raw_index_key = self.indexer.project(hidden_states)
        num_tokens = int(hidden_states.shape[0])
        query = query.view(num_tokens, self.num_heads, self.head_dim)
        key = key.view(num_tokens, self.num_kv_heads, self.head_dim)
        value = value.view(num_tokens, self.num_kv_heads, self.head_dim)
        output = torch.empty_like(query)
        layer_name = _encode_layer_name(self.layer_name)
        if current_platform.opaque_attention_op():
            torch.ops.vllm.qwen3_8_flash_next_qsa_with_output(
                positions,
                query,
                key,
                value,
                index_query,
                raw_index_key,
                output,
                layer_name,
            )
        else:
            qwen3_8_flash_next_qsa_with_output(
                positions,
                query,
                key,
                value,
                index_query,
                raw_index_key,
                output,
                layer_name,
            )
        gated = output.flatten(-2) * torch.sigmoid(gate)
        projected, _ = self.o_proj(gated)
        return projected


def qwen3_8_flash_next_qsa_with_output(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    index_query: torch.Tensor,
    raw_index_key: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    """Run the package-owned QSA write, selector, and sparse-read transaction."""

    resolved_name = _resolve_layer_name(layer_name)
    layer = get_forward_context().no_compile_layers[resolved_name]
    if not isinstance(layer, Qwen3_8FlashNextQSAAttention):
        raise TypeError(f"{resolved_name} is not a Qwen3.8-Flash-Next QSA owner")
    layer._run_qsa(
        positions,
        query,
        key,
        value,
        index_query,
        raw_index_key,
        output,
    )


def _qwen3_8_flash_next_qsa_with_output_fake(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    index_query: torch.Tensor,
    raw_index_key: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    del (
        positions,
        query,
        key,
        value,
        index_query,
        raw_index_key,
        output,
        layer_name,
    )


direct_register_custom_op(
    op_name="qwen3_8_flash_next_qsa_with_output",
    op_func=qwen3_8_flash_next_qsa_with_output,
    mutates_args=["output"],
    fake_impl=_qwen3_8_flash_next_qsa_with_output_fake,
)


__all__ = [
    "QSAIndexer",
    "Qwen3_8FlashNextQSAAttention",
    "Qwen3_8FlashNextQSABackend",
    "Qwen3_8FlashNextQSAImpl",
    "Qwen3_8FlashNextQSAMetadata",
    "Qwen3_8FlashNextQSAMetadataBuilder",
    "apply_qsa_rope",
    "qwen3_8_flash_next_qsa_with_output",
]
