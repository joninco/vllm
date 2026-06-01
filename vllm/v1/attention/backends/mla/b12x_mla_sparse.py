# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""b12x sparse-MLA backend for SM120 / SM121 (consumer Blackwell).

Counterpart to ``SparseMLASm120Backend`` (FlashInfer V32 v2). Same envelope --
``fp8_ds_mla`` KV cache (656 B/token), head_size = 576, paged block_size = 64,
V32-family models with an ``index_topk`` config (DeepSeek V3.2, GLM-5.1, Kimi
K2.5) -- but the decode/extend kernels come from b12x's unified SM120 backend
via the ``b12x.integration.mla`` front door (``sparse_mla_decode_forward`` /
``sparse_mla_extend_forward``). On SM120+ CUDA those front-door functions route
to ``b12x/attention/mla/unified_sm120`` automatically (GLM_NSA q_head_dim==576
contract). Selecting this backend also selects b12x's sparse indexer/top-k path.

Workspace philosophy (vLLM eager path): b12x's kernels duck-type their scratch
object as a bag of attributes (``tmp_output`` / ``tmp_lse`` / ``output_buffer``
+ control pointers) plus ``set_split_chunk_config``. vLLM must not construct or
own a b12x workspace/arena, so this backend builds a fresh plain views container
per call and backs it with tensors borrowed from vLLM's shared
``current_workspace_manager()``.
"""

from dataclasses import dataclass
import os
from typing import TYPE_CHECKING, ClassVar, cast

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.distributed.parallel_state import get_dcp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import get_mla_dims
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
    SparseMLAAttentionImpl,
)
from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
)
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.workspace import current_workspace_manager
from vllm.triton_utils import tl, triton

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer

logger = init_logger(__name__)
_B12X_MLA_DEBUG = os.getenv("VLLM_B12X_MLA_DEBUG", "0") == "1"
_B12X_MLA_DEBUG_LIMIT = int(os.getenv("VLLM_B12X_MLA_DEBUG_LIMIT", "32"))
_B12X_MLA_DEBUG_MAX_REQS = int(os.getenv("VLLM_B12X_MLA_DEBUG_MAX_REQS", "4"))
_B12X_MLA_DEBUG_FILE = os.getenv(
    "VLLM_B12X_MLA_DEBUG_FILE", "/tmp/vllm_b12x_mla_debug.log"
)
_B12X_MLA_DEBUG_COUNT = 0

# Split-K tile width. Mirrors SparseMLASm120's _DECODE_SPLIT_TILE: the number of
# split-K chunks is ceil(topk / tile). This bounds the chunk dim of the borrowed
# mid_out/mid_lse scratch and the workspace ``max_chunks_per_row`` cap; b12x's
# wave-balanced planner picks num_splits <= this cap.
_DECODE_SPLIT_TILE = 64
_PREFILL_HEADS_PER_BLOCK = 16


def _cdiv(x: int, y: int) -> int:
    return (int(x) + int(y) - 1) // int(y)


def _debug_int_tensor(t: torch.Tensor, limit: int = 8) -> list[int]:
    if t.numel() == 0:
        return []
    return t.detach().flatten()[:limit].to("cpu").tolist()


def _debug_can_sync_cuda() -> bool:
    if not torch.cuda.is_available():
        return True
    try:
        return not torch.cuda.is_current_stream_capturing()
    except Exception:
        return False


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using %d", name, value, default)
        return default
    if parsed <= 0:
        logger.warning("Ignoring non-positive %s=%r; using %d", name, value, default)
        return default
    return parsed


@dataclass
class _B12XMLAScratchViews:
    """Plain vLLM-owned scratch views consumed by b12x MLA kernels."""

    mode: str
    device: torch.device
    dtype: torch.dtype
    kv_dtype: torch.dtype
    num_q_heads: int
    head_dim: int
    v_head_dim: int
    topk: int
    max_total_q: int
    max_batch: int
    page_size: int
    max_chunks_per_row: int
    tmp_output: torch.Tensor | None = None
    tmp_lse: torch.Tensor | None = None
    output_buffer: torch.Tensor | None = None
    kv_chunk_size_ptr: torch.Tensor | None = None
    num_chunks_ptr: torch.Tensor | None = None
    kv_chunk_size_value: int | None = None
    num_chunks_value: int | None = None
    fixed_capacity: bool = False
    use_cuda_graph: bool = False
    final_lse: torch.Tensor | None = None
    sm_scale_tensor: torch.Tensor | None = None
    sm_scale_value: float | None = None

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.num_q_heads = int(self.num_q_heads)
        self.head_dim = int(self.head_dim)
        self.v_head_dim = int(self.v_head_dim)
        self.topk = int(self.topk)
        self.max_total_q = int(self.max_total_q)
        self.max_batch = int(self.max_batch)
        self.page_size = int(self.page_size)
        self.max_chunks_per_row = int(self.max_chunks_per_row)

    def set_split_chunk_config(self, *, kv_chunk_size: int, num_chunks: int) -> None:
        kv_chunk_size = int(kv_chunk_size)
        num_chunks = int(num_chunks)
        if kv_chunk_size <= 0:
            raise ValueError(f"kv_chunk_size must be positive, got {kv_chunk_size}")
        if num_chunks <= 0 or num_chunks > self.max_chunks_per_row:
            raise ValueError(
                "num_chunks must be in "
                f"[1, {self.max_chunks_per_row}], got {num_chunks}"
            )
        if self.kv_chunk_size_ptr is None or self.num_chunks_ptr is None:
            raise RuntimeError("split MLA control pointers are missing")

        if self.kv_chunk_size_value != kv_chunk_size:
            self.kv_chunk_size_ptr.fill_(kv_chunk_size)
            self.kv_chunk_size_value = kv_chunk_size
        if self.num_chunks_value != num_chunks:
            self.num_chunks_ptr.fill_(num_chunks)
            self.num_chunks_value = num_chunks


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


@triton.jit
def _compact_page_table_valid_prefix_kernel(
    page_table_ptr,
    nsa_len_ptr,
    page_stride0,
    page_stride1,
    width: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    valid_col = offs < width
    vals = tl.load(
        page_table_ptr + row * page_stride0 + offs * page_stride1,
        mask=valid_col,
        other=-1,
    )
    # B12X consumes page_table_1 as a dense prefix of length nsa_cache_seqlens.
    is_valid = valid_col & (vals >= 0)
    compact_pos = tl.cumsum(is_valid.to(tl.int32), 0) - 1
    valid_count = tl.sum(is_valid.to(tl.int32), axis=0)
    row_base = page_table_ptr + row * page_stride0
    tl.store(row_base + compact_pos * page_stride1, vals, mask=is_valid)
    tl.store(
        row_base + offs * page_stride1,
        -1,
        mask=valid_col & (offs >= valid_count),
    )
    tl.store(nsa_len_ptr + row, valid_count)


def _compact_page_table_valid_prefix(
    page_table: torch.Tensor,
    nsa_cache_seqlens: torch.Tensor,
) -> None:
    width = page_table.shape[1]
    if width == 0 or page_table.shape[0] == 0:
        return
    block_n = triton.next_power_of_2(width)
    _compact_page_table_valid_prefix_kernel[(page_table.shape[0],)](
        page_table,
        nsa_cache_seqlens,
        page_table.stride(0),
        page_table.stride(1),
        width,
        BLOCK_N=block_n,
    )


class B12xMLASparseBackend(AttentionBackend):
    """b12x unified sparse-MLA backend (SM120 / SM121).

    Same envelope as ``SparseMLASm120Backend`` (head 576, fp8_ds_mla, block 64,
    index_topk) but driven by b12x's unified decode/extend kernels.
    """

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8_ds_mla",
        "fp8",  # alias for fp8_ds_mla on this backend (auto-converted by MLAAttention)
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # Must equal DeepseekV32IndexerBackend.get_supported_kernel_block_sizes
        # on CUDA (= [64]); the unified b12x decode/extend kernels dispatch
        # page_block_size == 64 natively (matches the fp8_ds_mla layout).
        return [64]

    @staticmethod
    def get_name() -> str:
        return "B12X_MLA_SPARSE"

    @staticmethod
    def get_impl_cls() -> type["B12xMLASparseImpl"]:
        return B12xMLASparseImpl

    @staticmethod
    def get_builder_cls() -> type["B12xMLASparseMetadataBuilder"]:
        return B12xMLASparseMetadataBuilder

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # GLM_NSA contract: q_head_dim = kv_lora_rank (512) + qk_rope_head_dim
        # (64) = 576. The unified decode raises on any other q_head_dim.
        return [576]

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        # Consumer Blackwell SM120 / SM121. The unified b12x kernels gate on
        # get_sm_version(device) >= 120 internally.
        return capability.major == 12

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
        device_capability: DeviceCapability,
    ) -> str | None:
        # Require an indexer-equipped (index_topk) model, same as SPARSE_MLA_SM120.
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        if vllm_config.model_config is not None:
            hf_text_config = vllm_config.model_config.hf_text_config
            if not hasattr(hf_text_config, "index_topk"):
                return "B12X_MLA_SPARSE requires a model with index_topk config"
        return None

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,  # = 1 for MLA
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if cache_dtype_str == "fp8_ds_mla":
            # V32 fp8_ds_mla packed: 656 B/token (512 NoPE + 16 inline FP32
            # scales + 128 BF16 RoPE). Mirrors the FlashMLA / SPARSE_MLA_SM120
            # layout; b12x's GLM_NSA decode reads the same record.
            return (num_blocks, block_size, 656)
        return (num_blocks, block_size, head_size)


@dataclass
class B12xMLASparseMetadata(AttentionMetadata):
    """Attention metadata for the B12X_MLA_SPARSE backend."""

    num_reqs: int
    max_query_len: int
    max_seq_len: int
    num_actual_tokens: int

    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    req_id_per_token: torch.Tensor
    # Per-request computed KV length (decode cache_seqlens_int32).
    seq_lens: torch.Tensor
    # Per-token causal KV length; clamped to topk to form nsa_cache_seqlens.
    # For pure decode this equals ``seq_lens`` (one token per request).
    cache_seq_lens_per_token: torch.Tensor

    block_size: int = 64
    topk_tokens: int = 2048


class B12xMLASparseMetadataBuilder(AttentionMetadataBuilder[B12xMLASparseMetadata]):
    """Builder for B12X_MLA_SPARSE attention metadata."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        self.vllm_config = vllm_config
        self.layer_names = layer_names
        self.kv_cache_spec = kv_cache_spec
        self.model_config = vllm_config.model_config
        self.device = device

        self.mla_dims = get_mla_dims(self.model_config)
        self.topk_tokens = vllm_config.model_config.hf_config.index_topk
        self.dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        self.dcp_rank = (
            get_dcp_group().rank_in_group if self.dcp_world_size > 1 else 0
        )
        self.cp_kv_cache_interleave_size = (
            vllm_config.parallel_config.cp_kv_cache_interleave_size
        )

        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        max_seqs = vllm_config.scheduler_config.max_num_seqs
        # Max-batched-token scratch buffers so cudagraph capture sees stable
        # allocations (sliced per build()).
        self.req_id_per_token_buffer = torch.empty(
            (max_tokens,), dtype=torch.int32, device=device
        )
        self.cache_seq_lens_per_token_buffer = torch.empty(
            (max_tokens,), dtype=torch.int32, device=device
        )
        self.cache_seq_lens_per_req_buffer = torch.empty(
            (max_seqs,), dtype=torch.int32, device=device
        )
        self.req_ids_arange = torch.arange(max_tokens, dtype=torch.int32, device=device)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> B12xMLASparseMetadata:
        del common_prefix_len, fast_build
        cm = common_attn_metadata
        num_tokens = cm.num_actual_tokens

        seq_lens_for_req = (
            cm.dcp_local_seq_lens
            if cm.dcp_local_seq_lens is not None
            else cm.seq_lens
        )

        # Pure decode is the hot path. Keep it device-only and reuse stable
        # buffers so CUDA graph capture sees fixed tensor addresses.
        if cm.max_query_len <= 1 and num_tokens == cm.num_reqs:
            self.req_id_per_token_buffer[:num_tokens].copy_(
                self.req_ids_arange[:num_tokens]
            )
            self.cache_seq_lens_per_token_buffer[:num_tokens].copy_(
                seq_lens_for_req[:num_tokens], non_blocking=True
            )
            self.cache_seq_lens_per_req_buffer[: cm.num_reqs].copy_(
                seq_lens_for_req[: cm.num_reqs], non_blocking=True
            )
        else:
            starts = np.asarray(cm.query_start_loc_cpu, dtype=np.int32)
            query_lens = np.diff(starts)
            num_query_tokens = int(starts[-1])
            if num_query_tokens > num_tokens:
                raise RuntimeError(
                    "B12X sparse MLA metadata received query_start_loc with "
                    f"{num_query_tokens} tokens, exceeding padded capacity "
                    f"{num_tokens}"
                )

            req_ids = np.zeros((num_tokens,), dtype=np.int32)
            if num_query_tokens:
                req_ids[:num_query_tokens] = np.repeat(
                    np.arange(cm.num_reqs, dtype=np.int32), query_lens
                )

            # Prefill/extend rows need exact lengths. The upper-bound CPU
            # tensor can include optimistic/padded lengths, which corrupts DCP
            # long-context causal limits and sparse-indexer physical slots.
            seq_lens_cpu = cm.seq_lens_cpu
            seq_lens_cpu_np = seq_lens_cpu.numpy().astype(np.int32, copy=False)

            # The sparse indexer emits request-local token indices. For DCP the
            # B12X kernels need the local causal cache length for each real query
            # row; padded CUDA-graph rows stay zero-length.
            per_token_lens = np.zeros((num_tokens,), dtype=np.int32)
            for req_id, q_len in enumerate(query_lens):
                if q_len <= 0:
                    continue
                start = int(starts[req_id])
                end = int(starts[req_id + 1])
                context_len = int(seq_lens_cpu_np[req_id]) - int(q_len)
                global_per_token_lens = torch.arange(
                    context_len + 1,
                    context_len + int(q_len) + 1,
                    dtype=torch.int32,
                )
                if cm.dcp_local_seq_lens is not None:
                    per_token_lens[start:end] = get_dcp_local_seq_lens(
                        global_per_token_lens,
                        self.dcp_world_size,
                        self.dcp_rank,
                        self.cp_kv_cache_interleave_size,
                    ).numpy()
                else:
                    per_token_lens[start:end] = global_per_token_lens.numpy()

            req_ids_t = torch.from_numpy(req_ids)
            per_token_lens_t = torch.from_numpy(per_token_lens)
            if req_ids_t.device.type == "cpu":
                req_ids_t = req_ids_t.pin_memory()
            if per_token_lens_t.device.type == "cpu":
                per_token_lens_t = per_token_lens_t.pin_memory()
            self.req_id_per_token_buffer[:num_tokens].copy_(
                req_ids_t, non_blocking=True
            )
            self.cache_seq_lens_per_token_buffer[:num_tokens].copy_(
                per_token_lens_t, non_blocking=True
            )
            self.cache_seq_lens_per_req_buffer[: cm.num_reqs].copy_(
                seq_lens_for_req[: cm.num_reqs], non_blocking=True
            )

        return B12xMLASparseMetadata(
            num_reqs=cm.num_reqs,
            max_query_len=cm.max_query_len,
            max_seq_len=cm.max_seq_len,
            num_actual_tokens=num_tokens,
            query_start_loc=cm.query_start_loc,
            slot_mapping=cm.slot_mapping,
            block_table=cm.block_table_tensor,
            req_id_per_token=self.req_id_per_token_buffer[:num_tokens],
            seq_lens=self.cache_seq_lens_per_req_buffer[: cm.num_reqs],
            cache_seq_lens_per_token=self.cache_seq_lens_per_token_buffer[:num_tokens],
            block_size=self.kv_cache_spec.block_size,
            topk_tokens=self.topk_tokens,
        )


class B12xMLASparseImpl(SparseMLAAttentionImpl[B12xMLASparseMetadata]):
    """b12x unified sparse-MLA implementation (decode + extend/prefill)."""

    can_return_lse_for_decode: bool = True
    supports_dcp_with_fp8_kvcache: bool = True

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
        topk_indice_buffer: torch.Tensor | None = None,
        indexer: "Indexer | None" = None,
        **mla_args,
    ) -> None:
        if any([alibi_slopes, sliding_window, logits_soft_cap]):
            raise NotImplementedError(
                "B12X_MLA_SPARSE does not support alibi_slopes / sliding_window "
                "/ logits_soft_cap"
            )
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "B12X_MLA_SPARSE only supports decoder self-attention"
            )

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype

        # MLA dims (absorbed: Q post-projection is [T, H, kv_lora_rank + rope]).
        self.kv_lora_rank: int = mla_args["kv_lora_rank"]
        self.qk_nope_head_dim: int = mla_args["qk_nope_head_dim"]
        self.qk_rope_head_dim: int = mla_args["qk_rope_head_dim"]
        self.v_head_dim: int = mla_args.get("v_head_dim", 512)
        # GLM_NSA contract: q_head_dim = kv_lora_rank (512) + qk_rope (64) = 576.
        self.q_head_dim = self.kv_lora_rank + self.qk_rope_head_dim
        self.force_contiguous_mla_bmm_input = True
        self.force_contiguous_mla_bmm_weight = True
        self.force_contiguous_mla_bmm_output = True

        assert indexer is not None, (
            "B12X_MLA_SPARSE requires a sparse-MLA indexer (model with "
            "index_topk in its config)."
        )
        self.topk_indices_buffer: torch.Tensor | None = indexer.topk_indices_buffer
        assert self.topk_indices_buffer is not None
        self.topk_tokens = int(self.topk_indices_buffer.shape[-1])

        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        scheduler_config = vllm_config.scheduler_config
        self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        max_batched = int(scheduler_config.max_num_batched_tokens)
        max_num_seqs = int(scheduler_config.max_num_seqs)
        self.block_size = 64

        # Split-K cap: ceil(topk / tile). Bounds the borrowed mid_out/mid_lse
        # chunk dim and the workspace max_chunks_per_row.
        self._num_splits_cap = max(1, _cdiv(self.topk_tokens, _DECODE_SPLIT_TILE))
        self._prefill_num_heads = (
            _cdiv(self.num_heads, _PREFILL_HEADS_PER_BLOCK)
            * _PREFILL_HEADS_PER_BLOCK
        )
        if self._prefill_num_heads != self.num_heads:
            logger.info_once(
                "Padding B12X_MLA_SPARSE prefill heads from %d to %d.",
                self.num_heads,
                self._prefill_num_heads,
            )

        # Decode query rows per request. Short speculative verify batches can
        # use the decode kernel instead of the heavier extend kernel.
        spec = getattr(vllm_config, "speculative_config", None)
        num_speculative_tokens = 0
        if spec is not None and getattr(spec, "num_speculative_tokens", None):
            num_speculative_tokens = int(spec.num_speculative_tokens)
        self._spec_extend_as_decode = (
            os.getenv("VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE", "1") != "0"
        )
        self._spec_decode_max_q = _env_int(
            "VLLM_B12X_MLA_SPEC_DECODE_MAX_Q",
            max(1 + num_speculative_tokens, 8),
        )
        q_per_req = max(
            1,
            1 + num_speculative_tokens,
            self._spec_decode_max_q if self._spec_extend_as_decode else 1,
        )
        self._decode_max_rows = min(max_num_seqs * q_per_req, max_batched)
        self._decode_num_heads_cap = max(
            int(self.num_heads), int(self.num_heads) * max(1, int(self.dcp_world_size))
        )

        self._max_num_seqs = int(max_num_seqs)
        self._max_batched = int(max_batched)

        # Lazily import b12x only on this opt-in path.
        from b12x.integration.mla import (
            sparse_mla_decode_forward,
            sparse_mla_extend_forward,
        )

        self._sparse_mla_decode_forward = sparse_mla_decode_forward
        self._sparse_mla_extend_forward = sparse_mla_extend_forward

        # Pre-touch the shared decode scratch at the max decode batch so the
        # workspace manager grows during warmup, before lock_workspace() runs
        # post-cudagraph-capture. Mirrors SparseMLASm120Impl.__init__.
        self._borrow_decode_scratch(
            self._decode_max_rows, num_q_heads=self._decode_num_heads_cap
        )

        # Q arrives BF16; the unified kernel quantizes inside.
        self.supports_quant_query_input = False

    def _borrow_decode_scratch(
        self, num_tokens: int, *, num_q_heads: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Borrow split-K scratch/control tensors from vLLM's workspace manager."""
        num_q_heads = int(num_q_heads or self.num_heads)
        return tuple(  # type: ignore[return-value]
            current_workspace_manager().get_simultaneous(
                (
                    (
                        num_tokens,
                        num_q_heads,
                        self._num_splits_cap,
                        self.kv_lora_rank,
                    ),
                    torch.bfloat16,
                ),
                ((num_tokens, num_q_heads, self._num_splits_cap), torch.float32),
                ((num_tokens, num_q_heads), torch.float32),
                ((1,), torch.int32),
                ((1,), torch.int32),
            )
        )

    def _make_scratch_views(
        self,
        *,
        mode: str,
        max_total_q: int,
        num_q_heads: int,
        output_buffer: torch.Tensor,
        tmp_output: torch.Tensor | None = None,
        tmp_lse: torch.Tensor | None = None,
        final_lse: torch.Tensor | None = None,
        kv_chunk_size_ptr: torch.Tensor | None = None,
        num_chunks_ptr: torch.Tensor | None = None,
    ) -> _B12XMLAScratchViews:
        return _B12XMLAScratchViews(
            mode=mode,
            device=self.device,
            dtype=torch.bfloat16,
            kv_dtype=torch.uint8,
            num_q_heads=num_q_heads,
            head_dim=self.q_head_dim,
            v_head_dim=self.kv_lora_rank,
            topk=self.topk_tokens,
            max_total_q=max_total_q,
            max_batch=max_total_q if mode == "decode" else self._max_num_seqs,
            page_size=self.block_size,
            max_chunks_per_row=self._num_splits_cap,
            tmp_output=tmp_output,
            tmp_lse=tmp_lse,
            output_buffer=output_buffer,
            final_lse=final_lse,
            kv_chunk_size_ptr=kv_chunk_size_ptr,
            num_chunks_ptr=num_chunks_ptr,
        )

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: B12xMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # q arrives as (mqa_ql_nope[T, H, kv_lora_rank], mqa_q_pe[T, H, rope]);
        # b12x's GLM_NSA contract wants a single contiguous [T, H, 576] tensor.
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)
        q = q.contiguous()

        num_actual_toks = q.shape[0]
        lse = None

        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]
        per_token_cache = attn_metadata.cache_seq_lens_per_token[:num_actual_toks]
        causal_lens = per_token_cache
        if causal_lens.dtype != torch.int32:
            causal_lens = causal_lens.to(torch.int32)
        causal_lens = causal_lens.contiguous()

        # Per-request topk indices -> physical cache slot ids. B12X consumes the
        # table as a dense prefix of length nsa_cache_seqlens, so preserve the
        # converter's valid count and compact any holes before launching B12X.
        # Causal filtering must happen before compact/count: DCP prefill top-k
        # rows can contain later local tokens that would otherwise survive a
        # count-only clamp after compaction.
        page_table_1, nsa_cache_seqlens = cast(
            tuple[torch.Tensor, torch.Tensor],
            triton_convert_req_index_to_global_index(
                attn_metadata.req_id_per_token[:num_actual_toks],
                attn_metadata.block_table,
                topk_indices,
                BLOCK_SIZE=attn_metadata.block_size,
                NUM_TOPK_TOKENS=topk_indices.shape[1],
                return_valid_counts=True,
                causal_lens=causal_lens,
            ),
        )
        page_table_1 = page_table_1.to(torch.int32).contiguous()
        nsa_cache_seqlens = nsa_cache_seqlens.to(torch.int32).contiguous()
        _compact_page_table_valid_prefix(page_table_1, nsa_cache_seqlens)
        topk_width = page_table_1.shape[1]

        # nsa_cache_seqlens: per-token count of selected KV rows to attend,
        # additionally clamped to the causal KV length for prefill/verify rows.
        causal_nsa = torch.clamp(causal_lens, max=topk_width)
        torch.minimum(nsa_cache_seqlens, causal_nsa, out=nsa_cache_seqlens)
        _mask_page_table_after_nsa_len(page_table_1, nsa_cache_seqlens)
        # Per-request KV length (validated but unused by the unified kernels).
        cache_seqlens = attn_metadata.seq_lens.to(torch.int32).contiguous()

        if (
            _B12X_MLA_DEBUG
            and self.dcp_world_size > 1
            and _debug_can_sync_cuda()
        ):
            global _B12X_MLA_DEBUG_COUNT
            debug_count = _B12X_MLA_DEBUG_COUNT
            should_debug = (
                int(cache_seqlens.max().item()) >= 1024
                or int(attn_metadata.max_query_len) >= 1024
            )
            if should_debug and debug_count < _B12X_MLA_DEBUG_LIMIT:
                try:
                    rows = min(_B12X_MLA_DEBUG_MAX_REQS, num_actual_toks)
                    cols = min(16, topk_indices.shape[1])
                    valid_counts = (page_table_1 >= 0).sum(dim=1)
                    prefix_invalid = page_table_1 < 0
                    prefix_counts = torch.argmax(prefix_invalid.to(torch.int32), dim=1)
                    prefix_counts = torch.where(
                        prefix_invalid.any(dim=1),
                        prefix_counts,
                        torch.full_like(prefix_counts, page_table_1.shape[1]),
                    )
                    kv_rows = int(kv_c_and_k_pe_cache.shape[0]) * int(
                        kv_c_and_k_pe_cache.shape[1]
                    )
                    payload = (
                        f"kind=mla layer={getattr(layer, 'layer_name', '')} "
                        f"count={debug_count} dcp={self.dcp_world_size} "
                        f"max_q={attn_metadata.max_query_len} "
                        f"max_seq={attn_metadata.max_seq_len} "
                        f"num_reqs={attn_metadata.num_reqs} "
                        f"num_toks={num_actual_toks} heads={int(q.shape[1])} "
                        f"kv_rows={kv_rows} "
                        f"topk_min={int(topk_indices.min().item())} "
                        f"topk_max={int(topk_indices.max().item())} "
                        f"causal_min={int(causal_lens.min().item())} "
                        f"causal_max={int(causal_lens.max().item())} "
                        f"page_min={int(page_table_1.min().item())} "
                        f"page_max={int(page_table_1.max().item())} "
                        f"nsa_min={int(nsa_cache_seqlens.min().item())} "
                        f"nsa_max={int(nsa_cache_seqlens.max().item())} "
                        f"valid_min={int(valid_counts.min().item())} "
                        f"valid_max={int(valid_counts.max().item())} "
                        f"prefix_min={int(prefix_counts.min().item())} "
                        f"prefix_max={int(prefix_counts.max().item())} "
                        f"bad_valid={int((valid_counts < nsa_cache_seqlens).sum().item())} "
                        f"bad_prefix={int((prefix_counts < nsa_cache_seqlens).sum().item())} "
                        f"oob={int((page_table_1 >= kv_rows).sum().item())} "
                        f"neg={int((page_table_1 < 0).sum().item())} "
                        f"cache={_debug_int_tensor(cache_seqlens[:rows])} "
                        f"causal={_debug_int_tensor(causal_lens[:rows])} "
                        f"nsa={_debug_int_tensor(nsa_cache_seqlens[:rows])} "
                        f"valid={_debug_int_tensor(valid_counts[:rows])} "
                        f"prefix={_debug_int_tensor(prefix_counts[:rows])} "
                        f"topk0={_debug_int_tensor(topk_indices[:1, :cols])} "
                        f"page0={_debug_int_tensor(page_table_1[:1, :cols])}"
                    )
                    logger.warning("B12X_MLA_DEBUG %s", payload)
                    with open(_B12X_MLA_DEBUG_FILE, "a", encoding="utf-8") as f:
                        f.write(payload + "\n")
                except Exception:
                    logger.exception("Failed to write B12X MLA debug payload")
                _B12X_MLA_DEBUG_COUNT = debug_count + 1

        # KV cache -> flat (num_slots, 1, nbytes) uint8 (b12x requires rank-3
        # uint8; page_size tells it the per-block stride). page_table_1 are
        # physical slot ids consistent with block_size == page_size.
        kv_u8 = kv_c_and_k_pe_cache.view(torch.uint8)
        kv_cache = kv_u8.reshape(-1, 1, kv_u8.shape[-1])
        if not kv_cache.is_contiguous():
            kv_cache = kv_cache.contiguous()

        is_decode = attn_metadata.max_query_len <= 1
        use_decode_kernel = is_decode or (
            self._spec_extend_as_decode
            and attn_metadata.max_query_len <= self._spec_decode_max_q
            and num_actual_toks
            <= attn_metadata.num_reqs * self._spec_decode_max_q
        )
        if use_decode_kernel:
            num_q_heads = int(q.shape[1])
            output = q.new_empty(
                (num_actual_toks, num_q_heads, self.kv_lora_rank), dtype=q.dtype
            )
            mid_out, mid_lse, final_lse, kv_chunk_size_ptr, num_chunks_ptr = (
                self._borrow_decode_scratch(num_actual_toks, num_q_heads=num_q_heads)
            )
            ws = self._make_scratch_views(
                mode="decode",
                max_total_q=self._decode_max_rows,
                num_q_heads=num_q_heads,
                output_buffer=output,
                tmp_output=mid_out,
                tmp_lse=mid_lse,
                final_lse=final_lse,
                kv_chunk_size_ptr=kv_chunk_size_ptr,
                num_chunks_ptr=num_chunks_ptr,
            )
            cache_seqlens = (
                attn_metadata.seq_lens
                if is_decode
                else attn_metadata.cache_seq_lens_per_token[:num_actual_toks]
            )
            cache_seqlens = cache_seqlens.to(torch.int32).contiguous()
            if self.need_to_return_lse_for_decode:
                out, lse = self._sparse_mla_decode_forward(
                    q_all=q,
                    kv_cache=kv_cache,
                    page_table_1=page_table_1,
                    cache_seqlens_int32=cache_seqlens,
                    nsa_cache_seqlens_int32=nsa_cache_seqlens,
                    workspace=ws,
                    sm_scale=self.scale,
                    v_head_dim=self.kv_lora_rank,
                    return_lse=True,
                    lse_scale="natural",
                )
                return out, lse
            out = self._sparse_mla_decode_forward(
                q_all=q,
                kv_cache=kv_cache,
                page_table_1=page_table_1,
                cache_seqlens_int32=cache_seqlens,
                nsa_cache_seqlens_int32=nsa_cache_seqlens,
                workspace=ws,
                sm_scale=self.scale,
                v_head_dim=self.kv_lora_rank,
            )
        else:
            # Extend / prefill -> single-pass unified prefill (no split-K
            # scratch needed; only output_buffer is read). The b12x prefill
            # kernel currently requires head blocks of 16, so high-TP shards
            # with fewer local heads are padded and sliced back after launch.
            num_q_heads = int(q.shape[1])
            prefill_num_heads = (
                _cdiv(num_q_heads, _PREFILL_HEADS_PER_BLOCK)
                * _PREFILL_HEADS_PER_BLOCK
            )
            if prefill_num_heads == num_q_heads:
                prefill_q = q
            else:
                prefill_q = q.new_zeros(
                    (num_actual_toks, prefill_num_heads, self.q_head_dim)
                )
                prefill_q[:, :num_q_heads, :].copy_(q)

            output = q.new_empty(
                (num_actual_toks, prefill_num_heads, self.kv_lora_rank),
                dtype=q.dtype,
            )
            ws = self._make_scratch_views(
                mode="extend",
                max_total_q=self._max_batched,
                num_q_heads=prefill_num_heads,
                output_buffer=output,
            )
            if self.need_to_return_lse_for_decode:
                out, lse = self._sparse_mla_extend_forward(
                    q_all=prefill_q,
                    kv_cache=kv_cache,
                    selected_token_offsets=page_table_1,
                    cache_seqlens_int32=cache_seqlens,
                    nsa_cache_seqlens_int32=nsa_cache_seqlens,
                    workspace=ws,
                    sm_scale=self.scale,
                    v_head_dim=self.kv_lora_rank,
                    return_lse=True,
                    lse_scale="natural",
                )
            else:
                out = self._sparse_mla_extend_forward(
                    q_all=prefill_q,
                    kv_cache=kv_cache,
                    selected_token_offsets=page_table_1,
                    cache_seqlens_int32=cache_seqlens,
                    nsa_cache_seqlens_int32=nsa_cache_seqlens,
                    workspace=ws,
                    sm_scale=self.scale,
                    v_head_dim=self.kv_lora_rank,
                )
                lse = None
            if prefill_num_heads != num_q_heads:
                dense_out = q.new_empty(
                    (num_actual_toks, num_q_heads, self.kv_lora_rank),
                    dtype=q.dtype,
                )
                dense_out.copy_(out[:, :num_q_heads, :])
                out = dense_out
                if lse is not None:
                    lse = lse[:, :num_q_heads].contiguous()
        return out, lse
