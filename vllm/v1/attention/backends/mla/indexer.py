# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
import os

import torch

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_dcp_group
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.utils.platform_utils import num_compute_units
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.mla.compressor_utils import get_compressed_slot_mapping
from vllm.v1.attention.backends.utils import (
    get_dcp_local_seq_lens,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec, MLAAttentionSpec
from vllm.v1.worker.cp_utils import get_total_cp_world_size

logger = init_logger(__name__)
_B12X_MLA_DEBUG = os.getenv("VLLM_B12X_MLA_DEBUG", "0") == "1"
_B12X_MLA_DEBUG_LIMIT = int(os.getenv("VLLM_B12X_MLA_DEBUG_LIMIT", "32"))
_B12X_MLA_DEBUG_FILE = os.getenv(
    "VLLM_B12X_MLA_DEBUG_FILE", "/tmp/vllm_b12x_mla_debug.log"
)


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


@triton.jit
def _prepare_uniform_decode_kernel(
    seq_lens_ptr,
    decode_seq_lens_ptr,
    block_table_ptr,
    block_table_stride,
    expanded_block_table_ptr,
    expanded_bt_stride,
    decode_lens_ptr,
    max_decode_len,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.program_id(0)
    req_id = idx // max_decode_len
    local_idx = idx % max_decode_len

    # Compute number of KVs attended to by this token.
    seq_len = tl.load(seq_lens_ptr + req_id)
    per_token_seq_len = seq_len - max_decode_len + local_idx + 1
    tl.store(decode_seq_lens_ptr + idx, per_token_seq_len)

    # Copy block table row.
    src = block_table_ptr + req_id * block_table_stride
    dst = expanded_block_table_ptr + idx * expanded_bt_stride
    for i in tl.range(0, expanded_bt_stride, BLOCK_SIZE):
        off = i + tl.arange(0, BLOCK_SIZE)
        mask = off < expanded_bt_stride
        src_block = tl.load(src + off, mask=mask)
        tl.store(dst + off, src_block, mask=mask)

    # All reqs now have decode_len = 1.
    tl.store(decode_lens_ptr + idx, 1)


def split_indexer_prefill_chunks(
    seq_lens_cpu: torch.Tensor,
    query_lens_cpu: torch.Tensor,
    workspace_size: int,
    max_logits_bytes: int,
    request_offset: int = 0,
) -> list[tuple[slice, slice]]:
    """
    Split prefill requests into chunks for the sparse indexer, respecting:
    - N constraint: total_seq_lens <= workspace_size (existing O(N) workspace)
    - Logits constraint: M * N * 4 <= max_logits_bytes

    When a single request-level chunk still exceeds the logits budget,
    sub-chunks on the query dimension (M) to bound peak memory.

    Returns list of (req_slice, query_slice) tuples.
    """
    chunks: list[tuple[slice, slice]] = []
    n = len(seq_lens_cpu)
    max_logits_elems = max_logits_bytes // 4
    end = 0

    while end < n:
        start, chunk_m, chunk_n = end, 0, 0

        while end < n:
            q, s = query_lens_cpu[end].item(), seq_lens_cpu[end].item()
            new_m, new_n = chunk_m + q, chunk_n + s
            if new_n <= workspace_size and new_m * new_n <= max_logits_elems:
                chunk_m, chunk_n = new_m, new_n
                end += 1
            else:
                break

        # A single request can exceed the budget, requiring sub-chunking
        # on the query dimension.
        if end == start:
            chunk_m, chunk_n = query_lens_cpu[end].item(), seq_lens_cpu[end].item()
            end += 1

        req_slice = slice(start + request_offset, end + request_offset)
        max_q = max(1, max_logits_elems // chunk_n) if chunk_n > 0 else chunk_m
        for q_off in range(0, chunk_m, max_q):
            sub_m = min(max_q, chunk_m - q_off)
            chunks.append((req_slice, slice(q_off, q_off + sub_m)))

    return chunks


def _kv_spans_from_batches(
    start_seq_loc: torch.Tensor,
    seq_len_per_batch: torch.Tensor,
    device: torch.device,
    query_slice: slice | None = None,
    *,
    dcp_world_size: int = 1,
    dcp_rank: int = 0,
    cp_kv_cache_interleave_size: int = 1,
    global_seq_len_per_batch: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    q = start_seq_loc.to(dtype=torch.long)
    local_lens = seq_len_per_batch.to(dtype=torch.long)
    global_lens = (
        local_lens
        if global_seq_len_per_batch is None
        else global_seq_len_per_batch.to(dtype=torch.long)
    )
    assert q.dim() == 1 and local_lens.dim() == 1
    assert q.numel() == local_lens.numel() + 1
    assert global_lens.dim() == 1 and global_lens.numel() == local_lens.numel()

    counts = q[1:] - q[:-1]
    total_queries = int(q[-1].item())
    num_reqs = local_lens.numel()
    if total_queries == 0:
        return (
            torch.empty(0, dtype=torch.int32, device=device),
            torch.empty(0, dtype=torch.int32, device=device),
        )

    if query_slice is None:
        slice_start, slice_stop = 0, total_queries
    else:
        slice_start = 0 if query_slice.start is None else query_slice.start
        slice_stop = total_queries if query_slice.stop is None else query_slice.stop
        if slice_start < 0 or slice_stop < slice_start or slice_stop > total_queries:
            raise ValueError(
                f"Invalid query_slice={query_slice} for {total_queries} tokens"
            )

    slice_len = slice_stop - slice_start
    if slice_len == 0:
        return (
            torch.empty(0, dtype=torch.int32, device=device),
            torch.empty(0, dtype=torch.int32, device=device),
        )

    if num_reqs == 1 and dcp_world_size <= 1:
        base = int((local_lens[0] - counts[0]).item()) + slice_start + 1
        return (
            torch.zeros(slice_len, dtype=torch.int32, device=device),
            torch.arange(base, base + slice_len, dtype=torch.int32, device=device),
        )

    kv_starts_per_batch = torch.cumsum(local_lens, dim=0) - local_lens

    if slice_start != 0 or slice_stop != total_queries:
        first_batch = int(
            torch.searchsorted(
                q, torch.tensor(slice_start, dtype=q.dtype), right=True
            ).item()
            - 1
        )
        last_batch = int(
            torch.searchsorted(
                q, torch.tensor(slice_stop - 1, dtype=q.dtype), right=True
            ).item()
            - 1
        )
        first_batch = max(0, min(first_batch, num_reqs - 1))
        last_batch = max(first_batch, min(last_batch, num_reqs - 1))

        batch_ids_compact = torch.arange(first_batch, last_batch + 1)
        overlap_start = torch.maximum(
            q[batch_ids_compact], torch.tensor(slice_start, dtype=q.dtype)
        )
        overlap_stop = torch.minimum(
            q[batch_ids_compact + 1], torch.tensor(slice_stop, dtype=q.dtype)
        )
        overlap_counts = overlap_stop - overlap_start
        valid = overlap_counts > 0
        batch_ids_compact = batch_ids_compact[valid]
        overlap_start = overlap_start[valid]
        overlap_counts = overlap_counts[valid]

        batch_id = torch.repeat_interleave(
            batch_ids_compact, overlap_counts, output_size=slice_len
        )
        segment_offsets = torch.cumsum(overlap_counts, dim=0) - overlap_counts
        token_pos_in_slice = torch.arange(slice_len, dtype=torch.long)
        global_token_pos = torch.repeat_interleave(
            overlap_start, overlap_counts, output_size=slice_len
        ) + (
            token_pos_in_slice
            - torch.repeat_interleave(
                segment_offsets, overlap_counts, output_size=slice_len
            )
        )
    else:
        batch_id = torch.repeat_interleave(
            torch.arange(num_reqs), counts, output_size=total_queries
        )
        global_token_pos = torch.arange(total_queries, dtype=torch.long)

    pos_within_req = global_token_pos - q[batch_id] + 1
    if dcp_world_size > 1:
        global_end = global_lens[batch_id] - counts[batch_id] + pos_within_req
        local_pos = get_dcp_local_seq_lens(
            global_end.to(torch.int32),
            dcp_world_size,
            dcp_rank,
            cp_kv_cache_interleave_size,
        ).to(torch.long)
    else:
        local_pos = local_lens[batch_id] - counts[batch_id] + pos_within_req

    start_tensor = kv_starts_per_batch[batch_id]
    end_location = start_tensor + local_pos
    return start_tensor.int().to(device), end_location.int().to(device)


class DeepseekV32IndexerBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "DEEPSEEK_V32_INDEXER"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [1, 64] if current_platform.is_rocm() else [64]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [32, 64, 128]

    @staticmethod
    def get_builder_cls() -> type["DeepseekV32IndexerMetadataBuilder"]:
        return DeepseekV32IndexerMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        assert num_kv_heads == 1
        return (num_blocks, block_size, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            # DeepseekV32Indexer kernels do not support cross-layer
            # KV cache layout. Identity permutation keeps num_layers
            # first, signaling incompatibility.
            return (0, 1, 2, 3)
        return (0, 1, 2)


class B12xNonCompressedIndexerBackend(DeepseekV32IndexerBackend):
    @staticmethod
    def get_name() -> str:
        return "B12X_NON_COMPRESSED_INDEXER"


class DeepseekV4IndexerBackend(DeepseekV32IndexerBackend):
    @staticmethod
    def get_name() -> str:
        return "DEEPSEEK_V4_INDEXER"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [256]


@dataclass
class DeepseekV32IndexerPrefillChunkMetadata:
    block_table: torch.Tensor
    cu_seqlen_ks: torch.Tensor
    cu_seqlen_ke: torch.Tensor
    cu_seq_lens: torch.Tensor
    token_to_seq: torch.Tensor
    total_seq_lens: int
    token_start: int
    token_end: int
    num_reqs: int
    skip_kv_gather: bool = False


@dataclass
class DeepseekV32IndexerPrefillMetadata:
    chunks: list[DeepseekV32IndexerPrefillChunkMetadata]


@dataclass
class DeepSeekV32IndexerDecodeMetadata:
    block_table: torch.Tensor
    # seq_lens: per-token effective context lengths.
    #   - flatten path / plain decode: 1D (batch_size,)
    #   - native MTP path: 2D (B, next_n) where [b,j] = L_b - next_n + j + 1
    # Both fp8_fp4_paged_mqa_logits and the topk kernels accept both shapes.
    seq_lens: torch.Tensor
    decode_lens: torch.Tensor
    requires_padding: bool
    schedule_metadata: torch.Tensor | None
    compress_ratio: int = 1


@dataclass
class DeepseekV32IndexerMetadata:
    # FIXME (zyongye)
    # hacky way to access the data now, need to be in chunked meta
    seq_lens: torch.Tensor
    max_seq_len: int
    slot_mapping: torch.Tensor

    # New for MLA (compared to FlashAttention)
    # For handling prefill decode split
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int

    decode: DeepSeekV32IndexerDecodeMetadata | None = None
    prefill: DeepseekV32IndexerPrefillMetadata | None = None


def get_max_prefill_buffer_size(vllm_config: VllmConfig):
    max_model_len = vllm_config.model_config.max_model_len
    # NOTE(Chen): 40 is a magic number for controlling the prefill buffer size.
    # Each entry is 128 fp8 bytes and 4 scale bytes for a total of 132 bytes.
    # The flashmla_sparse backend uses a workspace size of 5 * max_model_len.
    # The memory usage of the workspace there is 576 * 2 bytes; so we size this as
    # (576 * 2 // 132) * 5 = 40 to maximize this workspace size while still fitting
    # within the flashmla_sparse workspace.
    # For DeepSeek-V3.2, the max_model_len is 163840.
    #   40 * 163840 * 132 = 865075200 bytes = 825 MB
    return max_model_len * 40


class DeepseekV32IndexerMetadataBuilder(AttentionMetadataBuilder):
    reorder_batch_threshold: int = 1
    natively_supported_next_n_fp4: list[int] = [1, 2]
    # TODO (matt): integrate kernel with next_n = 4 support

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        return AttentionCGSupport.UNIFORM_BATCH

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        scheduler_config = self.vllm_config.scheduler_config
        # NOTE(Chen):an estimated max size of flattened_kv. Need to double check.
        self.max_prefill_buffer_size = get_max_prefill_buffer_size(self.vllm_config)
        self.num_speculative_tokens = (
            self.vllm_config.speculative_config.num_speculative_tokens
            if self.vllm_config.speculative_config
            else 0
        )
        self.use_fp4_indexer_cache = (
            self.vllm_config.attention_config.use_fp4_indexer_cache
        )

        assert (
            current_platform.is_device_capability_family(100)
            or not self.use_fp4_indexer_cache
        ), (
            "use_fp4_indexer_cache requires Blackwell datacenter GPUs "
            "(sm_10x, e.g. B200/GB200); sm_120 (consumer Blackwell) and "
            "earlier architectures are not supported."
        )

        next_n = self.num_speculative_tokens + 1
        self.reorder_batch_threshold += self.num_speculative_tokens
        # NOTE(zyongye) fp4 indexer cache only natively supports next_n in
        # natively_supported_next_n_fp4; for other next_n values we fall back
        # to the flattening path. Outside the SM100 datacenter family the FP8
        # paged MQA logits kernel has the same [1, 2] constraint (deepgemm
        # smxx_fp8_fp4_paged_mqa_logits.hpp:233), so flatten there too.
        self.use_flattening = (
            self.use_fp4_indexer_cache
            or not current_platform.is_device_capability_family(100)
        ) and next_n not in self.natively_supported_next_n_fp4
        self.dcp_world_size = (
            self.vllm_config.parallel_config.decode_context_parallel_size
        )
        self.dcp_rank = (
            get_dcp_group().rank_in_group if self.dcp_world_size > 1 else 0
        )
        self.cp_kv_cache_interleave_size = (
            self.vllm_config.parallel_config.cp_kv_cache_interleave_size
        )

        sm_count = num_compute_units(self.device.index)
        self.num_sms = sm_count

        self.offsets_buffer = torch.arange(
            next_n, device=self.device, dtype=torch.int32
        )
        self.decode_lens_buffer = torch.zeros(
            (scheduler_config.max_num_batched_tokens,),
            dtype=torch.int32,
            device=self.device,
        )
        # Shared workspace for decode seq_lens. Native MTP views this as
        # (B, max_decode_len) at runtime, keeping context_lens contiguous even
        # when max_decode_len is smaller than next_n.
        self.decode_seq_lens_buffer = torch.zeros(
            (scheduler_config.max_num_batched_tokens,),
            dtype=torch.int32,
            device=self.device,
        )
        self.arange_buffer = torch.arange(
            max(
                scheduler_config.max_num_seqs * next_n,
                scheduler_config.max_num_batched_tokens,
            ),
            dtype=torch.int32,
            device=self.device,
        )
        max_num_blocks_per_req = cdiv(
            self.vllm_config.model_config.max_model_len,
            self.kv_cache_spec.block_size * get_total_cp_world_size(),
        )
        self.expanded_block_table_buffer = torch.zeros(
            (
                scheduler_config.max_num_batched_tokens,
                max_num_blocks_per_req,
            ),
            dtype=torch.int32,
            device=self.device,
        )

        # See: DeepGMM/csrc/apis/attention.hpp
        self.scheduler_metadata_buffer = torch.empty(
            (self.num_sms + 1, 2), dtype=torch.int32, device=self.device
        )

        # KV compression. Default to 1 for no compression.
        self.compress_ratio = 1
        # Get compress_ratio for DeepseekV4 support
        if isinstance(self.kv_cache_spec, MLAAttentionSpec):
            self.compress_ratio = self.kv_cache_spec.compress_ratio

        # Pre-allocate buffers for CUDA graph compatibility when
        if self.compress_ratio > 1:
            # compress_ratio > 1 (DeepseekV4)
            # Compressed slot mapping output buffer
            self.compressed_slot_mapping_buffer = torch.zeros(
                (scheduler_config.max_num_batched_tokens,),
                dtype=torch.int64,
                device=self.device,
            )
            # Buffer for compressed seq_lens in decode path
            self.expanded_seq_lens_buffer = torch.zeros(
                (scheduler_config.max_num_batched_tokens,),
                dtype=torch.int32,
                device=self.device,
            )

    def _maybe_build_b12x_schedule_metadata(
        self,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        num_decode_tokens: int,
        requires_padding: bool,
    ) -> torch.Tensor | None:
        if not envs.VLLM_USE_B12X_SPARSE_INDEXER or requires_padding:
            return None

        schedule_seq_lens = seq_lens
        if schedule_seq_lens.dim() == 2:
            batch_size, next_n = schedule_seq_lens.shape
            if num_decode_tokens != int(batch_size * next_n):
                return None
            schedule_seq_lens = schedule_seq_lens.reshape(-1)
        if schedule_seq_lens.dim() != 1:
            return None

        from b12x.integration.indexer import (
            build_paged_mqa_schedule_metadata,
            uses_paged_mqa_schedule,
        )

        if not uses_paged_mqa_schedule(
            q_rows=int(schedule_seq_lens.shape[0]),
            max_pages=int(block_table.shape[1]),
        ):
            return None

        return build_paged_mqa_schedule_metadata(
            schedule_seq_lens.contiguous(),
            self.kv_cache_spec.storage_block_size,
            self.num_sms,
            out=self.scheduler_metadata_buffer,
        )

    def _maybe_build_deep_gemm_schedule_metadata(
        self,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        if current_platform.is_cuda():
            from vllm.utils.deep_gemm import (
                get_paged_mqa_logits_metadata,
                has_deep_gemm,
            )

            if has_deep_gemm():
                self.scheduler_metadata_buffer[:] = get_paged_mqa_logits_metadata(
                    seq_lens,
                    self.kv_cache_spec.storage_block_size,
                    self.num_sms,
                )
        return self.scheduler_metadata_buffer

    def _prepare_decode_tensors(
        self,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        decode_lens: torch.Tensor,
        decode_lens_cpu: torch.Tensor,
        query_start_loc: torch.Tensor,
        num_decodes: int,
        num_decode_tokens: int,
        use_native: bool,
        next_n: int,
        max_decode_len: int,
        global_seq_lens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, bool]:
        """Expand seq_lens/block_table/decode_lens for the decode kernels.

        Flatten path (not use_native, max_decode_len > 1):
          Each multi-token decode request is expanded into individual
          single-token entries so the kernel always sees next_n=1.

        Native path (use_native or max_decode_len == 1):
          Plain decode or spec-decode with 2D per-token context lengths.

        Returns (seq_lens, block_table, decode_lens, batch_size, requires_padding).
        seq_lens is 1D (batch_size,) for flatten/plain, 2D (B, max_decode_len)
        for native MTP.
        """
        min_decode_len = int(decode_lens_cpu.min().item())
        if not use_native and max_decode_len > 1:
            assert self.decode_seq_lens_buffer.dim() == 1
            if min_decode_len == max_decode_len:
                # Uniform decode lengths.
                num_decode_tokens = num_decodes * max_decode_len
                _prepare_uniform_decode_kernel[(num_decode_tokens,)](
                    seq_lens,
                    self.decode_seq_lens_buffer,
                    block_table,
                    block_table.stride(0),
                    self.expanded_block_table_buffer,
                    self.expanded_block_table_buffer.stride(0),
                    self.decode_lens_buffer,
                    max_decode_len,
                    BLOCK_SIZE=1024,
                )
                if global_seq_lens is not None and self.dcp_world_size > 1:
                    expanded_global = (
                        global_seq_lens[:num_decodes].unsqueeze(1)
                        - max_decode_len
                        + 1
                        + self.offsets_buffer[:max_decode_len]
                    ).reshape(-1)
                    self.decode_seq_lens_buffer[:num_decode_tokens].copy_(
                        get_dcp_local_seq_lens(
                            expanded_global,
                            self.dcp_world_size,
                            self.dcp_rank,
                            self.cp_kv_cache_interleave_size,
                        ),
                        non_blocking=True,
                    )
                self.decode_seq_lens_buffer[num_decode_tokens:] = 0
                seq_lens = self.decode_seq_lens_buffer[:num_decode_tokens]
                block_table = self.expanded_block_table_buffer[:num_decode_tokens]
                decode_lens = self.decode_lens_buffer[:num_decode_tokens]
                return seq_lens, block_table, decode_lens, num_decode_tokens, False
            else:
                # Variable decode lengths.
                # Assume 4 requests with seq_lens [10, 7, 12, 0] (the final req is
                # padding) and decode_lens [3, 1, 4, 0] in the below example comments.
                # The context lengths are therefore
                # [10-3, 7-1, 12-4, 0-0] = [7, 6, 8, 0].

                # 3 + 1 + 4 + 0 = 8
                actual_expanded = int(decode_lens_cpu.sum().item())

                # Fuse expanded_base and expanded_starts into a single
                # repeat_interleave:
                # seq_len_i = (context_start[b] - query_start_loc[b]) + arange[i] + 1
                # where context_start[b] = seq_lens[b] - decode_lens[b].
                # Example: offsets = [7-0, 6-3, 8-4, 0-8] = [7, 3, 4, -8]
                # expanded_offsets  = [7, 7, 7, 3, 4, 4, 4, 4]
                # result            = [8, 9, 10, 7, 9, 10, 11, 12]
                seq_lens_for_expansion = (
                    global_seq_lens if global_seq_lens is not None else seq_lens
                )
                expanded_offsets = torch.repeat_interleave(
                    seq_lens_for_expansion - decode_lens - query_start_loc,
                    decode_lens,
                    output_size=actual_expanded,
                )

                # [8, 9, 10, 7, 9, 10, 11, 12, ...] where ... is unused buffer space
                expanded_seq_lens = (
                    expanded_offsets + self.arange_buffer[:actual_expanded] + 1
                )
                if global_seq_lens is not None and self.dcp_world_size > 1:
                    expanded_seq_lens = get_dcp_local_seq_lens(
                        expanded_seq_lens,
                        self.dcp_world_size,
                        self.dcp_rank,
                        self.cp_kv_cache_interleave_size,
                    )
                self.decode_seq_lens_buffer[:actual_expanded] = expanded_seq_lens
                self.decode_seq_lens_buffer[actual_expanded:] = 0
                seq_lens = self.decode_seq_lens_buffer[:num_decode_tokens]

                # Give each of the flattened entries the same block table row as the
                # original request.
                self.expanded_block_table_buffer[:actual_expanded] = (
                    torch.repeat_interleave(
                        block_table, decode_lens, dim=0, output_size=actual_expanded
                    )
                )
                if actual_expanded < num_decode_tokens:
                    self.expanded_block_table_buffer[
                        actual_expanded:num_decode_tokens, 0
                    ] = 0
                block_table = self.expanded_block_table_buffer[:num_decode_tokens]

                # All reqs now have decode_len=1
                self.decode_lens_buffer[:num_decode_tokens] = 1
                decode_lens = self.decode_lens_buffer[:num_decode_tokens]
                return seq_lens, block_table, decode_lens, num_decode_tokens, False
        else:
            # Native path: plain decode (next_n==1) or spec decode
            # with 2D per-token context lengths (next_n > 1).
            #
            # When decode_lens are not truly uniform (e.g. some requests have
            # decode_len < next_n due to padding or short prefills), the simple
            # reshape in sparse_attn_indexer won't work. Use pack_seq_triton
            # (requires_padding) instead.
            requires_padding = min_decode_len != max_decode_len
            if use_native and next_n > 1:
                assert self.decode_seq_lens_buffer.dim() == 1
                # (B, max_decode_len): token j attends to
                # L - max_decode_len + j + 1 KV tokens.
                seq_lens_buffer = self.decode_seq_lens_buffer[
                    : num_decodes * max_decode_len
                ].view(num_decodes, max_decode_len)
                seq_lens_for_expansion = (
                    global_seq_lens if global_seq_lens is not None else seq_lens
                )
                expanded_seq_lens = (
                    seq_lens_for_expansion.unsqueeze(1)
                    - max_decode_len
                    + 1
                    + self.offsets_buffer[:max_decode_len]
                )
                if global_seq_lens is not None and self.dcp_world_size > 1:
                    expanded_seq_lens = get_dcp_local_seq_lens(
                        expanded_seq_lens.reshape(-1),
                        self.dcp_world_size,
                        self.dcp_rank,
                        self.cp_kv_cache_interleave_size,
                    ).reshape(num_decodes, max_decode_len)
                seq_lens_buffer[:] = expanded_seq_lens
                seq_lens = seq_lens_buffer
            return seq_lens, block_table, decode_lens, num_decodes, requires_padding

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DeepseekV32IndexerMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_tokens = common_attn_metadata.num_actual_tokens
        query_start_loc = common_attn_metadata.query_start_loc
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        seq_lens = common_attn_metadata.seq_lens
        slot_mapping = common_attn_metadata.slot_mapping
        block_table = common_attn_metadata.block_table_tensor

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold,
                require_uniform=not self.use_flattening,
            )
        )

        assert num_decodes + num_prefills == num_reqs
        assert num_decode_tokens + num_prefill_tokens == num_tokens

        compressed_slot_mapping = slot_mapping
        compressed_seq_lens = seq_lens
        indexer_seq_lens = (
            common_attn_metadata.dcp_local_seq_lens
            if common_attn_metadata.dcp_local_seq_lens is not None
            else seq_lens
        )
        if self.compress_ratio > 1:
            compressed_slot_mapping = get_compressed_slot_mapping(
                num_tokens,
                query_start_loc,
                seq_lens,
                block_table,
                self.kv_cache_spec.storage_block_size,
                self.compress_ratio,
                out=self.compressed_slot_mapping_buffer,
            )
            compressed_seq_lens = seq_lens // self.compress_ratio
            indexer_seq_lens = indexer_seq_lens // self.compress_ratio

        prefill_metadata = None
        if num_prefills > 0:
            # Prefill sparse-indexer chunks need exact lengths. Upper-bound
            # lengths are only safe for allocation; using them for DCP spans can
            # select future/non-local KV rows in long-context requests.
            global_seq_lens_cpu = common_attn_metadata.seq_lens_cpu
            has_dcp_local_seq_lens = self.dcp_world_size > 1
            if has_dcp_local_seq_lens:
                indexer_seq_lens_cpu = common_attn_metadata.dcp_local_seq_lens_cpu
                if indexer_seq_lens_cpu is None:
                    indexer_seq_lens_cpu = get_dcp_local_seq_lens(
                        global_seq_lens_cpu.to(torch.int32),
                        self.dcp_world_size,
                        self.dcp_rank,
                        self.cp_kv_cache_interleave_size,
                    )
            else:
                indexer_seq_lens_cpu = global_seq_lens_cpu
            seq_lens_cpu = global_seq_lens_cpu
            compressed_seq_lens_cpu = (
                indexer_seq_lens_cpu // self.compress_ratio
                if self.compress_ratio > 1
                else indexer_seq_lens_cpu
            )
            if _B12X_MLA_DEBUG and _debug_can_sync_cuda():
                debug_count = getattr(self, "_debug_prefill_build_count", 0)
                if debug_count < _B12X_MLA_DEBUG_LIMIT:
                    try:
                        payload = (
                            f"kind=indexer_prefill_build count={debug_count} "
                            f"dcp_world={self.dcp_world_size} "
                            f"dcp_rank={self.dcp_rank} "
                            f"has_cm_dcp={common_attn_metadata.dcp_local_seq_lens is not None} "
                            f"has_cm_dcp_cpu={common_attn_metadata.dcp_local_seq_lens_cpu is not None} "
                            f"global={_debug_int_tensor(global_seq_lens_cpu)} "
                            f"indexer={_debug_int_tensor(indexer_seq_lens_cpu)}"
                        )
                        logger.warning("B12X_MLA_DEBUG %s", payload)
                        with open(_B12X_MLA_DEBUG_FILE, "a", encoding="utf-8") as f:
                            f.write(payload + "\n")
                    except Exception:
                        logger.exception("Failed to write B12X indexer build debug")
                    setattr(self, "_debug_prefill_build_count", debug_count + 1)
            prefill_query_lens_cpu = torch.diff(
                query_start_loc_cpu[num_decodes : num_decodes + num_prefills + 1]
            )
            max_logits_bytes = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
            chunk_specs = split_indexer_prefill_chunks(
                compressed_seq_lens_cpu[num_decodes:],
                prefill_query_lens_cpu,
                self.max_prefill_buffer_size,
                max_logits_bytes,
                request_offset=num_decodes,
            )

            chunks = []
            for req_slice, query_slice in chunk_specs:
                if has_dcp_local_seq_lens:
                    if self.compress_ratio != 1:
                        raise RuntimeError(
                            "DCP sparse indexer prefill with compressed MLA KV "
                            "is not wired yet."
                        )
                    metadata = build_dcp_prefill_chunk_metadata(
                        req_slice.start,
                        req_slice.stop,
                        query_start_loc_cpu,
                        indexer_seq_lens_cpu,
                        global_seq_lens_cpu,
                        common_attn_metadata.block_table_tensor,
                        self.device,
                        query_slice=query_slice,
                        skip_kv_gather=query_slice.start > 0,
                        dcp_world_size=self.dcp_world_size,
                        dcp_rank=self.dcp_rank,
                        cp_kv_cache_interleave_size=(
                            self.cp_kv_cache_interleave_size
                        ),
                    )
                else:
                    metadata = build_prefill_chunk_metadata(
                        req_slice.start,
                        req_slice.stop,
                        query_start_loc,
                        query_start_loc_cpu,
                        seq_lens,
                        compressed_seq_lens,
                        compressed_seq_lens_cpu,
                        common_attn_metadata.block_table_tensor,
                        self.compress_ratio,
                        query_slice=query_slice,
                        skip_kv_gather=query_slice.start > 0,
                    )
                # Skip when total_seq_lens is 0 (i.e., no compressed token).
                if metadata is not None:
                    chunks.append(metadata)
            prefill_metadata = DeepseekV32IndexerPrefillMetadata(chunks)

        decode_metadata = None
        if num_decodes > 0:
            torch.diff(
                common_attn_metadata.query_start_loc[: num_decodes + 1],
                out=self.decode_lens_buffer[:num_decodes],
            )
            decode_lens = self.decode_lens_buffer[:num_decodes]
            decode_lens_cpu = torch.diff(
                common_attn_metadata.query_start_loc_cpu[: num_decodes + 1]
            )

            global_decode_seq_lens = common_attn_metadata.seq_lens[:num_decodes]
            seq_lens = (
                common_attn_metadata.dcp_local_seq_lens[:num_decodes]
                if common_attn_metadata.dcp_local_seq_lens is not None
                else global_decode_seq_lens
            )
            block_table = common_attn_metadata.block_table_tensor[:num_decodes, ...]

            max_decode_len = int(decode_lens_cpu.max().item())
            next_n = 1 + self.num_speculative_tokens
            use_native = not self.use_flattening and max_decode_len <= next_n

            seq_lens, block_table, decode_lens, batch_size, requires_padding = (
                self._prepare_decode_tensors(
                    seq_lens=seq_lens,
                    block_table=block_table,
                    decode_lens=decode_lens,
                    decode_lens_cpu=decode_lens_cpu,
                    query_start_loc=common_attn_metadata.query_start_loc[:num_decodes],
                    num_decodes=num_decodes,
                    num_decode_tokens=num_decode_tokens,
                    use_native=use_native,
                    next_n=next_n,
                    max_decode_len=max_decode_len,
                    global_seq_lens=global_decode_seq_lens
                    if common_attn_metadata.dcp_local_seq_lens is not None
                    else None,
                )
            )

            # For DeepseekV4 (compress_ratio > 1), the indexer KV cache stores
            # compressed tokens. Convert uncompressed seq_lens to compressed.
            if self.compress_ratio > 1:
                # True iff seq_lens aliases decode_seq_lens_buffer (flatten or
                # native wrote it); False iff it aliases common_attn_metadata.
                seq_lens_is_local_view = (use_native and next_n > 1) or (
                    not use_native and max_decode_len > 1
                )
                if seq_lens_is_local_view:
                    seq_lens //= self.compress_ratio
                else:
                    # Copy to avoid mutating shared state; keeps CG address stable.
                    self.expanded_seq_lens_buffer[:num_decodes] = (
                        seq_lens // self.compress_ratio
                    )
                    self.expanded_seq_lens_buffer[num_decodes:num_decode_tokens] = 0
                    seq_lens = self.expanded_seq_lens_buffer[:num_decode_tokens]

            # Non-MTP: deep_gemm paged MQA logits requires 2D context_lens
            # (csrc/apis/attention.hpp). Unsqueeze to (B, 1) so downstream
            # kernels see the same (B, next_n) layout as the MTP path.
            if seq_lens.dim() == 1:
                seq_lens = seq_lens.unsqueeze(-1)

            if envs.VLLM_USE_B12X_SPARSE_INDEXER:
                schedule_metadata = self._maybe_build_b12x_schedule_metadata(
                    seq_lens,
                    block_table,
                    num_decode_tokens,
                    requires_padding,
                )
            else:
                # DeepGEMM is required for paged MQA logits on CUDA devices.
                schedule_metadata = self._maybe_build_deep_gemm_schedule_metadata(
                    seq_lens
                )

            decode_metadata = DeepSeekV32IndexerDecodeMetadata(
                block_table=block_table,
                seq_lens=seq_lens,
                decode_lens=decode_lens,
                requires_padding=requires_padding,
                schedule_metadata=schedule_metadata,
                compress_ratio=self.compress_ratio,
            )

        attn_metadata = DeepseekV32IndexerMetadata(
            seq_lens=common_attn_metadata.seq_lens,
            max_seq_len=common_attn_metadata.max_seq_len,
            slot_mapping=compressed_slot_mapping,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            prefill=prefill_metadata,
            decode=decode_metadata,
        )

        return attn_metadata


def build_dcp_prefill_chunk_metadata(
    start_idx: int,
    end_idx: int,
    query_start_loc_cpu: torch.Tensor,
    local_seq_lens_cpu: torch.Tensor,
    global_seq_lens_cpu: torch.Tensor,
    block_table: torch.Tensor,
    device: torch.device,
    query_slice: slice | None = None,
    skip_kv_gather: bool = False,
    *,
    dcp_world_size: int = 1,
    dcp_rank: int = 0,
    cp_kv_cache_interleave_size: int = 1,
) -> DeepseekV32IndexerPrefillChunkMetadata:
    local_lens = local_seq_lens_cpu[start_idx:end_idx]
    global_lens = global_seq_lens_cpu[start_idx:end_idx]
    total_seq_lens = int(local_lens.sum().item())
    num_reqs = end_idx - start_idx

    prefill_query_start_loc = (
        query_start_loc_cpu[start_idx : end_idx + 1] - query_start_loc_cpu[start_idx]
    )
    cu_seqlen_ks, cu_seqlen_ke = _kv_spans_from_batches(
        prefill_query_start_loc,
        local_lens,
        device,
        query_slice=query_slice,
        dcp_world_size=dcp_world_size,
        dcp_rank=dcp_rank,
        cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
        global_seq_len_per_batch=global_lens,
    )

    if _B12X_MLA_DEBUG and _debug_can_sync_cuda():
        debug_count = getattr(build_dcp_prefill_chunk_metadata, "_debug_count", 0)
        if debug_count < _B12X_MLA_DEBUG_LIMIT:
            try:
                payload = (
                    f"kind=indexer_dcp_prefill count={debug_count} "
                    f"reqs={num_reqs} req_slice={start_idx}:{end_idx} "
                    f"query_slice={query_slice.start if query_slice else 0}:"
                    f"{query_slice.stop if query_slice else int(prefill_query_start_loc[-1].item())} "
                    f"total_local={total_seq_lens} "
                    f"local={_debug_int_tensor(local_lens)} "
                    f"global={_debug_int_tensor(global_lens)} "
                    f"qsl={_debug_int_tensor(prefill_query_start_loc)} "
                    f"ks_min={int(cu_seqlen_ks.min().item()) if cu_seqlen_ks.numel() else -1} "
                    f"ks_max={int(cu_seqlen_ks.max().item()) if cu_seqlen_ks.numel() else -1} "
                    f"ke_min={int(cu_seqlen_ke.min().item()) if cu_seqlen_ke.numel() else -1} "
                    f"ke_max={int(cu_seqlen_ke.max().item()) if cu_seqlen_ke.numel() else -1} "
                    f"empty_rows={int((cu_seqlen_ke <= cu_seqlen_ks).sum().item()) if cu_seqlen_ke.numel() else 0} "
                    f"ks={_debug_int_tensor(cu_seqlen_ks)} "
                    f"ke={_debug_int_tensor(cu_seqlen_ke)}"
                )
                logger.warning("B12X_MLA_DEBUG %s", payload)
                with open(_B12X_MLA_DEBUG_FILE, "a", encoding="utf-8") as f:
                    f.write(payload + "\n")
            except Exception:
                logger.exception("Failed to write B12X indexer debug payload")
            setattr(build_dcp_prefill_chunk_metadata, "_debug_count", debug_count + 1)

    if num_reqs == 1:
        token_to_seq = torch.zeros(total_seq_lens, dtype=torch.int32, device=device)
        cu_seq_lens = torch.tensor(
            [0, total_seq_lens], dtype=torch.int32, device=device
        )
    else:
        seq_idx = torch.arange(0, num_reqs, dtype=torch.int32)
        token_to_seq = torch.repeat_interleave(
            seq_idx, local_lens, output_size=total_seq_lens
        ).to(device)
        cu_seq_lens = (
            torch.cat(
                [
                    torch.zeros(1, dtype=torch.int32),
                    local_lens.cumsum(dim=0).to(torch.int32),
                ]
            )
            .to(torch.int32)
            .to(device)
        )

    token_start = query_start_loc_cpu[start_idx].item()
    if query_slice is not None:
        token_end = token_start + query_slice.stop
        token_start = token_start + query_slice.start
        skip_kv_gather = skip_kv_gather or query_slice.start > 0
    else:
        token_end = query_start_loc_cpu[end_idx].item()

    return DeepseekV32IndexerPrefillChunkMetadata(
        cu_seqlen_ks=cu_seqlen_ks,
        cu_seqlen_ke=cu_seqlen_ke,
        cu_seq_lens=cu_seq_lens,
        token_to_seq=token_to_seq,
        total_seq_lens=total_seq_lens,
        block_table=block_table[start_idx:end_idx],
        token_start=token_start,
        token_end=token_end,
        num_reqs=num_reqs,
        skip_kv_gather=skip_kv_gather,
    )


def build_prefill_chunk_metadata(
    start_idx: int,
    end_idx: int,
    query_start_loc: torch.Tensor,
    query_start_loc_cpu: torch.Tensor,
    uncompressed_seq_lens: torch.Tensor,
    compressed_seq_lens: torch.Tensor,
    compressed_seq_lens_cpu: torch.Tensor,
    block_table: torch.Tensor,
    compress_ratio: int,
    query_slice: slice | None = None,
    skip_kv_gather: bool = False,
) -> DeepseekV32IndexerPrefillChunkMetadata | None:
    total_seq_lens = compressed_seq_lens_cpu[start_idx:end_idx].sum().item()
    if total_seq_lens == 0:
        return None

    num_reqs = end_idx - start_idx
    device = block_table.device
    token_to_seq = torch.empty(total_seq_lens, dtype=torch.int32, device=device)

    cu_seq_lens = torch.empty(num_reqs + 1, dtype=torch.int32, device=device)
    # Assigning to slice avoids cpu sync.
    cu_seq_lens[:1] = 0
    torch.cumsum(compressed_seq_lens[start_idx:end_idx], dim=0, out=cu_seq_lens[1:])

    query_start_loc = (
        query_start_loc[start_idx : end_idx + 1] - query_start_loc[start_idx]
    )

    total_query_len = int(
        (query_start_loc_cpu[end_idx] - query_start_loc_cpu[start_idx]).item()
    )
    if query_slice is not None:
        qs_start = query_slice.start
        qs_stop = query_slice.stop
    else:
        qs_start = 0
        qs_stop = total_query_len
    output_query_len = qs_stop - qs_start

    cu_seq_len_ks = torch.empty(output_query_len, dtype=torch.int32, device=device)
    cu_seq_len_ke = torch.empty(output_query_len, dtype=torch.int32, device=device)

    _build_prefill_chunk_metadata_kernel[(num_reqs,)](
        query_start_loc,
        uncompressed_seq_lens[start_idx:end_idx],
        cu_seq_lens,
        token_to_seq,
        cu_seq_len_ks,
        cu_seq_len_ke,
        qs_start,
        qs_stop,
        BLOCK_SIZE=1024,
        COMPRESS_RATIO=compress_ratio,
    )

    token_start = query_start_loc_cpu[start_idx].item()
    if query_slice is not None:
        token_end = token_start + qs_stop
        token_start = token_start + qs_start
        skip_kv_gather = skip_kv_gather or qs_start > 0
    else:
        token_end = query_start_loc_cpu[end_idx].item()

    return DeepseekV32IndexerPrefillChunkMetadata(
        cu_seqlen_ks=cu_seq_len_ks,
        cu_seqlen_ke=cu_seq_len_ke,
        cu_seq_lens=cu_seq_lens,
        token_to_seq=token_to_seq,
        total_seq_lens=total_seq_lens,
        block_table=block_table[start_idx:end_idx],
        token_start=token_start,
        token_end=token_end,
        num_reqs=num_reqs,
        skip_kv_gather=skip_kv_gather,
    )


@triton.jit
def _build_prefill_chunk_metadata_kernel(
    # Inputs
    query_start_loc_ptr,
    uncompressed_seq_lens_ptr,
    cu_compressed_seq_lens_ptr,
    # Outputs
    token_to_seq_ptr,
    cu_compressed_seq_len_ks_ptr,
    cu_compressed_seq_len_ke_ptr,
    query_slice_start,
    query_slice_stop,
    BLOCK_SIZE: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
):
    batch_idx = tl.program_id(0)

    query_start = tl.load(query_start_loc_ptr + batch_idx)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    query_len = query_end - query_start

    seq_start = tl.load(cu_compressed_seq_lens_ptr + batch_idx)
    seq_end = tl.load(cu_compressed_seq_lens_ptr + batch_idx + 1)
    compressed_seq_len = seq_end - seq_start

    uncompressed_seq_len = tl.load(uncompressed_seq_lens_ptr + batch_idx)
    start_pos = uncompressed_seq_len - query_len

    for i in range(0, query_len, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        abs_pos = query_start + offset
        mask = (
            (offset < query_len)
            & (abs_pos >= query_slice_start)
            & (abs_pos < query_slice_stop)
        )
        out_pos = abs_pos - query_slice_start

        # Compute cu_seq_len_ks
        tl.store(cu_compressed_seq_len_ks_ptr + out_pos, seq_start, mask=mask)

        # Compute cu_seq_len_ke
        seq_len_per_token = (start_pos + 1 + offset) // COMPRESS_RATIO
        tl.store(
            cu_compressed_seq_len_ke_ptr + out_pos,
            seq_start + seq_len_per_token,
            mask=mask,
        )

    # Compute token_to_seq
    for i in range(0, compressed_seq_len, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        mask = offset < compressed_seq_len
        tl.store(token_to_seq_ptr + seq_start + offset, batch_idx, mask=mask)
