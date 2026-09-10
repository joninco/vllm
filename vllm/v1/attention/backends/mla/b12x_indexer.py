# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B12x DSA indexer for non-compressed sparse MLA models."""

import bisect
import os
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.distributed as dist
from torch import nn

import vllm.envs as envs
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed import get_dcp_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
from vllm.utils.b12x import B12xWarmupUnit, get_b12x_dsa_indexer
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.mla import b12x_topk_sort
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerBackend,
    DeepSeekV32IndexerDecodeMetadata,
    DeepseekV32IndexerMetadata,
    DeepseekV32IndexerMetadataBuilder,
    DeepseekV32IndexerPrefillChunkMetadata,
    split_indexer_prefill_chunks,
)
from vllm.v1.kv_cache_interface import KVCacheSpec
from vllm.v1.worker.block_table import get_block_table_width
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_logger(__name__)

_INDEX_HEAD_DIM = 128
_INDEX_SCALE_BYTES = 4
_INDEX_PAGE_SIZE = 64
_INDEX_PAGE_WIDTH = _INDEX_PAGE_SIZE * (_INDEX_HEAD_DIM + _INDEX_SCALE_BYTES)
_PREFILL_PROFILE_SUPERTILE_K = 32 * 1024


def _prefill_profile_q_rows(max_q_rows: int) -> int:
    max_logits_elems = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024 // 4
    supertile_k = int(
        os.environ.get("B12X_PAGED_INDEX_SUPERTILE_K", _PREFILL_PROFILE_SUPERTILE_K)
    )
    supertile_k = max(supertile_k, 256)
    return min(max(int(max_q_rows), 1), max(1, max_logits_elems // supertile_k))


def _is_current_stream_capturing(tensor: torch.Tensor) -> bool:
    return tensor.is_cuda and torch.cuda.is_current_stream_capturing()


@dataclass
class B12xIndexerDecodeMetadata(DeepSeekV32IndexerDecodeMetadata):
    active_width: torch.Tensor | None = None


class B12xIndexerMetadataBuilder(DeepseekV32IndexerMetadataBuilder):
    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: KVCacheSpec,
    ) -> AttentionCGSupport:
        return AttentionCGSupport.ALWAYS

    def __init__(self, *args, block_table_width: int, **kwargs) -> None:
        super().__init__(*args, block_table_width=block_table_width, **kwargs)
        self.use_flattening = False
        self.supports_varlen = False
        self.active_width_buffer = torch.zeros(
            (1,), dtype=torch.int32, device=self.device
        )
        if self.dcp_world_size > 1 and self.active_width_buffer.is_cuda:
            from b12x.comm.pcie._dcp_attention_metadata import (
                precompile_dcp_sequence_lengths,
            )

            precompile_dcp_sequence_lengths(self.active_width_buffer.device.index)

    def _dcp_localize_decode_seq_lens(
        self,
        seq_lens: torch.Tensor,
        num_decodes: int,
        seq_lens_is_buffer_view: bool,
    ) -> torch.Tensor:
        if not (
            self.dcp_world_size > 1
            and seq_lens.is_cuda
            and seq_lens.dtype == torch.int32
            and seq_lens.is_contiguous()
        ):
            return super()._dcp_localize_decode_seq_lens(
                seq_lens, num_decodes, seq_lens_is_buffer_view
            )
        from b12x.comm.pcie._dcp_attention_metadata import (
            localize_dcp_sequence_lengths,
        )

        out = (
            seq_lens
            if seq_lens_is_buffer_view
            else self.decode_seq_lens_buffer[:num_decodes]
        )
        localize_dcp_sequence_lengths(
            seq_lens.view(-1),
            out.view(-1),
            self.dcp_world_size,
            self.dcp_rank,
            self.cp_kv_cache_interleave_size,
        )
        return out

    def _supports_native_decode(self, next_n: int) -> bool:
        return True

    def _split_prefill_chunks(
        self,
        compressed_seq_lens_cpu: torch.Tensor,
        prefill_query_lens_cpu: torch.Tensor,
        num_decodes: int,
        max_logits_bytes: int,
    ) -> list[tuple[slice, slice]]:
        return [
            chunk
            for prefill_idx in range(len(prefill_query_lens_cpu))
            for chunk in split_indexer_prefill_chunks(
                compressed_seq_lens_cpu[
                    num_decodes + prefill_idx : num_decodes + prefill_idx + 1
                ],
                prefill_query_lens_cpu[prefill_idx : prefill_idx + 1],
                self.max_prefill_buffer_size,
                max_logits_bytes,
                request_offset=num_decodes + prefill_idx,
            )
        ]

    def build(self, *args, **kwargs) -> DeepseekV32IndexerMetadata:
        metadata = super().build(*args, **kwargs)
        self.active_width_buffer.fill_(int(metadata.max_seq_len))
        if metadata.decode is not None:
            decode = metadata.decode
            seq_lens = decode.seq_lens.reshape(-1).contiguous()
            fields = vars(decode).copy()
            fields["seq_lens"] = seq_lens
            fields["schedule_metadata"] = None
            metadata.decode = B12xIndexerDecodeMetadata(
                **fields,
                active_width=self.active_width_buffer,
            )
        return metadata


class B12xIndexerBackend(DeepseekV32IndexerBackend):
    @classmethod
    def supports_pcp(cls) -> bool:
        return False

    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls) -> bool:
        return False

    @staticmethod
    def get_name() -> str:
        return "B12X_INDEXER"

    @staticmethod
    def get_builder_cls() -> type[B12xIndexerMetadataBuilder]:
        return B12xIndexerMetadataBuilder


class B12xIndexerCache(DeepseekV32IndexerCache):
    def get_attn_backend(self) -> type[B12xIndexerBackend]:
        return B12xIndexerBackend


def _require_b12x_indexer() -> Any:
    module = get_b12x_dsa_indexer()
    if module is None:
        raise RuntimeError("B12X sparse MLA requires `pip install vllm[b12x]`.")
    if not module.is_supported():
        raise RuntimeError("B12X sparse indexer is not supported on this device.")
    if int(module.PAGED_INDEX_PAGE_SIZE) != _INDEX_PAGE_SIZE:
        raise RuntimeError(
            "B12X sparse indexer page size changed: expected "
            f"{_INDEX_PAGE_SIZE}, got {module.PAGED_INDEX_PAGE_SIZE}."
        )
    for name in (
        "Caps",
        "bind",
        "plan",
        "run",
    ):
        getattr(module, name)
    return module


def _flatten_index_cache(kv_cache: torch.Tensor) -> torch.Tensor:
    expected_tail = (_INDEX_PAGE_SIZE, _INDEX_HEAD_DIM + _INDEX_SCALE_BYTES)
    if (
        kv_cache.ndim != 3
        or kv_cache.dtype != torch.uint8
        or tuple(kv_cache.shape[1:]) != expected_tail
    ):
        raise RuntimeError(
            "B12X indexer cache must have shape "
            f"[num_blocks, {expected_tail[0]}, {expected_tail[1]}] and dtype "
            f"uint8, got shape={tuple(kv_cache.shape)} dtype={kv_cache.dtype}."
        )
    if kv_cache.stride(1) != expected_tail[1] or kv_cache.stride(2) != 1:
        raise RuntimeError(
            "B12X indexer cache requires contiguous page payloads, got stride "
            f"{tuple(kv_cache.stride())}."
        )
    return kv_cache.as_strided(
        (int(kv_cache.shape[0]), _INDEX_PAGE_WIDTH),
        (int(kv_cache.stride(0)), 1),
    )


def _plan_output_space(
    output_physical_slots: bool, sort_selection: bool, q_rows: int
) -> str:
    """Index space an indexer plan of ``q_rows`` rows emits. Physical slots
    come from the indexer itself unless the selection sort serves that row
    count, in which case the plan emits logical positions for the sort to
    convert (``b12x_topk_sort``)."""
    if not output_physical_slots:
        return "logical"
    if sort_selection and b12x_topk_sort.active(q_rows):
        return "logical"
    return "physical"


def _decode_plan_sizes(
    capture_sizes: list[int], max_num_seqs: int, sort_selection: bool
) -> list[int]:
    """Row counts of the decode plans prepared before CUDA-graph capture.

    Args:
        capture_sizes: The CUDA-graph capture sizes.
        max_num_seqs: The largest decode batch.
        sort_selection: Whether the selection sort serves this indexer.

    Returns:
        The capture sizes within ``max_num_seqs``, ``max_num_seqs`` itself
        and, with the sort active and its row gate below ``max_num_seqs``,
        the gate itself, so that a decode batch within the gate selects a
        plan emitting logical positions for the sort whatever the capture
        sizes are (a plan is selected by the smallest prepared row count at
        or above the batch).
    """
    sizes = {int(size) for size in capture_sizes if 0 < int(size) <= max_num_seqs}
    sizes.add(max_num_seqs)
    if sort_selection and 0 < b12x_topk_sort.MAX_TOKENS < max_num_seqs:
        sizes.add(b12x_topk_sort.MAX_TOKENS)
    return sorted(sizes)


def _plan_emits_physical_slots(plan: Any) -> bool:
    """Whether an indexer plan writes physical cache slots. The scratch caps
    a compiled plan carries record the choice as ``output_physical_slots``;
    the API caps it was built from record it as ``output_index_space``."""
    caps = plan.caps
    physical = getattr(caps, "output_physical_slots", None)
    if physical is None:
        physical = getattr(caps, "output_index_space", "logical") == "physical"
    return bool(physical)


def _run_paged_topk(
    *,
    module: Any,
    plan: Any,
    q: torch.Tensor,
    weights: torch.Tensor,
    kv_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    active_width: torch.Tensor | None,
    output: torch.Tensor,
    return_scores: bool = False,
) -> torch.Tensor | None:
    specs = plan.shapes_and_dtypes()
    if return_scores:
        # Scores occupy the same prefix during selection and candidate packing.
        specs = (((q.shape[0], output.shape[1]), torch.float32), *specs)
    buffers = current_workspace_manager().get_simultaneous(*specs)
    scores, scratch = (buffers[0], buffers[1:]) if return_scores else (None, buffers)
    if active_width is None:
        raise RuntimeError("B12X DSA requires a device active-width scalar.")
    binding = module.bind(
        plan,
        scratch=scratch,
        q_fp8=q,
        query_weights=weights,
        index_k_cache=_flatten_index_cache(kv_cache),
        page_table=block_table,
        cache_lengths=seq_lens,
        active_width=active_width,
        output_indices=output,
        output_scores=scores,
    )
    module.run(binding)
    return scores


def _dcp_merge_shapes(rows: int, topk: int, world_size: int):
    return (
        ((rows, topk), torch.float32),
        ((rows, topk, 2), torch.float32),
        ((world_size, rows, topk, 2), torch.float32),
    )


def _gather_dcp_candidates(group, packed, gathered) -> None:
    communicator = group.device_communicator
    pynccl = getattr(communicator, "pynccl_comm", None)
    if pynccl is not None and not pynccl.disabled:
        pynccl.all_gather(gathered, packed)
    else:
        dist.all_gather_into_tensor(
            gathered.flatten(0, 1),
            packed,
            group=group.device_group,
        )


def _merge_dcp_topk(
    indices: torch.Tensor,
    scores: torch.Tensor,
    dcp_rank: int,
    dcp_world_size: int,
    interleave: int,
    shard_group=None,
) -> None:
    if dcp_world_size <= 1 or indices.numel() == 0:
        return
    topk = int(indices.shape[1])
    if topk not in (512, 1024, 2048):
        raise RuntimeError(
            "B12X DCP indexer merge requires index_topk in (512, 1024, 2048), "
            f"got {topk}."
        )
    from b12x.comm.pcie.dcp_candidate_topk import (
        pack_dcp_candidates,
        rank_major_topk,
    )

    score_view, packed, gathered = current_workspace_manager().get_simultaneous(
        *_dcp_merge_shapes(indices.shape[0], topk, dcp_world_size)
    )
    if scores.data_ptr() != score_view.data_ptr():
        raise RuntimeError("DCP scores must use the reserved merge workspace prefix")
    pack_dcp_candidates(indices, scores, packed, dcp_rank, dcp_world_size, interleave)
    from vllm.v1.attention.ops.b12x_dcp import active_dcp_transport

    # Captured DCP transport is bound to the attention group's geometry.
    # Optional indexer replicas use their own communicator.
    group = get_dcp_group() if shard_group is None else shard_group
    binding = active_dcp_transport() if shard_group is None else None
    if binding is not None and binding.merge_candidates(packed, indices):
        return
    _gather_dcp_candidates(group, packed, gathered)
    rank_major_topk(gathered, indices)


def _prefill_owner_shapes(rows: int, topk: int, shards: int):
    return (
        ((rows, topk), torch.float32),
        ((rows, topk, 2), torch.float32),
        ((rows, topk, 2), torch.float32),
        ((rows // shards, topk), torch.int32),
    )


def _is_target_indexer(prefix: str, target_layers: int) -> bool:
    from vllm.model_executor.models.utils import extract_layer_index

    try:
        index = extract_layer_index(prefix)
    except (AssertionError, IndexError, ValueError):
        return False
    return 0 <= index < target_layers


def _restore_prefill_indices(group, local: torch.Tensor, output: torch.Tensor) -> None:
    """Restore row order using nonaliasing input for every collective backend."""
    rows = local.shape[0]
    expected = output.narrow(0, group.rank_in_group * rows, rows)
    if (
        output.shape[0] != rows * group.world_size
        or expected.data_ptr() != local.data_ptr()
    ):
        raise RuntimeError("Query-split output must alias its replica row interval")
    (send,) = current_workspace_manager().get_simultaneous(
        (tuple(local.shape), torch.int32)
    )
    send.copy_(local)
    _gather_dcp_candidates(group, send, output.view(group.world_size, rows, -1))


def _exchange_prefill_owner_candidates(group, packed, received) -> None:
    """Exchange equal row partitions on the existing shard communicator."""
    communicator = getattr(group, "device_communicator", None)
    pynccl = getattr(communicator, "pynccl_comm", None)
    if pynccl is None or pynccl.disabled:
        dist.all_to_all_single(received, packed, group=group.device_group)
        return
    size, rank = group.world_size, group.rank_in_group
    if (
        packed.shape != received.shape
        or packed.dtype != received.dtype
        or packed.device != received.device
        or not packed.is_contiguous()
        or not received.is_contiguous()
        or packed.shape[0] % size
    ):
        raise ValueError("Owner exchange requires matching contiguous equal partitions")
    nbytes = packed.numel() * packed.element_size()
    if max(packed.data_ptr(), received.data_ptr()) < min(
        packed.data_ptr() + nbytes, received.data_ptr() + nbytes
    ):
        raise ValueError("Owner exchange send and receive buffers must not overlap")
    sends = packed.chunk(size, dim=0)
    receives = received.chunk(size, dim=0)
    stream = torch.cuda.current_stream(device=packed.device)
    with torch.cuda.stream(stream):
        receives[rank].copy_(sends[rank])
        pynccl.group_start()
        try:
            for peer in range(size):
                if peer != rank:
                    pynccl.send(sends[peer], peer, stream)
                    pynccl.recv(receives[peer], peer, stream)
        finally:
            pynccl.group_end()


def _merge_prefill_topk_by_owner(
    indices: torch.Tensor,
    scores: torch.Tensor,
    output: torch.Tensor,
    shard_group,
    tp_group,
    interleave: int,
) -> bool:
    """Select each row once using the same stable selector as replicated DCP."""
    shards = int(shard_group.world_size)
    rows, topk = indices.shape
    if shards <= 1 or rows == 0 or rows % shards:
        return False
    if tp_group.world_size % shards:
        return False
    replicas = tp_group.world_size // shards
    if output.shape != (rows * replicas, topk):
        return False
    rank = tp_group.rank_in_group
    if rank % shards != shard_group.rank_in_group:
        raise RuntimeError("Owner merge requires matching TP and indexer shard order")
    local = output.narrow(0, (rank // shards) * rows, rows)
    if local.data_ptr() != indices.data_ptr():
        raise RuntimeError("Owner merge input must alias its TP query partition")
    from b12x.comm.pcie.dcp_candidate_topk import pack_dcp_candidates, rank_major_topk

    score_view, packed, received, selected = (
        current_workspace_manager().get_simultaneous(
            *_prefill_owner_shapes(rows, topk, shards)
        )
    )
    if score_view.data_ptr() != scores.data_ptr():
        raise RuntimeError("Owner scores must use the reserved workspace prefix")
    pack_dcp_candidates(
        indices, scores, packed, shard_group.rank_in_group, shards, interleave
    )
    _exchange_prefill_owner_candidates(shard_group, packed, received)
    rank_major_topk(received.view(shards, rows // shards, topk, 2), selected)
    _gather_dcp_candidates(
        tp_group, selected, output.view(tp_group.world_size, rows // shards, topk)
    )
    return True


class B12xSparseIndexer(nn.Module):
    def __init__(
        self,
        k_cache,
        quant_block_size: int,
        scale_fmt: str,
        topk_tokens: int,
        head_dim: int,
        max_model_len: int,
        max_total_seq_len: int,
        topk_indices_buffer: torch.Tensor | None,
        skip_k_cache_insert: bool = False,
        use_fp4_cache: bool = False,
        compress_ratio: int = 1,
        num_q_heads: int | None = None,
        output_physical_slots: bool = False,
    ) -> None:
        super().__init__()
        del quant_block_size, scale_fmt, max_total_seq_len
        if not skip_k_cache_insert:
            raise ValueError("B12X requires the fused DSA index-cache insert path.")
        if use_fp4_cache:
            raise ValueError("B12X indexing requires the FP8 index cache.")
        if compress_ratio != 1:
            raise ValueError(
                "The non-compressed B12X indexer requires compress_ratio=1."
            )
        if head_dim != _INDEX_HEAD_DIM:
            raise ValueError(
                f"B12X indexing requires head_dim={_INDEX_HEAD_DIM}, got {head_dim}."
            )
        if topk_indices_buffer is None:
            raise ValueError("B12X indexing requires a top-k output buffer.")
        if num_q_heads is None or int(num_q_heads) <= 0:
            raise ValueError(
                "B12X indexing requires a positive index query head count."
            )
        self._module = _require_b12x_indexer()
        self.k_cache = k_cache
        self.topk_tokens = int(topk_tokens)
        self.max_model_len = int(max_model_len)
        self.topk_indices_buffer = topk_indices_buffer
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        self.attention_dcp_world_size = (
            vllm_config.parallel_config.decode_context_parallel_size
        )
        # Attention DCP consumes global token IDs even when indexer KV is local.
        self.output_physical_slots = bool(output_physical_slots) and (
            self.attention_dcp_world_size == 1
        )
        # Deterministic selection order (b12x_topk_sort): plans whose row
        # count is within the sort gate emit logical positions, which the
        # sort rewrites ascending and converts to physical slots in place.
        self.sort_selection = (
            self.output_physical_slots
            and b12x_topk_sort.ENABLED
            and b12x_topk_sort.is_supported(topk_indices_buffer.device)
        )
        self.active_width_cap = torch.full(
            (1,),
            self.max_model_len,
            dtype=torch.int32,
            device=topk_indices_buffer.device,
        )
        max_q_rows = int(topk_indices_buffer.shape[0])
        max_page_table_width = get_block_table_width(
            max(1, (self.max_model_len + _INDEX_PAGE_SIZE - 1) // _INDEX_PAGE_SIZE),
            _INDEX_PAGE_SIZE,
        )
        scheduler_config = vllm_config.scheduler_config
        parallel_config = vllm_config.parallel_config
        max_num_seqs = int(scheduler_config.max_num_seqs)

        def make_plan(*, mode: str, q_rows: int):
            return self._module.plan(
                self._module.Caps(
                    device=topk_indices_buffer.device,
                    num_q_heads=int(num_q_heads),
                    max_q_rows=q_rows,
                    max_page_table_width=max_page_table_width,
                    topk=self.topk_tokens,
                    mode=mode,
                    max_batch=q_rows if mode == "decode" else max_num_seqs,
                    output_index_space=_plan_output_space(
                        self.output_physical_slots, self.sort_selection, q_rows
                    ),
                )
            )

        self._make_plan = make_plan
        capture_sizes = vllm_config.compilation_config.cudagraph_capture_sizes or []
        self._decode_plan_sizes = _decode_plan_sizes(
            capture_sizes, max_num_seqs, self.sort_selection
        )
        self._decode_plans = {
            rows: make_plan(mode="decode", q_rows=rows)
            for rows in self._decode_plan_sizes
        }
        prefill_profile_rows = _prefill_profile_q_rows(max_q_rows)
        prefill_plan_sizes = {prefill_profile_rows}
        if self.sort_selection and 0 < b12x_topk_sort.MAX_TOKENS < prefill_profile_rows:
            # Single-request prefill chunks within the sort gate (short
            # prompts, short final chunks) sort through a logical plan.
            prefill_plan_sizes.add(b12x_topk_sort.MAX_TOKENS)
        self._prefill_plan_sizes = sorted(prefill_plan_sizes)
        self._prefill_plans = {
            rows: make_plan(mode="prefill", q_rows=rows)
            for rows in self._prefill_plan_sizes
        }
        self.dcp_world_size = getattr(
            k_cache, "dcp_shard_count", self.attention_dcp_world_size
        )
        self.dcp_rank = (
            get_dcp_group().rank_in_group % self.dcp_world_size
            if self.dcp_world_size > 1
            else 0
        )
        self._indexer_shard_group = None
        if 1 < self.dcp_world_size < self.attention_dcp_world_size:
            from vllm.distributed.parallel_state import get_indexer_dcp_group

            self._indexer_shard_group = get_indexer_dcp_group(self.dcp_world_size)
        self.cp_kv_cache_interleave_size = parallel_config.cp_kv_cache_interleave_size
        self._prefill_query_group = None
        self._prefill_shard_group = None
        self._prefill_owner_merge = False
        self._prefill_min_context = 0
        if self.attention_dcp_world_size > 1 and envs.VLLM_DCP_QUERY_SPLIT:
            from vllm.distributed.parallel_state import (
                get_indexer_dcp_group,
                get_indexer_query_split_group,
            )

            target_layers = getattr(
                vllm_config.model_config.hf_config, "num_hidden_layers", 0
            )
            if _is_target_indexer(k_cache.prefix, target_layers):
                self._prefill_query_group = get_indexer_query_split_group(
                    self.dcp_world_size
                )
                self._prefill_shard_group = get_indexer_dcp_group(self.dcp_world_size)
                self._prefill_owner_merge = envs.VLLM_DCP_TOPK_OWNER_MERGE
                self._prefill_min_context = envs.VLLM_DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS
                if self._prefill_min_context < 0:
                    raise ValueError("Query split minimum context must be nonnegative")
                logger.info_once(
                    "DCP indexer prefill: shards=%d replicas=%d "
                    "owner_merge=%s min_context=%d",
                    self.dcp_world_size,
                    self._prefill_query_group.world_size,
                    self._prefill_owner_merge,
                    self._prefill_min_context,
                )
        object.__setattr__(self, "b12x_warmup_provider", self)

    def _get_plan(self, mode: str, q_rows: int) -> Any:
        q_rows = int(q_rows)
        if mode == "decode":
            index = bisect.bisect_left(self._decode_plan_sizes, q_rows)
            if index < len(self._decode_plan_sizes):
                return self._decode_plans[self._decode_plan_sizes[index]]
            plans = self._decode_plans
            plan_sizes = self._decode_plan_sizes
        else:
            index = bisect.bisect_left(self._prefill_plan_sizes, q_rows)
            if index < len(self._prefill_plan_sizes):
                return self._prefill_plans[self._prefill_plan_sizes[index]]
            plans = self._prefill_plans
            plan_sizes = self._prefill_plan_sizes
        if _is_current_stream_capturing(self.topk_indices_buffer):
            raise RuntimeError(
                f"B12X DSA {mode} plan for {q_rows} rows was not prepared before "
                "CUDA graph capture."
            )
        plan = self._make_plan(mode=mode, q_rows=q_rows)
        plans[q_rows] = plan
        bisect.insort(plan_sizes, q_rows)
        return plan

    def _sorts(self, plan: Any) -> bool:
        """Whether ``plan`` emits logical positions that the sort converts."""
        return self.sort_selection and not _plan_emits_physical_slots(plan)

    def _reserve_profile_workspace(self) -> None:
        manager = current_workspace_manager()
        for plan in (*self._decode_plans.values(), *self._prefill_plans.values()):
            specs = plan.shapes_and_dtypes()
            if self.dcp_world_size > 1:
                specs = (
                    ((int(plan.caps.max_q_rows), self.topk_tokens), torch.float32),
                    *specs,
                )
                manager.reserve_all(*specs)
            elif getattr(self, "_prefill_query_group", None) is not None:
                manager.reserve_all(*specs)
            else:
                manager.get_simultaneous(*specs)
        if getattr(self, "_prefill_query_group", None) is not None:
            manager.reserve_all((tuple(self.topk_indices_buffer.shape), torch.int32))
        if self.dcp_world_size > 1:
            rows = int(self.topk_indices_buffer.shape[0])
            manager.reserve_all(
                *_dcp_merge_shapes(rows, self.topk_tokens, self.dcp_world_size)
            )
            if (
                getattr(self, "_prefill_query_group", None) is not None
                and self._prefill_owner_merge
            ):
                manager.reserve_all(
                    *_prefill_owner_shapes(rows, self.topk_tokens, self.dcp_world_size)
                )
            score_bytes = rows * self.topk_tokens * 4
            logger.info_once(
                "DCP indexer workspace budget: rows=%d, scores=%d bytes, "
                "packed_candidates=%d bytes, rank_major_gather=%d bytes; "
                "shared with plan scratch, reserved in every execution slot",
                rows,
                score_bytes,
                2 * score_bytes,
                2 * self.dcp_world_size * score_bytes,
            )

    def get_b12x_warmup_unit(
        self,
        layer: torch.nn.Module,
        token_counts: tuple[int, ...],
        output_dtype: torch.dtype,
    ) -> B12xWarmupUnit:
        del layer, token_counts, output_dtype

        def compile() -> None:
            if self.dcp_world_size > 1 or self._prefill_query_group is not None:
                self._reserve_profile_workspace()
            if self.dcp_world_size > 1:
                from b12x.comm.pcie.dcp_candidate_topk import precompile_rank_major_topk

                precompile_rank_major_topk(
                    self.topk_tokens,
                    self.dcp_world_size,
                    self.topk_indices_buffer.device,
                )
            kv_cache = self.k_cache.kv_cache
            if kv_cache.numel() == 0:
                # Warmup before the index KV cache is allocated (the
                # memory-profiling pass, which precedes the graph-memory
                # profiler's capture loop). Compile against a placeholder
                # cache with the production page layout: the compiled variants
                # depend on the page layout, not the page count, so the
                # post-allocation warmup and serving reuse them.
                kv_cache = torch.zeros(
                    (2, _INDEX_PAGE_SIZE, _INDEX_HEAD_DIM + _INDEX_SCALE_BYTES),
                    dtype=torch.uint8,
                    device=self.topk_indices_buffer.device,
                )
            plans = (*self._decode_plans.values(), *self._prefill_plans.values())
            for plan in plans:
                caps = plan.caps
                rows = int(caps.max_q_rows)
                mode = str(caps.mode)
                q = torch.zeros(
                    (rows, int(caps.num_q_heads), _INDEX_HEAD_DIM),
                    dtype=torch.float8_e4m3fn,
                    device=caps.device,
                )
                weights = torch.zeros(
                    (rows, int(caps.num_q_heads)),
                    dtype=torch.float32,
                    device=caps.device,
                )
                cache_lengths = torch.full(
                    (rows,),
                    min(self.max_model_len, _INDEX_PAGE_SIZE),
                    dtype=torch.int32,
                    device=caps.device,
                )
                page_rows = rows if mode == "decode" else 1
                page_table = torch.zeros(
                    (page_rows, int(caps.max_page_table_width)),
                    dtype=torch.int32,
                    device=caps.device,
                )
                if mode == "prefill":
                    page_table = page_table.expand(rows, -1)
                output = torch.empty(
                    (rows, self.topk_tokens),
                    dtype=torch.int32,
                    device=caps.device,
                )
                scores = _run_paged_topk(
                    module=self._module,
                    plan=plan,
                    q=q,
                    weights=weights,
                    kv_cache=kv_cache,
                    seq_lens=cache_lengths,
                    block_table=page_table,
                    active_width=self.active_width_cap,
                    output=output,
                    return_scores=self.dcp_world_size > 1,
                )
                if scores is not None:
                    _merge_dcp_topk(
                        output,
                        scores,
                        self.dcp_rank,
                        self.dcp_world_size,
                        self.cp_kv_cache_interleave_size,
                        shard_group=self._indexer_shard_group,
                    )
                if self._sorts(plan):
                    b12x_topk_sort.precompile(self.max_model_len, caps.device)
                    b12x_topk_sort.sort_convert(
                        output,
                        cache_lengths,
                        page_table,
                        _INDEX_PAGE_SIZE,
                        self.max_model_len,
                    )

        plan_key = tuple(
            (
                str(plan.caps.mode),
                int(plan.caps.max_q_rows),
                int(plan.caps.max_page_table_width),
                getattr(plan.layout, "route", None),
            )
            for plan in (*self._decode_plans.values(), *self._prefill_plans.values())
        )
        return B12xWarmupUnit(
            name="DSA indexer",
            key=(
                type(self),
                self.topk_indices_buffer.device,
                self.topk_tokens,
                self.dcp_world_size,
                self.cp_kv_cache_interleave_size,
                self.output_physical_slots,
                self.sort_selection,
                plan_key,
            ),
            compile=compile,
        )

    def _local_context_eligible(
        self,
        chunk: DeepseekV32IndexerPrefillChunkMetadata,
        context_cache: torch.Tensor | None,
        metadata: DeepseekV32IndexerMetadata,
    ) -> bool:
        """Whether ``chunk`` scores against the step-local key copy.

        The route replaces the sharded scoring and cross-rank candidate merge
        for a request whose whole context is among the step's tokens (see
        ``DeepseekV32IndexerMetadataBuilder._assign_local_context``). Decode
        rows and captured streams keep the sharded path: the copy is written
        by the eager prefill kernel of the same step.
        """
        return (
            self.dcp_world_size > 1
            and context_cache is not None
            and getattr(chunk, "context_base_page", -1) >= 0
            and chunk.context_seq_lens is not None
            and chunk.context_block_table is not None
            and metadata.decode is None
            and not _is_current_stream_capturing(self.topk_indices_buffer)
        )

    def _run_local_context_chunk(
        self,
        chunk: DeepseekV32IndexerPrefillChunkMetadata,
        context_cache: torch.Tensor,
        q_quant: torch.Tensor,
        weights: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Score every row of ``chunk`` against the step-local key copy.

        Rows keep their request-relative causal bounds and the selection is
        emitted as logical positions, which equal global token ids because
        the request has no cached history. No candidate is exchanged.
        """
        start, end = chunk.token_start, chunk.token_end
        rows = end - start
        assert chunk.context_seq_lens is not None
        assert chunk.context_block_table is not None
        seq_lens = chunk.context_seq_lens
        pages = int(chunk.context_block_table.shape[1])
        block_table = chunk.context_block_table.expand(rows, pages)
        plan = self._get_plan("prefill", rows)
        _run_paged_topk(
            module=self._module,
            plan=plan,
            q=q_quant[start:end].contiguous(),
            weights=weights[start:end].contiguous(),
            kv_cache=context_cache,
            seq_lens=seq_lens,
            block_table=block_table,
            active_width=self.active_width_cap,
            output=output,
            return_scores=False,
        )
        if self._sorts(plan):
            b12x_topk_sort.sort_convert(
                output,
                seq_lens,
                block_table,
                _INDEX_PAGE_SIZE,
                self.max_model_len,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor | None,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        del hidden_states
        if not isinstance(q_quant, torch.Tensor):
            raise ValueError("B12X indexing requires FP8 index queries.")
        if k is not None:
            raise ValueError("B12X index K must be written by the fused cache path.")

        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if not isinstance(attn_metadata, dict):
            if (
                forward_context.cudagraph_runtime_mode == CUDAGraphMode.NONE
                and forward_context.batch_descriptor is not None
            ):
                self._reserve_profile_workspace()
            return self.topk_indices_buffer

        metadata = cast(DeepseekV32IndexerMetadata, attn_metadata[self.k_cache.prefix])

        if metadata.prefill is not None:
            context_cache = getattr(metadata.prefill, "context_cache", None)
            for chunk in metadata.prefill.chunks:
                if chunk.num_reqs != 1:
                    raise RuntimeError(
                        "B12X sparse prefill requires single-request chunks."
                    )
                start, end = chunk.token_start, chunk.token_end
                full_output = self.topk_indices_buffer[start:end, : self.topk_tokens]
                if self._local_context_eligible(chunk, context_cache, metadata):
                    self._run_local_context_chunk(
                        chunk, context_cache, q_quant, weights, full_output
                    )
                    continue
                query_group = getattr(self, "_prefill_query_group", None)
                split = (
                    getattr(self, "attention_dcp_world_size", self.dcp_world_size) > 1
                    and query_group is not None
                    and query_group.world_size > 1
                    and metadata.decode is None
                    and forward_context.cudagraph_runtime_mode == CUDAGraphMode.NONE
                    and not _is_current_stream_capturing(q_quant)
                    and chunk.total_seq_lens > 0
                    and chunk.total_seq_lens >= self._prefill_min_context
                    and (end - start) % query_group.world_size == 0
                )
                relative_start, relative_end = 0, end - start
                if split:
                    assert query_group is not None
                    rows = (end - start) // query_group.world_size
                    relative_start = query_group.rank_in_group * rows
                    relative_end = relative_start + rows
                    start, end = start + relative_start, start + relative_end
                q_chunk = q_quant[start:end].contiguous()
                weights_chunk = weights[start:end].contiguous()
                output = self.topk_indices_buffer[start:end, : self.topk_tokens]
                seq_lens = (
                    chunk.cu_seqlen_ke[relative_start:relative_end]
                    - chunk.cu_seqlen_ks[relative_start:relative_end]
                ).contiguous()
                local_rows = (
                    chunk.local_total_seq_lens
                    if self.dcp_world_size > 1
                    else chunk.total_seq_lens
                )
                active_pages = max(
                    1, (int(local_rows) + _INDEX_PAGE_SIZE - 1) // _INDEX_PAGE_SIZE
                )
                active_pages = min(active_pages, int(chunk.block_table.shape[1]))
                block_table = chunk.block_table[:1, :active_pages].expand(
                    int(q_chunk.shape[0]), active_pages
                )
                plan = self._get_plan("prefill", int(q_chunk.shape[0]))
                score_chunk = _run_paged_topk(
                    module=self._module,
                    plan=plan,
                    q=q_chunk,
                    weights=weights_chunk,
                    kv_cache=self.k_cache.kv_cache,
                    seq_lens=seq_lens,
                    block_table=block_table,
                    active_width=self.active_width_cap,
                    output=output,
                    return_scores=self.dcp_world_size > 1,
                )
                if self._sorts(plan):
                    # In line: prefill runs eagerly, and the selection is
                    # read by the attention op that follows.
                    b12x_topk_sort.sort_convert(
                        output,
                        seq_lens,
                        block_table,
                        _INDEX_PAGE_SIZE,
                        self.max_model_len,
                    )
                if score_chunk is not None:
                    if split and self._prefill_owner_merge:
                        from vllm.distributed.parallel_state import get_tp_group

                        if _merge_prefill_topk_by_owner(
                            output,
                            score_chunk,
                            full_output,
                            self._prefill_shard_group,
                            get_tp_group(),
                            self.cp_kv_cache_interleave_size,
                        ):
                            continue
                    _merge_dcp_topk(
                        output,
                        score_chunk,
                        self.dcp_rank,
                        self.dcp_world_size,
                        self.cp_kv_cache_interleave_size,
                        shard_group=self._indexer_shard_group,
                    )
                if split:
                    _restore_prefill_indices(query_group, output, full_output)

        if metadata.decode is not None:
            decode = metadata.decode
            if decode.requires_padding:
                raise RuntimeError("B12X sparse decode does not support padded rows.")
            seq_lens = decode.seq_lens.reshape(-1).contiguous()
            block_table = decode.block_table
            if int(block_table.shape[0]) != int(seq_lens.shape[0]):
                if int(seq_lens.shape[0]) % int(block_table.shape[0]) != 0:
                    raise RuntimeError(
                        "B12X sparse decode could not align lengths and page tables."
                    )
                block_table = block_table.repeat_interleave(
                    int(seq_lens.shape[0]) // int(block_table.shape[0]), dim=0
                )
            num_tokens = metadata.num_decode_tokens
            output = self.topk_indices_buffer[:num_tokens, : self.topk_tokens]
            plan = self._get_plan("decode", num_tokens)
            decode_seq_lens = seq_lens[:num_tokens]
            decode_block_table = block_table[:num_tokens].contiguous()
            score_slice = _run_paged_topk(
                module=self._module,
                plan=plan,
                q=q_quant[:num_tokens].contiguous(),
                weights=weights[:num_tokens].contiguous(),
                kv_cache=self.k_cache.kv_cache,
                seq_lens=decode_seq_lens,
                block_table=decode_block_table,
                active_width=getattr(decode, "active_width", None),
                output=output,
                return_scores=self.dcp_world_size > 1,
            )
            if self._sorts(plan):
                # Side stream inside full graphs; joined in the sparse MLA
                # backend's forward_mqa before the selection is read.
                b12x_topk_sort.sort_convert_async(
                    output,
                    decode_seq_lens,
                    decode_block_table,
                    _INDEX_PAGE_SIZE,
                    self.max_model_len,
                )
            if score_slice is not None:
                _merge_dcp_topk(
                    output,
                    score_slice,
                    self.dcp_rank,
                    self.dcp_world_size,
                    self.cp_kv_cache_interleave_size,
                    shard_group=self._indexer_shard_group,
                )

        return self.topk_indices_buffer
