# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for the local-context indexer route under DCP.

A prefill request whose whole context is among the step's tokens is scored
against a step-local copy of its index keys on every rank, without a
cross-rank candidate merge. These tests cover the metadata assignment, the
indexer dispatch and the fused producer's argument handling with mocked
kernels; GPU equivalence against the sharded route is validated separately.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import CUDAGraphMode
from vllm.models.deepseek_v32.common import kernels
from vllm.v1.attention.backends.mla import b12x_indexer as indexer
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadataBuilder,
    DeepseekV32IndexerPrefillChunkMetadata,
    DeepseekV32IndexerPrefillMetadata,
)

PAGE = 64
RECORD = 132


def make_builder(max_tokens=256, max_seqs=4):
    builder = DeepseekV32IndexerMetadataBuilder.__new__(
        DeepseekV32IndexerMetadataBuilder
    )
    pages = -(-max_tokens // PAGE) + max_seqs
    builder.context_cache = torch.zeros((pages, PAGE, RECORD), dtype=torch.uint8)
    builder.context_slot_mapping_buffer = torch.full(
        (max_tokens,), -1, dtype=torch.int32
    )
    builder.context_arange_buffer = torch.arange(
        max(max_tokens, pages) + 1, dtype=torch.int32
    )
    return builder


def chunk(token_start, token_end, total_seq_lens, num_reqs=1):
    return DeepseekV32IndexerPrefillChunkMetadata(
        block_table=torch.zeros((num_reqs, 1), dtype=torch.int32),
        cu_seqlen_ks=torch.zeros(token_end - token_start, dtype=torch.int32),
        cu_seqlen_ke=torch.zeros(token_end - token_start, dtype=torch.int32),
        cu_seq_lens=torch.zeros(num_reqs + 1, dtype=torch.int32),
        token_to_seq=torch.zeros(1, dtype=torch.int32),
        total_seq_lens=total_seq_lens,
        token_start=token_start,
        token_end=token_end,
        num_reqs=num_reqs,
    )


def test_assign_marks_fresh_requests_and_keeps_history_chunks_sharded():
    builder = make_builder()
    # Request 0: 100 fresh tokens; request 1: 20 tokens over 500 of context;
    # request 2: 70 fresh tokens split into two query slices.
    starts = torch.tensor([0, 100, 120, 190], dtype=torch.int32)
    seq_lens = torch.tensor([100, 500, 70], dtype=torch.int32)
    chunks = [
        chunk(0, 100, 100),
        chunk(100, 120, 500),
        chunk(120, 160, 70),
        chunk(160, 190, 70),
    ]
    prefill = DeepseekV32IndexerPrefillMetadata(chunks)
    builder._assign_local_context(prefill, starts, seq_lens, 190)

    assert prefill.context_cache is builder.context_cache
    # The producer runs over the step's padded rows: the mapping keeps the
    # buffer's full length and marks every row beyond the step's tokens.
    assert prefill.context_slot_mapping is builder.context_slot_mapping_buffer
    slots = prefill.context_slot_mapping.tolist()
    assert len(slots) == 256
    assert slots[:100] == list(range(100))
    assert slots[100:120] == [-1] * 20
    assert slots[120:190] == list(range(2 * PAGE, 2 * PAGE + 70))
    assert slots[190:] == [-1] * 66

    first, history, part_a, part_b = chunks
    assert first.context_base_page == 0
    assert first.context_query_offset == 0
    assert first.context_seq_lens.tolist() == list(range(1, 101))
    assert first.context_block_table.tolist() == [[0, 1]]
    assert history.context_base_page == -1
    assert history.context_seq_lens is None
    assert part_a.context_base_page == 2 and part_b.context_base_page == 2
    assert part_a.context_query_offset == 0 and part_b.context_query_offset == 40
    assert part_a.context_seq_lens.tolist() == list(range(1, 41))
    assert part_b.context_seq_lens.tolist() == list(range(41, 71))
    assert part_b.context_block_table.tolist() == [[2, 3]]


def test_assign_clears_stale_slots_beyond_the_step():
    builder = make_builder()
    builder.context_slot_mapping_buffer.fill_(5)
    prefill = DeepseekV32IndexerPrefillMetadata([chunk(0, 8, 8)])
    builder._assign_local_context(
        prefill, torch.tensor([0, 8], dtype=torch.int32), torch.tensor([8]), 8
    )
    slots = prefill.context_slot_mapping.tolist()
    assert slots[:8] == list(range(8))
    assert slots[8:] == [-1] * 248


def test_assign_without_fresh_requests_leaves_prefill_metadata_untouched():
    builder = make_builder()
    prefill = DeepseekV32IndexerPrefillMetadata([chunk(0, 8, 300)])
    builder._assign_local_context(
        prefill, torch.tensor([0, 8], dtype=torch.int32), torch.tensor([300]), 8
    )
    assert prefill.context_cache is None
    assert prefill.context_slot_mapping is None
    assert prefill.chunks[0].context_base_page == -1


def test_assign_respects_copy_capacity_and_multi_request_chunks():
    # Three copy pages: 100 tokens take two, so the 92-token request that
    # would need two more stays on the sharded route.
    builder = make_builder(max_tokens=192, max_seqs=0)
    starts = torch.tensor([0, 100, 192], dtype=torch.int32)
    seq_lens = torch.tensor([100, 92], dtype=torch.int32)
    chunks = [chunk(0, 100, 100), chunk(100, 192, 92)]
    prefill = DeepseekV32IndexerPrefillMetadata(chunks)
    builder._assign_local_context(prefill, starts, seq_lens, 192)
    assert chunks[0].context_base_page == 0
    assert chunks[1].context_base_page == -1
    assert prefill.context_slot_mapping[100:].tolist() == [-1] * 92

    builder = make_builder()
    joint = DeepseekV32IndexerPrefillMetadata([chunk(0, 16, 16, num_reqs=2)])
    builder._assign_local_context(
        joint, torch.tensor([0, 8, 16], dtype=torch.int32), torch.tensor([8, 8]), 16
    )
    assert joint.context_cache is None


def make_indexer(monkeypatch, rows=8, base_page=0, decode=False, dcp=4):
    obj = indexer.B12xSparseIndexer.__new__(indexer.B12xSparseIndexer)
    torch.nn.Module.__init__(obj)
    obj.k_cache = SimpleNamespace(
        prefix="model.layers.0.indexer", kv_cache=torch.empty(1)
    )
    obj.topk_tokens = 2
    obj.topk_indices_buffer = torch.full((rows, 2), -1, dtype=torch.int32)
    obj.dcp_world_size, obj.dcp_rank = dcp, 1
    obj.attention_dcp_world_size = dcp
    obj._indexer_shard_group = None
    obj.cp_kv_cache_interleave_size = 1
    obj.active_width_cap = torch.tensor([100])
    obj.max_model_len = 100
    obj._module = None
    obj._prefill_query_group = SimpleNamespace(world_size=2, rank_in_group=1)
    obj._prefill_shard_group = SimpleNamespace(world_size=4, rank_in_group=1)
    obj._prefill_owner_merge = False
    obj._prefill_min_context = 0
    obj._plan = lambda *args: "plan"
    obj._sorts = lambda plan: False
    context_cache = torch.zeros((4, PAGE, RECORD), dtype=torch.uint8)
    chunk_meta = SimpleNamespace(
        num_reqs=1,
        token_start=0,
        token_end=rows,
        cu_seqlen_ks=torch.zeros(rows, dtype=torch.int32),
        cu_seqlen_ke=torch.arange(1, rows + 1, dtype=torch.int32),
        local_total_seq_lens=rows,
        total_seq_lens=rows,
        block_table=torch.arange(4).reshape(1, 4),
        context_base_page=base_page,
        context_query_offset=0,
        context_seq_lens=torch.arange(1, rows + 1, dtype=torch.int32),
        context_block_table=torch.arange(1, dtype=torch.int32).view(1, 1),
    )
    prefill = SimpleNamespace(
        chunks=[chunk_meta],
        context_cache=context_cache,
        context_slot_mapping=torch.arange(rows, dtype=torch.int32),
    )
    metadata = SimpleNamespace(
        prefill=prefill,
        decode=SimpleNamespace(requires_padding=True) if decode else None,
    )
    ctx = SimpleNamespace(
        attn_metadata={obj.k_cache.prefix: metadata},
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )
    monkeypatch.setattr(indexer, "get_forward_context", lambda: ctx)
    calls = []

    def score(**kw):
        calls.append(kw)
        kw["output"].copy_(kw["q"][:, :1].to(torch.int32).expand(-1, 2))
        return (
            None
            if not kw["return_scores"]
            else torch.zeros_like(kw["output"], dtype=torch.float32)
        )

    monkeypatch.setattr(indexer, "_run_paged_topk", score)
    merges = []
    monkeypatch.setattr(
        indexer, "_merge_dcp_topk", lambda *args, **kwargs: merges.append(args)
    )
    restores = []
    monkeypatch.setattr(
        indexer, "_restore_prefill_indices", lambda *args: restores.append(args)
    )
    q = torch.arange(rows, dtype=torch.float32).reshape(rows, 1)
    return obj, q, calls, merges, restores, context_cache


def test_local_context_chunk_scores_every_row_against_the_copy(monkeypatch):
    obj, q, calls, merges, restores, context_cache = make_indexer(monkeypatch)
    result = obj.forward(q, q, None, q + 100)
    (call,) = calls
    assert call["kv_cache"] is context_cache
    assert call["q"].flatten().tolist() == list(range(8))
    assert call["weights"].flatten().tolist() == list(range(100, 108))
    assert call["seq_lens"].tolist() == list(range(1, 9))
    assert call["block_table"].shape == (8, 1)
    assert call["return_scores"] is False
    assert call["output"].data_ptr() == result.data_ptr()
    assert result[:, 0].tolist() == list(range(8))
    assert not merges and not restores


@pytest.mark.parametrize("kwargs", [dict(base_page=-1), dict(decode=True), dict(dcp=1)])
def test_local_context_route_requires_eligibility(monkeypatch, kwargs):
    obj, q, calls, merges, restores, context_cache = make_indexer(monkeypatch, **kwargs)
    if kwargs.get("decode"):
        with pytest.raises(RuntimeError, match="padded rows"):
            obj.forward(q, q, None, q)
    else:
        obj.forward(q, q, None, q)
    assert calls[0]["kv_cache"] is not context_cache


def test_capturing_stream_keeps_the_sharded_route(monkeypatch):
    obj, q, calls, merges, restores, context_cache = make_indexer(monkeypatch)
    monkeypatch.setattr(indexer, "_is_current_stream_capturing", lambda tensor: True)
    obj.forward(q, q, None, q)
    assert calls[0]["kv_cache"] is not context_cache


class _FakeKernel:
    def __init__(self):
        self.launches = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.launches.append((grid, args, kwargs))

        return launch


def _producer_inputs(tokens=4):
    return dict(
        positions=torch.arange(tokens),
        q_c=torch.zeros(tokens, 8),
        q_rms_norm_w=torch.ones(8),
        q_rms_eps=1e-6,
        kv_c=torch.zeros(tokens, 4),
        kv_rms_norm_w=torch.ones(4),
        kv_rms_eps=1e-6,
        k_pe=torch.zeros(tokens, 2),
        k_rope_cos_sin_cache=torch.zeros(16, 2),
        index_k=torch.zeros(tokens, 8),
        index_k_layer_norm_w=torch.ones(8),
        index_k_layer_norm_bias=torch.zeros(8),
        index_k_layer_norm_eps=1e-6,
        index_k_rope_cos_sin_cache=torch.zeros(16, 2),
        topk_indices_buffer=torch.zeros(tokens, 2, dtype=torch.int32),
    )


def test_producer_passes_step_local_copy_geometry(monkeypatch):
    fake = _FakeKernel()
    monkeypatch.setattr(kernels, "_fused_norm_rope_kernel", fake)
    monkeypatch.setattr(
        kernels,
        "current_platform",
        SimpleNamespace(is_arch_support_pdl=lambda: False),
    )
    local_cache = torch.zeros((3, PAGE, RECORD), dtype=torch.uint8)
    local_slots = torch.tensor([0, 1, -1, 3], dtype=torch.int32)
    shard_cache = torch.zeros((2, PAGE, RECORD), dtype=torch.uint8)
    kernels.fused_norm_rope(
        **_producer_inputs(),
        indexer_k_cache=shard_cache,
        indexer_slot_mapping=torch.tensor([0, -1, -1, 5], dtype=torch.int32),
        indexer_local_cache=local_cache,
        indexer_local_slot_mapping=local_slots,
    )
    ((grid, args, kwargs),) = fake.launches
    assert grid == (4, 4)
    position = {id(a): i for i, a in enumerate(args) if isinstance(a, torch.Tensor)}
    slot_index = position[id(local_slots)]
    assert args[slot_index + 1].dtype == torch.float8_e4m3fn
    assert args[slot_index + 1].untyped_storage().data_ptr() == (
        local_cache.untyped_storage().data_ptr()
    )
    assert args[slot_index + 2].dtype == torch.float32
    assert args[slot_index + 3] == PAGE
    assert args[slot_index + 4] == PAGE * RECORD
    assert kwargs["HAS_INDEXER"] is True


def test_producer_drops_step_local_copy_on_shared_layers(monkeypatch):
    fake = _FakeKernel()
    monkeypatch.setattr(kernels, "_fused_norm_rope_kernel", fake)
    monkeypatch.setattr(
        kernels,
        "current_platform",
        SimpleNamespace(is_arch_support_pdl=lambda: False),
    )
    kernels.fused_norm_rope(
        **_producer_inputs(),
        has_indexer=False,
        indexer_local_cache=torch.zeros((1, PAGE, RECORD), dtype=torch.uint8),
        indexer_local_slot_mapping=torch.zeros(4, dtype=torch.int32),
    )
    ((grid, args, kwargs),) = fake.launches
    assert kwargs["HAS_INDEXER"] is False
    # Both cache blocks collapse to the disabled placeholders: slot mapping,
    # cache and scale views None, block size 1, block stride 0 — the shard
    # block (preceded by the attention slot mapping) and the local block.
    shape = ["T" if isinstance(a, torch.Tensor) else a for a in args]
    disabled = [None, None, None, None, 1, 0, None, None, None, 1, 0]
    assert any(
        shape[i : i + len(disabled)] == disabled
        for i in range(len(shape) - len(disabled) + 1)
    )


@pytest.mark.parametrize("missing", ["cache", "slots"])
def test_producer_rejects_a_partial_step_local_copy(monkeypatch, missing):
    monkeypatch.setattr(kernels, "_fused_norm_rope_kernel", _FakeKernel())
    extra = dict(
        indexer_local_cache=None
        if missing == "cache"
        else torch.zeros((1, PAGE, RECORD), dtype=torch.uint8),
        indexer_local_slot_mapping=None
        if missing == "slots"
        else torch.zeros(4, dtype=torch.int32),
    )
    with pytest.raises(ValueError, match="step-local index K copy"):
        kernels.fused_norm_rope(**_producer_inputs(), **extra)
