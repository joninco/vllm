# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU dispatch checks for prefill partitioning and exact owner transport.

Mocked scoring/collectives expose query, length and candidate ordering without
CUDA. GPU selector equivalence is a separate required validation.
"""

import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from vllm.config import CUDAGraphMode
from vllm.v1.attention.backends.mla import b12x_indexer as indexer


def make_indexer(
    monkeypatch,
    rows=8,
    split=True,
    owner=False,
    mixed=False,
    mode=CUDAGraphMode.NONE,
    context=100,
    threshold=0,
):
    obj = indexer.B12xSparseIndexer.__new__(indexer.B12xSparseIndexer)
    torch.nn.Module.__init__(obj)
    obj.k_cache = SimpleNamespace(
        prefix="model.layers.0.indexer", kv_cache=torch.empty(1)
    )
    obj.topk_tokens = 2
    obj.topk_indices_buffer = torch.full((rows, 2), -1, dtype=torch.int32)
    obj.dcp_world_size, obj.dcp_rank = 4, 1
    obj._indexer_shard_group = None
    obj.cp_kv_cache_interleave_size = 1
    obj.active_width_cap = torch.tensor([100])
    obj._module = None
    obj._prefill_query_group = (
        SimpleNamespace(world_size=2, rank_in_group=1) if split else None
    )
    obj._prefill_shard_group = SimpleNamespace(world_size=4, rank_in_group=1)
    obj._prefill_owner_merge = owner
    obj._prefill_min_context = threshold
    obj._plan = lambda *args: None
    obj._sorts = lambda plan: False
    chunk = SimpleNamespace(
        num_reqs=1,
        token_start=0,
        token_end=rows,
        cu_seqlen_ks=torch.arange(rows, dtype=torch.int32),
        cu_seqlen_ke=2 * torch.arange(rows, dtype=torch.int32) + 1,
        local_total_seq_lens=100,
        total_seq_lens=context,
        block_table=torch.arange(4).reshape(1, 4),
    )
    decode = SimpleNamespace(requires_padding=True) if mixed else None
    metadata = SimpleNamespace(prefill=SimpleNamespace(chunks=[chunk]), decode=decode)
    ctx = SimpleNamespace(
        attn_metadata={obj.k_cache.prefix: metadata}, cudagraph_runtime_mode=mode
    )
    monkeypatch.setattr(indexer, "get_forward_context", lambda: ctx)
    calls = []

    def score(**kw):
        calls.append(kw)
        kw["output"].copy_(kw["q"][:, :1].to(torch.int32).expand(-1, 2))
        return torch.zeros_like(kw["output"], dtype=torch.float32)

    monkeypatch.setattr(indexer, "_run_paged_topk", score)
    monkeypatch.setattr(indexer, "_merge_dcp_topk", lambda *args, **kwargs: None)
    restores = []
    monkeypatch.setattr(
        indexer, "_restore_prefill_indices", lambda *args: restores.append(args)
    )
    q = torch.arange(rows, dtype=torch.float32).reshape(rows, 1)
    return obj, q, calls, restores


def test_prefill_split_narrows_queries_weights_causal_lengths_and_output(monkeypatch):
    obj, q, calls, restores = make_indexer(monkeypatch)
    result = obj.forward(q, q, None, q + 100)
    call = calls[0]
    assert call["q"].flatten().tolist() == [4, 5, 6, 7]
    assert call["weights"].flatten().tolist() == [104, 105, 106, 107]
    assert call["seq_lens"].tolist() == [5, 6, 7, 8]
    assert call["output"].data_ptr() == result[4:].data_ptr()
    assert restores[0][1].shape == (4, 2)
    assert restores[0][2].shape == (8, 2)
    obj.forward(q, q + 10, None, q + 200)
    assert calls[1]["q"].flatten().tolist() == [14, 15, 16, 17]


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(rows=7),
        dict(context=0),
        dict(threshold=101),
        dict(split=False),
        dict(mode=CUDAGraphMode.FULL),
    ],
)
def test_prefill_ineligible_batches_keep_all_rows(monkeypatch, kwargs):
    obj, q, calls, restores = make_indexer(monkeypatch, **kwargs)
    obj.forward(q, q, None, q)
    assert calls[0]["q"].shape == q.shape
    assert not restores


def test_mixed_batch_does_not_partition_prefill(monkeypatch):
    obj, q, calls, restores = make_indexer(monkeypatch, mixed=True)
    with pytest.raises(RuntimeError, match="padded rows"):
        obj.forward(q, q, None, q)
    assert calls[0]["q"].shape == q.shape
    assert not restores


def test_dcp1_ignores_partition_state(monkeypatch):
    obj, q, calls, restores = make_indexer(monkeypatch)
    obj.dcp_world_size = 1
    obj.forward(q, q, None, q)
    assert calls[0]["q"].shape == q.shape
    assert not restores


def test_stream_capture_disables_partition_even_with_eager_context(monkeypatch):
    obj, q, calls, restores = make_indexer(monkeypatch)
    monkeypatch.setattr(indexer, "_is_current_stream_capturing", lambda tensor: True)
    obj.forward(q, q, None, q)
    assert calls[0]["q"].shape == q.shape
    assert not restores


def test_multiple_request_chunks_keep_absolute_output_and_relative_causal_rows(
    monkeypatch,
):
    obj, q, calls, restores = make_indexer(monkeypatch)
    metadata = indexer.get_forward_context().attn_metadata[obj.k_cache.prefix]
    first = metadata.prefill.chunks[0]
    first.token_end = 4
    first.cu_seqlen_ks = torch.arange(4, dtype=torch.int32)
    first.cu_seqlen_ke = first.cu_seqlen_ks + torch.arange(1, 5)
    second = SimpleNamespace(**vars(first))
    second.token_start, second.token_end = 4, 8
    second.cu_seqlen_ke = second.cu_seqlen_ks + torch.arange(11, 15)
    metadata.prefill.chunks = [first, second]
    obj.forward(q, q, None, q)
    assert [call["q"].flatten().tolist() for call in calls] == [[2, 3], [6, 7]]
    assert [call["seq_lens"].tolist() for call in calls] == [[3, 4], [13, 14]]
    assert restores[1][2].data_ptr() == obj.topk_indices_buffer[4:].data_ptr()


@pytest.mark.parametrize(
    "prefix,expected",
    [
        ("model.layers.0.indexer", True),
        ("model.layers.77.indexer", True),
        ("model.layers.78.indexer", False),
        ("model.layers.79.indexer", False),
        ("indexer", False),
        ("layers.0.draft.1", False),
    ],
)
def test_drafter_and_unknown_layer_identity_do_not_enable_prefill_split(
    prefix, expected
):
    assert indexer._is_target_indexer(prefix, 78) is expected


@pytest.mark.parametrize("eligible", [False, True])
def test_forward_owner_merge_skips_duplicate_merge_and_restore(monkeypatch, eligible):
    obj, q, calls, restores = make_indexer(monkeypatch, owner=True)
    from vllm.distributed import parallel_state

    monkeypatch.setattr(parallel_state, "get_tp_group", lambda: object())
    owners, replicated = [], []

    def owner(*args):
        owners.append(args)
        return eligible

    monkeypatch.setattr(indexer, "_merge_prefill_topk_by_owner", owner)
    monkeypatch.setattr(
        indexer, "_merge_dcp_topk", lambda *args, **kwargs: replicated.append(args)
    )
    obj.forward(q, q, None, q)
    assert len(owners) == 1
    assert len(replicated) == int(not eligible)
    assert len(restores) == int(not eligible)


def test_profile_reserves_owner_and_restore_scratch_before_execution(monkeypatch):
    obj, _, _, _ = make_indexer(monkeypatch, owner=True)
    obj._prepared_plans = {}
    reserved = []
    manager = SimpleNamespace(reserve_all=lambda *specs: reserved.append(specs))
    monkeypatch.setattr(indexer, "current_workspace_manager", lambda: manager)
    obj._reserve_profile_workspace()
    assert indexer._prefill_owner_shapes(8, 2, 4) in reserved
    assert (((8, 2), torch.int32),) in reserved


class Workspace:
    def __init__(self, scores=None):
        self.scores = scores

    def get_simultaneous(self, *specs):
        result = [torch.empty(shape, dtype=dtype) for shape, dtype in specs]
        if self.scores is not None:
            result[0] = self.scores
        return result


def test_indices_restore_uses_nonaliasing_int32_input(monkeypatch):
    output = torch.arange(16, dtype=torch.int32).reshape(8, 2)
    local = output[4:]
    group = SimpleNamespace(world_size=2, rank_in_group=1)
    monkeypatch.setattr(indexer, "current_workspace_manager", lambda: Workspace())

    def gather(group_, send, gathered):
        assert send.dtype == torch.int32
        assert send.data_ptr() != local.data_ptr()
        assert torch.equal(send, local)
        assert gathered.data_ptr() == output.data_ptr()

    monkeypatch.setattr(indexer, "_gather_dcp_candidates", gather)
    indexer._restore_prefill_indices(group, local, output)
    with pytest.raises(RuntimeError, match="alias"):
        indexer._restore_prefill_indices(group, local.clone(), output)


def test_owner_transport_retains_rank_major_scores_ids_and_partition_order(monkeypatch):
    output = torch.full((8, 2), -1, dtype=torch.int32)
    indices = output[:4]
    indices.copy_(torch.arange(8).reshape(4, 2))
    scores = torch.tensor([[0.0, -0.0], [3.0, 3.0], [2.0, 1.0], [0.0, -1.0]])
    shard = SimpleNamespace(world_size=2, rank_in_group=1, device_group="shard")
    tp = SimpleNamespace(world_size=4, rank_in_group=1)
    monkeypatch.setattr(indexer, "current_workspace_manager", lambda: Workspace(scores))
    seen = []

    def pack(ids, values, packed, rank, size, interleave):
        packed[..., 0].copy_(values)
        packed[..., 1].copy_(ids)
        seen.append(values.view(torch.int32).clone())

    def exchange(received, packed, group):
        assert group == "shard"
        received.copy_(packed)

    def select(received, selected):
        assert received.shape == (2, 2, 2, 2)
        assert torch.equal(received[..., 0].reshape(4, 2).view(torch.int32), seen[0])
        selected.copy_(torch.tensor([[11, 12], [21, 22]]))

    def gather(group, selected, gathered):
        assert group is tp
        for rank in range(4):
            gathered[rank].copy_(selected + rank * 100)

    monkeypatch.setitem(
        sys.modules,
        "b12x.comm.pcie.dcp_candidate_topk",
        SimpleNamespace(pack_dcp_candidates=pack, rank_major_topk=select),
    )
    monkeypatch.setattr(indexer.dist, "all_to_all_single", exchange)
    monkeypatch.setattr(indexer, "_gather_dcp_candidates", gather)
    assert indexer._merge_prefill_topk_by_owner(indices, scores, output, shard, tp, 1)
    assert output[:, 0].tolist() == [11, 21, 111, 121, 211, 221, 311, 321]
    assert not indexer._merge_prefill_topk_by_owner(
        indices[:3], scores[:3], output, shard, tp, 1
    )


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("fail_send", [False, True])
def test_owner_pynccl_preserves_peer_slices_stream_and_group_closure(
    monkeypatch, rank, fail_send
):
    packed = torch.arange(32, dtype=torch.float32).reshape(8, 2, 2)
    received = torch.full_like(packed, -1)
    stream = object()
    events: list[object] = []
    active = False

    @contextmanager
    def use_stream(value):
        nonlocal active
        assert value is stream
        active = True
        try:
            yield
        finally:
            active = False

    def start():
        assert active
        # The self partition is copied on the explicit stream before grouping.
        assert torch.equal(
            received[rank * 2 : rank * 2 + 2], packed[rank * 2 : rank * 2 + 2]
        )
        events.append("start")

    def send(tensor, peer, used_stream):
        assert active and used_stream is stream
        assert tensor.data_ptr() == packed[peer * 2].data_ptr()
        assert tensor.shape == (2, 2, 2)
        events.append(("send", peer))
        if fail_send:
            raise RuntimeError("send failure")

    def recv(tensor, peer, used_stream):
        assert active and used_stream is stream
        assert tensor.data_ptr() == received[peer * 2].data_ptr()
        tensor.fill_(peer + 100)
        events.append(("recv", peer))

    def end():
        assert active
        events.append("end")

    pynccl = SimpleNamespace(
        disabled=False, group_start=start, group_end=end, send=send, recv=recv
    )
    group = SimpleNamespace(
        world_size=4,
        rank_in_group=rank,
        device_communicator=SimpleNamespace(pynccl_comm=pynccl),
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: stream)
    monkeypatch.setattr(torch.cuda, "stream", use_stream)
    if fail_send:
        with pytest.raises(RuntimeError, match="send failure"):
            indexer._exchange_prefill_owner_candidates(group, packed, received)
        assert events == [
            "start",
            ("send", next(p for p in range(4) if p != rank)),
            "end",
        ]
    else:
        indexer._exchange_prefill_owner_candidates(group, packed, received)
        expected = [
            op for p in range(4) if p != rank for op in [("send", p), ("recv", p)]
        ]
        assert events == ["start", *expected, "end"]
        for peer in range(4):
            if peer != rank:
                assert torch.all(received[peer * 2 : peer * 2 + 2] == peer + 100)
    assert not active
    with pytest.raises(ValueError, match="overlap"):
        indexer._exchange_prefill_owner_candidates(group, packed, packed)


@pytest.mark.parametrize("pynccl", [None, SimpleNamespace(disabled=True)])
def test_owner_exchange_falls_back_when_pynccl_unavailable(monkeypatch, pynccl):
    packed = torch.arange(16, dtype=torch.float32).reshape(4, 2, 2)
    received = torch.empty_like(packed)
    group = SimpleNamespace(
        device_group=object(), device_communicator=SimpleNamespace(pynccl_comm=pynccl)
    )
    calls = []

    def exchange(output, source, group):
        calls.append(group)
        assert output is received and source is packed
        output.copy_(source)

    monkeypatch.setattr(indexer.dist, "all_to_all_single", exchange)
    indexer._exchange_prefill_owner_candidates(group, packed, received)
    assert calls == [group.device_group]
    assert torch.equal(received, packed)


def test_single_shard_indexer_splits_queries_across_attention_replicas(monkeypatch):
    obj, q, calls, restores = make_indexer(monkeypatch, rows=8)
    obj.attention_dcp_world_size = 4
    obj.dcp_world_size, obj.dcp_rank = 1, 0
    obj._prefill_query_group = SimpleNamespace(world_size=8, rank_in_group=3)
    obj.forward(q, q, None, q + 100)
    assert calls[0]["q"].flatten().tolist() == [3]
    assert calls[0]["return_scores"] is False
    assert restores[0][1].shape == (1, 2)
    assert restores[0][2].shape == (8, 2)


def test_single_shard_query_split_reserves_every_workspace_lane(monkeypatch):
    from vllm.v1.worker.workspace import WorkspaceManager, use_workspace_lane

    manager = WorkspaceManager(torch.device("cpu"), num_lanes=2)
    monkeypatch.setattr(indexer, "current_workspace_manager", lambda: manager)
    obj = indexer.B12xSparseIndexer.__new__(indexer.B12xSparseIndexer)
    torch.nn.Module.__init__(obj)
    obj.dcp_world_size = 1
    obj._prefill_query_group = SimpleNamespace(world_size=8)
    obj.topk_tokens = 16
    obj.topk_indices_buffer = torch.empty((32, 16), dtype=torch.int32)
    plan = SimpleNamespace(
        scratch_specs=lambda: (SimpleNamespace(shape=(4096,), dtype=torch.uint8),)
    )
    obj._prepared_plans = {("decode", 1): plan, ("prefill", 32): plan}
    obj._reserve_profile_workspace()
    manager.lock()
    for lane in (0, 1):
        with use_workspace_lane(lane):
            (indices,) = manager.get_simultaneous(((32, 16), torch.int32))
            (scratch,) = manager.get_simultaneous(((4096,), torch.uint8))
            assert indices.nbytes == 2048
            assert scratch.nbytes == 4096


def test_single_shard_auxiliary_preparation_needs_no_cross_shard_kernel():
    obj = indexer.B12xSparseIndexer.__new__(indexer.B12xSparseIndexer)
    torch.nn.Module.__init__(obj)
    obj.dcp_world_size = 1
    obj.sort_selection = False
    assert obj.prepare_dcp_prefill((8,)) is False


def _chunk_rows(context, query_len=8192):
    builder = object.__new__(indexer.B12xIndexerMetadataBuilder)
    builder.max_prefill_buffer_size = 40 * 262144
    chunks = builder._split_prefill_chunks(
        torch.tensor([context], dtype=torch.int32),
        torch.tensor([query_len], dtype=torch.int32),
        0,
        512 * 1024 * 1024,
    )
    return [query.stop - query.start for _, query in chunks]


def _prepared_prefill_indexer(max_q_rows):
    obj = indexer.B12xSparseIndexer.__new__(indexer.B12xSparseIndexer)
    obj.topk_indices_buffer = torch.empty((1, 1), dtype=torch.int32)
    obj._prefill_plan_sizes = indexer._prefill_plan_row_counts(max_q_rows, False)
    obj._prepared_plans = {
        ("prefill", rows): f"prefill-{rows}" for rows in obj._prefill_plan_sizes
    }

    def compile_at_runtime(mode, q_rows):
        raise AssertionError(f"compiled a {mode} plan for {q_rows} rows while serving")

    obj._declare_plan = compile_at_runtime
    return obj


def test_prefill_plan_rows_cover_the_batched_token_limit(monkeypatch):
    monkeypatch.setenv("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", "512")
    monkeypatch.delenv("B12X_PAGED_INDEX_SUPERTILE_K", raising=False)
    assert indexer._prefill_plan_row_counts(8192, False) == [4096, 8192]
    assert indexer._prefill_plan_row_counts(2048, False) == [2048]
    with_sort = indexer._prefill_plan_row_counts(8192, True)
    assert with_sort[-1] == 8192 and 4096 in with_sort


@pytest.mark.parametrize(
    "context",
    [
        8192,  # first chunk of a request: one piece of every query row
        16384,  # logits budget still allows the whole 8192-row chunk
        24576,  # odd 5461-row head piece above the logits-budget rows
    ],
)
def test_prefill_chunks_use_plans_prepared_before_capture(monkeypatch, context):
    """Every prefill chunk the builder emits resolves to a prepared plan.

    Selection borrows plan scratch from the workspace reserved during
    profiling; a chunk beyond the prepared rows would compile a plan and
    grow the workspace under the captured decode graphs.
    """
    monkeypatch.setenv("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", "512")
    monkeypatch.delenv("B12X_PAGED_INDEX_SUPERTILE_K", raising=False)
    rows = _chunk_rows(context)
    assert max(rows) > indexer._prefill_profile_q_rows(8192)
    obj = _prepared_prefill_indexer(8192)
    for chunk_rows in rows:
        plan = obj._plan("prefill", chunk_rows)
        assert plan in obj._prepared_plans.values()
    assert obj._prefill_plan_sizes == [4096, 8192]
