# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""References for CKV budget, cache lifetime and event-ordered lease reuse.

CPU fixtures check geometry and event contracts. GPU cases use delayed producers
and consumers on real streams to check storage retirement and changed inputs.
Collective and attention correctness require separate distributed references.
"""

import gc

import pytest
import torch

from vllm.v1.attention.backends.mla.ckv_prefetch import (
    CKVPrefetchPlan,
    CKVPrefetchRegistry,
    CKVWorkspacePool,
)


class Event:
    def __init__(self, name, log):
        self.name = name
        self.log = log

    def synchronize(self):
        self.log.append(("synchronize", self.name))


class Stream:
    def __init__(self, name, log):
        self.name = name
        self.log = log

    def wait_event(self, event):
        self.log.append((self.name, "wait", event.name))

    def wait_stream(self, stream):
        self.log.append((self.name, "after", stream.name))


def make_plan(**kwargs):
    args = dict(
        requested_depth=1,
        budget_bytes=0,
        dcp_world_size=4,
        local_capacity=2,
        record_bytes=8,
        num_ubatches=1,
        num_lanes=2,
    )
    args.update(kwargs)
    return CKVPrefetchPlan.create(**args)


def make_state(**kwargs):
    registry = CKVPrefetchRegistry(
        CKVWorkspacePool(make_plan(**kwargs), torch.device("cpu"))
    )
    workspace = torch.empty(32)
    state = registry.for_workspace(workspace, lane=(0, 0))
    return registry, workspace, state


def gather(state, layer, stream, main, log):
    views = state.prepare_gather(layer, stream, producer_stream=main)
    event = Event(f"gather-{layer}", log)
    state.finish_gather(layer, event)
    return views, event


def test_budget_counts_staging_every_ring_slot_and_every_lane():
    plan = make_plan(requested_depth=3, budget_bytes=13 * 16, num_ubatches=2)
    assert plan.effective_depth == 2
    assert plan.ring_slots == 3
    assert plan.lane_nbytes == 13 * 16
    assert plan.total_nbytes == 4 * 13 * 16
    assert make_plan(requested_depth=3).lane_nbytes == 17 * 16
    assert make_plan(budget_bytes=5 * 16).effective_depth == 0


def test_single_layer_lane_keeps_depth_zero_storage():
    plan = make_plan(requested_depth=1, lane_layer_counts=(78, 1))
    assert plan.lane_depths == (1, 0)
    assert plan.effective_depth == 1
    assert plan.lane_ring_slots(0) == 2 and plan.lane_ring_slots(1) == 1
    assert plan.lane_nbytes_for(0) == 9 * 16 and plan.lane_nbytes_for(1) == 5 * 16
    assert plan.total_nbytes == 14 * 16
    assert plan.lane_offset(0, 1) == 9 * 16
    two = make_plan(requested_depth=1, lane_layer_counts=(78, 1), num_ubatches=2)
    assert two.total_nbytes == 2 * 14 * 16
    assert two.lane_offset(1, 0) == 14 * 16 and two.lane_offset(1, 1) == 23 * 16
    assert make_plan(requested_depth=3, lane_layer_counts=(3, 2)).lane_depths == (2, 1)
    with pytest.raises(ValueError, match="one positive count per lane"):
        make_plan(lane_layer_counts=(78,))
    with pytest.raises(ValueError, match="one positive count per lane"):
        make_plan(lane_layer_counts=(78, 0))
    with pytest.raises(ValueError, match="deepest lane"):
        CKVPrefetchPlan(2, 4, 2, 8, 1, 2, (1, 0))
    with pytest.raises(ValueError, match="one depth per lane"):
        CKVPrefetchPlan(1, 4, 2, 8, 1, 2, (1,))


def test_drafter_lane_state_uses_its_own_ring_and_no_lookahead():
    plan = make_plan(requested_depth=1, lane_layer_counts=(78, 1))
    registry = CKVPrefetchRegistry(CKVWorkspacePool(plan, torch.device("cpu")))
    workspace = torch.empty(32)
    target = registry.for_workspace(workspace, lane=(0, 0))
    drafter = registry.for_workspace(workspace, lane=(0, 1))
    assert target.ring_slots == 2 and target.lookahead_depth == 1
    assert drafter.ring_slots == 1 and drafter.lookahead_depth == 0
    assert drafter.storage.numel() == 5 * 16
    assert drafter.storage.data_ptr() == target.storage.data_ptr() + 9 * 16
    with pytest.raises(ValueError, match="outside the reservation"):
        drafter.views(1)
    log: list[tuple[str, ...]] = []
    main = Stream("main", log)
    drafter.register_cache(78, torch.empty(1))
    drafter.register_cache(79, torch.empty(1))
    assert drafter.targets(78) == []
    target.register_cache(0, torch.empty(1))
    target.register_cache(1, torch.empty(1))
    assert target.targets(0) == [1]
    gather(drafter, 78, main, main, log)
    assert drafter.consume(78, main).shape == (8, 8)
    drafter.finish_consumer(78, Event("consume-78", log))
    registry.clear()


@pytest.mark.parametrize("budget", [1, 5 * 16 - 1])
def test_positive_budget_rejects_mandatory_storage_shortfall(budget):
    with pytest.raises(ValueError, match="mandatory.*80 bytes per lane"):
        make_plan(budget_bytes=budget)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(requested_depth=-1),
        dict(budget_bytes=-1),
        dict(local_capacity=0),
        dict(num_lanes=0),
    ],
)
def test_invalid_reservations_fail_before_allocation(kwargs):
    with pytest.raises(ValueError):
        make_plan(**kwargs)


def test_lane_and_ring_views_are_disjoint_and_staging_is_shared():
    registry, workspace, state = make_state(num_ubatches=2)
    lane = registry.for_workspace(workspace, lane=(1, 1))
    local, slot0 = state.views(0)
    local1, slot1 = state.views(1)
    assert local.data_ptr() == local1.data_ptr()
    local.fill_(3)
    slot0.fill_(4)
    slot1.fill_(5)
    lane.storage.fill_(6)
    assert torch.all(local == 3)
    assert torch.all(slot0 == 4)
    assert torch.all(slot1 == 5)
    assert slot0.shape == (8, 8)
    assert (
        state.storage.untyped_storage().data_ptr()
        == lane.storage.untyped_storage().data_ptr()
    )
    with pytest.raises(ValueError, match="outside"):
        state.views(2)
    with pytest.raises(ValueError, match="outside"):
        registry.for_workspace(workspace, lane=(2, 0))
    with pytest.raises(RuntimeError, match="already has"):
        registry.pool.acquire((0, 0))


def test_gather_and_consumer_fence_ring_reuse_and_shared_staging():
    _, workspace, state = make_state()
    log: list[tuple[str, ...]] = []
    main, side = Stream("main", log), Stream("side", log)
    (_, first), _ = gather(state, 0, side, main, log)
    first.fill_(9)
    consumed = state.consume(0, main)
    assert consumed.data_ptr() == first.data_ptr()
    state.finish_consumer(0, Event("reader-0", log))
    gather(state, 1, side, main, log)
    # A synchronous fallback reuses staging only after the asynchronous gather.
    state.order_staging(main)
    (_, reused), _ = gather(state, 2, main, main, log)
    assert reused.data_ptr() == first.data_ptr()
    assert log == [
        ("side", "after", "main"),
        ("main", "wait", "gather-0"),
        ("side", "wait", "gather-0"),
        ("side", "after", "main"),
        ("main", "wait", "gather-1"),
        ("main", "wait", "gather-1"),
        ("main", "wait", "reader-0"),
        ("main", "after", "main"),
    ]
    state.close()
    assert ("synchronize", "gather-1") in log
    assert ("synchronize", "gather-2") in log
    assert workspace.numel() == 32


def test_close_completes_consumers_before_releasing_lane():
    registry, workspace, state = make_state()
    log: list[tuple[str, ...]] = []
    main = Stream("main", log)
    gather(state, 0, main, main, log)
    state.consume(0, main)
    state.finish_consumer(0, Event("reader", log))
    replacement = registry.for_workspace(workspace, lane=(0, 0), generation=1)
    assert log[-2:] == [("synchronize", "gather-0"), ("synchronize", "reader")]
    assert replacement.storage.data_ptr() == state.storage.data_ptr()
    with pytest.raises(RuntimeError, match="closed"):
        state.views(0)


def test_incomplete_producer_or_consumer_cannot_release_or_reuse_storage():
    registry, workspace, state = make_state(requested_depth=0)
    log: list[tuple[str, ...]] = []
    main = Stream("main", log)
    state.prepare_gather(0, main, producer_stream=main)
    with pytest.raises(RuntimeError, match="completion event"):
        state.close()
    state.finish_gather(0, Event("producer", log))
    state.consume(0, main)
    with pytest.raises(RuntimeError, match="completion event"):
        state.prepare_gather(1, main, producer_stream=main)
    with pytest.raises(RuntimeError, match="completion event"):
        registry.for_workspace(workspace, lane=(0, 0), generation=1)
    state.finish_consumer(0, Event("reader", log))
    state.close()
    assert ("synchronize", "reader") in log


def test_interrupted_step_discards_pending_but_preserves_write_dependency():
    _, workspace, state = make_state(requested_depth=0)
    log: list[tuple[str, ...]] = []
    main, side = Stream("main", log), Stream("side", log)
    state.enter_layer(4, main)
    gather(state, 5, side, main, log)
    with pytest.raises(RuntimeError, match="unconsumed"):
        state.prepare_gather(6, main, producer_stream=main)
    state.enter_layer(0, main)
    assert not state.pending
    gather(state, 0, main, main, log)
    assert ("main", "wait", "gather-5") in log
    assert workspace.numel() == 32


def test_first_request_learns_caches_and_missing_layer_stops_lookahead():
    _, workspace, state = make_state(requested_depth=3)
    assert state.targets(0) == []
    for index in [0, 1, 3, 4]:
        state.register_cache(index, torch.empty(1))
    assert state.targets(0) == [1]
    state.register_cache(2, torch.empty(1))
    assert state.targets(0) == [1, 2, 3]
    log: list[tuple[str, ...]] = []
    main = Stream("main", log)
    gather(state, 1, main, main, log)
    assert state.targets(0) == [2, 3]
    assert workspace.numel() == 32


def test_cache_replacement_finishes_users_and_clears_other_learned_caches():
    registry, workspace, state = make_state()
    for index in [0, 1]:
        state.register_cache(index, torch.empty(1))
    log: list[tuple[str, ...]] = []
    main = Stream("main", log)
    gather(state, 1, main, main, log)
    changed = torch.ones(1)
    state.register_cache(0, changed)
    assert log[-1] == ("synchronize", "gather-1")
    assert state.layer_caches == {0: changed}
    assert not state.pending
    registry.reset_caches()
    assert not state.layer_caches
    assert workspace.numel() == 32


def test_target_and_drafter_caches_and_storage_remain_separate():
    registry, workspace, target = make_state()
    draft = registry.for_workspace(workspace, lane=(0, 1))
    target.register_cache(0, torch.ones(1))
    draft.register_cache(0, torch.zeros(1))
    assert target is not draft
    assert target.storage.data_ptr() != draft.storage.data_ptr()
    target.reset_caches()
    assert 0 in draft.layer_caches
    assert registry.for_workspace(workspace, lane=(0, 1)) is draft


def test_view_identity_change_retires_same_lane_without_stealing_other_lane():
    registry, workspace, state = make_state()
    draft = registry.for_workspace(workspace, lane=(0, 1))
    assert registry.for_workspace(workspace.view(-1), lane=(0, 0)) is state
    replacement = registry.for_workspace(workspace[:16], lane=(0, 0))
    assert replacement is not state
    assert registry.states[(0, 1)] is draft


def test_dead_workspace_pruning_synchronizes_and_releases_lease():
    registry, workspace, state = make_state()
    log: list[tuple[str, ...]] = []
    main = Stream("main", log)
    gather(state, 0, main, main, log)
    del workspace
    gc.collect()
    assert state.workspace_storage_ref() is None
    replacement_workspace = torch.empty(32)
    replacement = registry.for_workspace(replacement_workspace, lane=(0, 0))
    assert replacement is not state
    assert ("synchronize", "gather-0") in log
    registry.clear()
    assert not registry.states


def test_gather_stream_is_created_once_per_lane():
    registry, workspace, target = make_state()
    draft = registry.for_workspace(workspace, lane=(0, 1))
    created: list[Stream] = []

    def factory():
        stream = Stream(f"side-{len(created)}", [])
        created.append(stream)
        return stream

    assert target.get_gather_stream(factory) is target.get_gather_stream(factory)
    assert draft.get_gather_stream(factory) is not target.get_gather_stream(factory)
    assert len(created) == 2


def test_changed_inputs_replace_gathered_bytes_without_changing_storage():
    _, workspace, state = make_state(requested_depth=0)
    log: list[tuple[str, ...]] = []
    main = Stream("main", log)
    addresses = []
    for value in [11, 29]:
        state.begin_step(main)
        staging, gathered = state.prepare_gather(0, main, producer_stream=main)
        staging.fill_(value)
        for rank in range(4):
            gathered[rank * 2 : (rank + 1) * 2].copy_(staging)
        state.finish_gather(0, Event(f"gather-{value}", log))
        result = state.consume(0, main)
        assert torch.all(result == value)
        addresses.append(result.data_ptr())
        state.finish_consumer(0, Event(f"reader-{value}", log))
    assert addresses[0] == addresses[1]
    assert ("main", "wait", "reader-11") in log
    assert workspace.numel() == 32


# These references exercise real stream dependencies with delayed device work.
# They do not establish all-rank communication or native attention correctness.
_CUDA_DELAY_CYCLES = 20_000_000


def _cuda_registry(depth=1):
    plan = CKVPrefetchPlan.create(
        requested_depth=depth,
        budget_bytes=0,
        dcp_world_size=4,
        local_capacity=16,
        record_bytes=8,
        num_lanes=2,
    )
    registry = CKVPrefetchRegistry(CKVWorkspacePool(plan, torch.device("cuda")))
    workspace = torch.empty(32, device="cuda")
    state = registry.for_workspace(workspace, lane=(0, 0))
    producer = torch.cuda.Stream()
    gather_stream = state.get_gather_stream()
    consumer = torch.cuda.Stream()
    for stream in (producer, gather_stream, consumer):
        stream.wait_stream(torch.cuda.current_stream())
    return registry, workspace, state, producer, gather_stream, consumer


def _cuda_gather(state, layer, value, producer, stream):
    # Distinct source allocations prevent the fixture from overwriting a source
    # while a preceding gather reads it. Ring and staging addresses remain shared.
    source = torch.empty((16, 8), dtype=torch.uint8, device="cuda")
    with torch.cuda.stream(producer):
        torch.cuda._sleep(_CUDA_DELAY_CYCLES)
        source.fill_(value)
    local, gathered = state.prepare_gather(layer, stream, producer_stream=producer)
    with torch.cuda.stream(stream):
        torch.cuda._sleep(_CUDA_DELAY_CYCLES)
        local.copy_(source)
        for rank in range(4):
            gathered[rank * 16 : (rank + 1) * 16].copy_(local)
        writer = torch.cuda.Event()
        writer.record()
    state.finish_gather(layer, writer)
    return source, gathered, writer


def _cuda_consume(state, layer, consumer):
    result = state.consume(layer, consumer)
    snapshot = torch.empty_like(result)
    with torch.cuda.stream(consumer):
        # Ring reuse must wait for this delayed read, not just the gather writer.
        torch.cuda._sleep(4 * _CUDA_DELAY_CYCLES)
        snapshot.copy_(result)
        reader = torch.cuda.Event()
        reader.record()
    state.finish_consumer(layer, reader)
    return snapshot, reader


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires coordinator-owned CUDA GPU"
)
def test_cuda_ring_reuse_preserves_changed_layer_and_step_bytes():
    registry, workspace, state, producer, side, consumer = _cuda_registry()
    sources, snapshots = [], []
    completions: list[torch.cuda.Event] = []
    addresses: dict[int, int] = {}
    try:
        for step in range(2):
            state.begin_step(consumer)
            for layer in range(4):
                state.enter_layer(layer, consumer)
                value = 17 + step * 31 + layer
                source, gathered, writer = _cuda_gather(
                    state, layer, value, producer, side
                )
                sources.append(source)
                slot = layer % state.pool.plan.ring_slots
                addresses.setdefault(slot, gathered.data_ptr())
                assert gathered.data_ptr() == addresses[slot]
                snapshot, reader = _cuda_consume(state, layer, consumer)
                snapshots.append((snapshot, value))
                completions.extend((writer, reader))
        state.close()
        assert all(event.query() for event in completions)
        for snapshot, value in snapshots:
            assert torch.equal(
                snapshot.cpu(), torch.full(snapshot.shape, value, dtype=torch.uint8)
            )
        assert workspace.numel() == 32
    finally:
        registry.clear()


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires coordinator-owned CUDA GPU"
)
def test_cuda_interrupted_step_and_missing_cache_preserve_dependencies():
    registry, workspace, state, producer, side, consumer = _cuda_registry(depth=0)
    sources = []
    try:
        state.enter_layer(4, consumer)
        source, _, interrupted_writer = _cuda_gather(state, 5, 97, producer, side)
        sources.append(source)
        # A restarted execution discards unconsumed work but must order its
        # staging and ring reuse after that work's real CUDA completion.
        state.enter_layer(0, consumer)
        assert not state.pending
        assert state.targets(0) == []
        source, _, writer = _cuda_gather(state, 0, 43, producer, consumer)
        sources.append(source)
        snapshot, reader = _cuda_consume(state, 0, consumer)
        state.close()
        assert interrupted_writer.query() and writer.query() and reader.query()
        assert torch.equal(
            snapshot.cpu(), torch.full(snapshot.shape, 43, dtype=torch.uint8)
        )
        assert workspace.numel() == 32
    finally:
        registry.clear()


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires coordinator-owned CUDA GPU"
)
def test_cuda_rebinding_retires_writers_and_readers_without_stealing_other_lane():
    registry, workspace, state, producer, side, consumer = _cuda_registry()
    sources = []
    snapshots = []
    try:
        cache0 = torch.zeros(1, device="cuda")
        cache2 = torch.zeros(1, device="cuda")
        state.register_cache(0, cache0)
        state.register_cache(2, cache2)
        assert state.targets(0) == []  # Missing layer 1 stops prefetch discovery.
        source, _, writer = _cuda_gather(state, 0, 61, producer, side)
        sources.append(source)
        snapshot, reader = _cuda_consume(state, 0, consumer)
        snapshots.append((snapshot, 61))
        # An additional unconsumed writer must also finish before lease release.
        source, _, pending_writer = _cuda_gather(state, 1, 103, producer, side)
        sources.append(source)
        draft = registry.for_workspace(workspace, lane=(0, 1))
        draft_cache = torch.ones(1, device="cuda")
        draft.register_cache(0, draft_cache)
        with torch.cuda.stream(producer):
            draft.storage.fill_(211)
        replacement_workspace = workspace[:16]
        replacement = registry.for_workspace(
            replacement_workspace, lane=(0, 0), generation=1
        )
        assert writer.query() and reader.query() and pending_writer.query()
        assert replacement.storage.data_ptr() == state.storage.data_ptr()
        assert not replacement.layer_caches
        assert registry.states[(0, 1)] is draft
        assert draft.layer_caches[0] is draft_cache
        with pytest.raises(RuntimeError, match="closed"):
            state.views(0)
        source, _, writer2 = _cuda_gather(replacement, 0, 151, producer, side)
        sources.append(source)
        snapshot2, reader2 = _cuda_consume(replacement, 0, consumer)
        snapshots.append((snapshot2, 151))
        replacement.register_cache(0, cache0)
        replacement.register_cache(0, cache2)
        assert writer2.query() and reader2.query()
        assert replacement.layer_caches == {0: cache2}
        assert draft.layer_caches[0] is draft_cache
        assert torch.equal(
            draft.storage.cpu(), torch.full(draft.storage.shape, 211, dtype=torch.uint8)
        )
        for result, value in snapshots:
            assert torch.equal(
                result.cpu(), torch.full(result.shape, value, dtype=torch.uint8)
            )
    finally:
        registry.clear()
