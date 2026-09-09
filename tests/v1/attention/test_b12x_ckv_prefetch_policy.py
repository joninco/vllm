# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU references for CKV budget, cache lifetime and event-ordered lease reuse.

Inputs are planned geometry, CPU workspace views, and fake CUDA events/streams.
Assertions detect missing dependencies and storage aliasing without GPU work;
they do not establish CUDA collective or attention correctness.
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
