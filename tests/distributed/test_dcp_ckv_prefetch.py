# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eight-worker references for native CKV collectives and stream ownership.

Coordinator-only, with the serving process stopped:
  uv run --no-project .venv/bin/python -m torch.distributed.run \
    --standalone --nproc-per-node=8 -m pytest -q -s \
    tests/distributed/test_dcp_ckv_prefetch.py

Two independent DCP4 groups use real NCCL and separate prefetch communicators.
Small backend reservations exercise collective warmup, synchronous fallback,
history prefetch and current native-byte insertion. Byte payloads are opaque
storage sentinels, not valid attention numerics. No model is loaded; scheduler
admission, full-size memory budgets and attention arithmetic need other tests.
"""

import os
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from vllm.distributed import parallel_state
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.v1.attention.backends.mla import b12x_mla_sparse as backend
from vllm.v1.attention.backends.mla.ckv_prefetch import (
    CKVPrefetchPlan,
    CKVPrefetchRegistry,
    CKVWorkspacePool,
)
from vllm.v1.worker.workspace import use_workspace_lane

pytestmark = pytest.mark.skip_global_cleanup

DCP = 4
PAGE = 64
RECORD = 656
CAPACITY = 128
CURRENT = 64
PADDED = 64


@pytest.fixture(scope="module")
def world():
    if int(os.environ.get("WORLD_SIZE", "0")) != 8:
        pytest.skip("requires coordinator-owned torchrun with eight workers")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    rank = dist.get_rank()
    handles: list[dist.ProcessGroup] = []
    selected = []
    for purpose in ("main", "prefetch"):
        for ranks in (list(range(4)), list(range(4, 8))):
            cpu_group = dist.new_group(
                ranks, backend="gloo", timeout=timedelta(seconds=180)
            )
            gpu_group = dist.new_group(
                ranks, backend="nccl", timeout=timedelta(seconds=180)
            )
            if rank in ranks:
                handles.extend((cpu_group, gpu_group))
                selected.append((purpose, cpu_group, gpu_group))
    device = torch.device("cuda", local_rank)
    groups = {}
    communicators = []
    for purpose, cpu_group, gpu_group in selected:
        comm = PyNcclCommunicator(cpu_group, device)
        communicators.append(comm)
        groups[purpose] = SimpleNamespace(
            world_size=DCP,
            rank_in_group=rank % DCP,
            device_group=gpu_group,
            device_communicator=SimpleNamespace(pynccl_comm=comm),
        )
    yield rank, device, groups
    torch.accelerator.synchronize()
    for comm in communicators:
        comm.destroy()
    for handle in reversed(handles):
        dist.destroy_process_group(handle)
    dist.destroy_process_group()


def _local(length, rank, interleave):
    cycles, tail = divmod(length, DCP * interleave)
    return cycles * interleave + min(interleave, max(0, tail - rank * interleave))


def _byte(replica, lane, step, layer, request, position):
    return (
        replica * 41 + lane * 23 + step * 37 + layer * 19 + request * 11 + position
    ) % 199


def _metadata(device, rank, lengths, queries, interleave):
    rank_lens_cpu = [[_local(n, r, interleave) for n in lengths] for r in range(DCP)]
    starts_cpu = [[0, row[0]] for row in rank_lens_cpu]
    lens = torch.tensor(rank_lens_cpu, dtype=torch.int32, device=device)
    starts = torch.tensor(starts_cpu, dtype=torch.int32, device=device)
    query_starts = torch.tensor(
        [0, queries[0], sum(queries)], dtype=torch.int32, device=device
    )
    dummy = torch.empty(1, dtype=torch.int32, device=device)
    return SimpleNamespace(
        dcp_ckv_gather_eligible=True,
        num_decode_tokens=0,
        is_spec_decode=False,
        num_actual_tokens=sum(queries),
        dcp_padded_total_tokens=PADDED,
        dcp_local_total_tokens=sum(rank_lens_cpu[rank]),
        ckv_selected_indices=dummy,
        ckv_active_counts=dummy,
        dcp_rank_req_starts=starts,
        dcp_rank_req_lens=lens,
        dcp_local_cu_seq_lens=torch.tensor(
            [0, rank_lens_cpu[rank][0], sum(rank_lens_cpu[rank])],
            dtype=torch.int32,
            device=device,
        ),
        global_cache_seq_lens_per_req=torch.tensor(
            lengths, dtype=torch.int32, device=device
        ),
        block_table=torch.tensor([[1], [0]], dtype=torch.int32, device=device),
        query_start_loc=query_starts,
        cp_kv_cache_interleave_size=interleave,
        num_reqs=2,
    ), starts_cpu


def _history(cache, replica, lane, step, layer, rank, lengths, queries, interleave):
    expected = torch.full(cache.shape, 239, dtype=torch.uint8)
    for request, (length, query) in enumerate(zip(lengths, queries)):
        for pos in range(length - query):
            if (pos // interleave) % DCP == rank:
                local = pos // (DCP * interleave) * interleave + pos % interleave
                expected[1 - request, local].fill_(
                    _byte(replica, lane, step, layer, request, pos)
                )
    cache.copy_(expected)


def _produce_current(
    cache, replica, lane, step, layer, rank, lengths, queries, interleave
):
    # Only current positions change: an outstanding history copy may still read
    # immutable history on the dedicated stream.
    torch.cuda._sleep((rank + 1) * 1_000_000)
    for request, (length, query) in enumerate(zip(lengths, queries)):
        for pos in range(length - query, length):
            if (pos // interleave) % DCP == rank:
                local = pos // (DCP * interleave) * interleave + pos % interleave
                cache[1 - request, local].fill_(
                    _byte(replica, lane, step, layer, request, pos)
                )


def _expected(replica, lane, step, layer, lengths, starts, interleave):
    result = torch.zeros((DCP * PADDED, RECORD), dtype=torch.uint8)
    for request, length in enumerate(lengths):
        for pos in range(length):
            rank = (pos // interleave) % DCP
            local = pos // (DCP * interleave) * interleave + pos % interleave
            result[rank * PADDED + starts[rank][request] + local].fill_(
                _byte(replica, lane, step, layer, request, pos)
            )
    return result


@pytest.mark.parametrize("transport", ["pynccl", "torch_nccl"])
@pytest.mark.parametrize("depth", [0, 1])
@pytest.mark.parametrize("interleave", [1, 4])
def test_startup_and_prefetch_use_real_collectives(
    world, monkeypatch, transport, depth, interleave
):
    rank, device, groups = world
    for group in groups.values():
        comm = group.device_communicator.pynccl_comm
        if transport == "pynccl":
            assert comm.available, "PyNccl path must be available for its reference"
            monkeypatch.setattr(comm, "disabled", False)
        else:
            monkeypatch.setattr(comm, "disabled", True)
    monkeypatch.setattr(backend, "get_dcp_group", lambda: groups["main"])
    monkeypatch.setattr(
        parallel_state, "get_dcp_ckv_prefetch_group", lambda: groups["prefetch"]
    )
    calls = []
    actual_collective = backend._dcp_all_gather_current_stream

    def observe(group, source, destination):
        calls.append(
            (group is groups["prefetch"], torch.cuda.current_stream().cuda_stream)
        )
        return actual_collective(group, source, destination)

    monkeypatch.setattr(backend, "_dcp_all_gather_current_stream", observe)
    plan = CKVPrefetchPlan.create(
        requested_depth=depth,
        budget_bytes=0,
        dcp_world_size=DCP,
        local_capacity=CAPACITY,
        record_bytes=RECORD,
        num_lanes=2,
    )
    current = CURRENT if depth else 0
    reservation = backend._CKVReservation(
        CKVPrefetchRegistry(CKVWorkspacePool(plan, device)),
        torch.empty((1, 2, current, RECORD), dtype=torch.uint8, device=device),
        torch.empty((1, 2, DCP * current, RECORD), dtype=torch.uint8, device=device),
    )
    impl = object.__new__(backend.B12xMLASparseImpl)
    impl._ckv_reservation = reservation
    impl._kernel_page_size = PAGE
    impl._cache_record_bytes = RECORD
    impl._ckv_local_capacity = CAPACITY
    impl._ckv_current_capacity = CURRENT
    impl._ckv_gather_enabled = True
    impl._kernel_page_size_finalized = True
    impl.dcp_world_size = DCP
    impl.dcp_rank = rank % DCP
    impl.prepare_profile_collectives()
    assert reservation.collectives_warmed
    warmup_calls = len(calls)
    impl.prepare_profile_collectives()
    assert len(calls) == warmup_calls
    assert len(reservation.gather_streams) == (2 if depth else 0)
    main_stream = torch.cuda.current_stream().cuda_stream
    assert all(stream != main_stream for is_side, stream in calls if is_side)
    assert any(is_side for is_side, _ in calls) == bool(depth)
    caches = [
        [
            torch.empty((2, PAGE, RECORD), dtype=torch.uint8, device=device)
            for _ in range(3)
        ]
        for _ in range(2)
    ]
    snapshots = []
    try:
        for lane in range(2):
            with use_workspace_lane(lane):
                for step, (lengths, queries) in enumerate(
                    (([9, 25], [5, 13]), ([17, 41], [9, 17]))
                ):
                    state = reservation.registry.states.get((0, lane))
                    if state is not None:
                        state.begin_step(torch.cuda.current_stream())
                    metadata, starts = _metadata(
                        device, rank % DCP, lengths, queries, interleave
                    )
                    for layer, cache in enumerate(caches[lane]):
                        _history(
                            cache,
                            rank // DCP,
                            lane,
                            step,
                            layer,
                            rank % DCP,
                            lengths,
                            queries,
                            interleave,
                        )
                    for layer, cache in enumerate(caches[lane]):
                        _produce_current(
                            cache,
                            rank // DCP,
                            lane,
                            step,
                            layer,
                            rank % DCP,
                            lengths,
                            queries,
                            interleave,
                        )
                        gathered, state, index = impl._consume_ckv(
                            cache,
                            metadata,
                            SimpleNamespace(
                                layer_name=f"model.layers.{layer}.self_attn"
                            ),
                            cache,
                        )
                        torch.cuda._sleep(3_000_000)
                        snapshots.append(
                            (
                                gathered.view(-1, RECORD)[: DCP * PADDED].clone(),
                                _expected(
                                    rank // DCP,
                                    lane,
                                    step,
                                    layer,
                                    lengths,
                                    starts,
                                    interleave,
                                ),
                            )
                        )
                        done = torch.cuda.Event()
                        done.record()
                        state.finish_consumer(index, done)
        reservation.registry.clear()
        torch.accelerator.synchronize()
        for observed, expected in snapshots:
            torch.testing.assert_close(observed.cpu(), expected, rtol=0, atol=0)
        assert any(is_side for is_side, _ in calls[warmup_calls:]) == bool(depth)
        dist.barrier()
    finally:
        reservation.registry.clear()
