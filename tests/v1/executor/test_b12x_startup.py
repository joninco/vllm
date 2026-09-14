# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cooperative control for the b12x warmup prelude."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch.distributed as dist

from vllm.v1.executor.abstract import Executor, _aggregate_b12x_progress
from vllm.v1.worker.b12x_startup import (
    B12xPreparationCoordinator,
    _all_gather,
    _authorize_tuning,
)


def _progress(*, done=False, pending=False, ready=()):
    return SimpleNamespace(
        done=done,
        pending_compilation=pending,
        ready_collectives=ready,
    )


class _Result:
    def __init__(self, events):
        self.events = events

    def close(self):
        self.events.append("result-close")


class _Job:
    def __init__(self, events, progress):
        self.events = events
        self.progress = iter(progress)
        self.session = SimpleNamespace(_pool=None, cancel_tuning=self.cancel)
        self.events_session = None
        self.keys = []
        self.tunings = []

    def cancel(self):
        self.events.append("cancel")

    def advance(self, *, collective_key=None, tuning=None):
        self.keys.append(collective_key)
        self.tunings.append(tuning)
        self.events.append("advance")
        return next(self.progress)

    def result(self):
        self.events.append("result")
        return _Result(self.events)

    def close(self):
        self.events.append("job-close")


class _Session:
    def __init__(self, job, events):
        self._job = job
        self.events = events
        self.state = "OPEN"
        self._pool = None

    def begin(self, requests, *, autotune=None):
        self.events.append(("begin", autotune))
        return self._job

    def cancel_tuning(self):
        self.events.append("cancel")


def _batches(autotune=True):
    return [((object(),), autotune)]


class _PeerWorld:
    ranks = (0, 1)

    def __init__(self, peer):
        self.peer = peer
        self.tcp_store_group = self

    def all_gather_obj(self, payload):
        peer = {
            "round": payload["round"],
            "global_rank": 1 if payload["global_rank"] == 0 else 0,
            "world_ranks": (0, 1),
            "stop": False,
            "ready": (),
            "local_done": False,
            "error": None,
            "cleanup_complete": False,
            **self.peer(payload),
        }
        return sorted((payload, peer), key=lambda item: item["global_rank"])


def test_control_exchange_isolated_from_world_collectives() -> None:
    store = dist.HashStore()

    class _CpuGroup:
        def get_group_store(self):
            return store

    cpu_group = _CpuGroup()

    class _World:
        ranks = (0, 1)
        world_size = 2

        def __init__(self, rank):
            self.rank_in_group = rank
            self.cpu_group = cpu_group

        def broadcast_object(self, *_args, **_kwargs):
            raise AssertionError("control traffic entered the world collective stream")

    worlds = [_World(rank) for rank in range(2)]
    for round_number in (3, 4):
        payloads = [
            {
                "round": round_number,
                "global_rank": rank,
                "world_ranks": (0, 1),
            }
            for rank in range(2)
        ]
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(_all_gather, world, payload)
                for world, payload in zip(worlds, payloads)
            ]
            gathered = [future.result(timeout=2) for future in futures]

        assert gathered == [payloads, payloads]


def test_tuning_authorization_selects_once_across_disjoint_rank_shards() -> None:
    gathered = [
        {
            "global_rank": 0,
            "tuning": (("query", (0, 1), {"width": 4}, 3.0, 0),),
        },
        {
            "global_rank": 1,
            "tuning": (("query", (0, 1), {"width": 2}, 1.0, 1),),
        },
    ]

    assert _authorize_tuning(gathered, (0, 1)) == (
        "query",
        (0, 1),
        {"width": 2},
        1.0,
        1,
    )


def test_progress_aggregates_candidate_shards_and_physical_compilations() -> None:
    from b12x.preparation import PreparationProgress

    outcomes = [
        {
            "global_rank": rank,
            "progress": PreparationProgress(
                running=True,
                pending_compilation=False,
                ready_collectives=(),
                done=False,
                phase="autotuning",
                component_id="sequence.gdn_prefill",
                request_name="shared-query",
                candidate_count=96,
                candidates_prepared=96,
                measured_candidates=96 * rank,
                compilations=300 + rank,
                active_compilations=rank,
                latest_round_us=(float(rank + 1),),
                elapsed_seconds=float(rank + 1),
                candidate_sharded=True,
            ),
        }
        for rank in range(2)
    ]

    progress = _aggregate_b12x_progress(outcomes)
    assert progress.candidate_count == 192
    assert progress.candidates_prepared == 192
    assert progress.measured_candidates == 96
    assert progress.compilations == 601
    assert progress.active_compilations == 1
    assert progress.latest_round_us == (1.0, 2.0)
    assert progress.elapsed_seconds == 2.0


def test_progress_does_not_finish_until_every_rank_finishes() -> None:
    from b12x.preparation import PreparationProgress

    ready = PreparationProgress(False, False, (), True, phase="ready")
    active = PreparationProgress(True, False, (), False, phase="priming")
    progress = _aggregate_b12x_progress(
        [
            {"global_rank": 0, "done": False, "progress": ready},
            {"global_rank": 1, "done": False, "progress": active},
        ]
    )
    assert progress.done is False
    assert progress.phase == "priming"


def test_progress_does_not_sum_replicated_fixed_candidate_count() -> None:
    from b12x.preparation import PreparationProgress

    fixed = PreparationProgress(
        True,
        True,
        (),
        False,
        phase="compiling",
        component_id="comm.pcie",
        request_name="fixed",
        candidate_count=1,
    )
    progress = _aggregate_b12x_progress(
        [
            {"global_rank": rank, "done": False, "progress": fixed}
            for rank in range(2)
        ]
    )
    assert progress.candidate_count == 1


def test_coordinator_returns_global_tuning_winner_to_local_job() -> None:
    from b12x.preparation import FrozenMapping, TuningRequirement

    events = []
    local = TuningRequirement("query", (0, 1), FrozenMapping({"width": 4}), 3.0, 0)
    job = _Job(
        events,
        [_progress(ready=()), _progress(done=True)],
    )
    job.progress = iter(
        [
            SimpleNamespace(
                done=False,
                pending_compilation=False,
                ready_collectives=(),
                ready_tuning=(local,),
            ),
            _progress(done=True),
        ]
    )

    def peer(payload):
        return {
            "tuning": (
                ("0,1|query", (0, 1), {"width": 2}, 1.0, 1),
            )
            if payload["round"] == 0
            else (),
            "local_done": payload["round"] >= 1,
            "cleanup_complete": payload["round"] >= 1,
        }

    coordinator = B12xPreparationCoordinator(
        _Session(job, events),
        _batches(),
        global_rank=0,
        world_group=_PeerWorld(peer),
    )

    assert coordinator.advance()["done"] is False
    assert coordinator.advance()["done"] is True
    assert job.tunings[0] is None
    assert job.tunings[1].assignment == FrozenMapping({"width": 2})


def test_local_only_cancel_still_completes_and_primes() -> None:
    events = []
    job = _Job(events, [_progress(done=True)])
    coordinator = B12xPreparationCoordinator(
        _Session(job, events),
        _batches(),
        global_rank=0,
        world_group=None,
        process_local_only=True,
    )

    outcome = coordinator.advance(cancel_tuning=True)

    assert outcome["done"] is True
    assert outcome["cleanup_complete"] is True
    assert events == [
        ("begin", True),
        "cancel",
        "advance",
        "result",
        "result-close",
    ]


def test_default_only_batch_disables_tuning_for_job() -> None:
    events = []
    coordinator = B12xPreparationCoordinator(
        _Session(_Job(events, [_progress(done=True)]), events),
        _batches(autotune=False),
        global_rank=0,
        world_group=None,
        process_local_only=True,
    )

    assert events == [("begin", False)]
    coordinator.abort()


def test_batches_run_in_order_and_finish_after_the_last() -> None:
    events = []
    first = _Job(events, [_progress(done=True)])
    second = _Job(events, [_progress(done=True)])
    session = _Session(first, events)
    jobs = iter([first, second])
    session.begin = lambda requests, *, autotune=None: (
        events.append(("begin", autotune)) or next(jobs)
    )
    coordinator = B12xPreparationCoordinator(
        session,
        [((object(),), True), ((object(),), False)],
        global_rank=0,
        world_group=None,
        process_local_only=True,
    )

    assert coordinator.advance()["done"] is False
    assert coordinator.advance()["done"] is True
    assert events == [
        ("begin", True), "advance", "result", "result-close",
        ("begin", False), "advance", "result", "result-close",
    ]


def test_pending_compilation_wait_is_bounded() -> None:
    events = []
    job = _Job(events, [_progress(pending=True)])

    class _Pool:
        def wait_for_progress(self, *, timeout):
            events.append(("wait", timeout))

    job.session._pool = _Pool()
    coordinator = B12xPreparationCoordinator(
        _Session(job, events),
        _batches(),
        global_rank=0,
        world_group=None,
        process_local_only=True,
    )

    assert coordinator.advance()["done"] is False
    assert ("wait", 0.05) in events


def test_collective_runs_only_after_every_participant_is_ready() -> None:
    events = []
    requirement = SimpleNamespace(key="shared", ranks=(0, 1))
    job = _Job(
        events,
        [_progress(ready=(requirement,)), _progress(done=True)],
    )

    def peer(payload):
        return {
            "ready": (("shared", (0, 1)),) if payload["round"] == 0 else (),
            "local_done": payload["round"] >= 1,
            "cleanup_complete": payload["round"] >= 1,
        }

    coordinator = B12xPreparationCoordinator(
        _Session(job, events),
        _batches(),
        global_rank=0,
        world_group=_PeerWorld(peer),
    )

    assert coordinator.advance()["done"] is False
    assert job.keys == [None]
    assert coordinator.advance()["done"] is True
    assert job.keys == [None, "shared"]


def test_nonparticipant_does_not_receive_another_rank_authorization() -> None:
    events = []
    requirement = SimpleNamespace(key="z-rank-one", ranks=(1,))
    job = _Job(
        events,
        [_progress(ready=(requirement,)), _progress(ready=(requirement,))],
    )

    def peer(payload):
        return {
            "ready": (("a-rank-zero", (0,)),)
            if payload["round"] == 0
            else (),
        }

    coordinator = B12xPreparationCoordinator(
        _Session(job, events),
        _batches(),
        global_rank=1,
        world_group=_PeerWorld(peer),
    )

    coordinator.advance()
    coordinator.advance()
    assert job.keys == [None, None]


def test_finished_rank_stays_in_world_rounds_until_peer_finishes() -> None:
    events = []
    job = _Job(events, [_progress(done=True)])

    def peer(payload):
        return {
            "local_done": payload["round"] >= 2,
            "cleanup_complete": payload["round"] >= 2,
        }

    coordinator = B12xPreparationCoordinator(
        _Session(job, events),
        _batches(),
        global_rank=0,
        world_group=_PeerWorld(peer),
    )

    assert coordinator.advance()["done"] is False
    assert coordinator.advance()["done"] is False
    assert coordinator.advance()["done"] is True
    assert events.count("advance") == 1


def test_peer_failure_is_reported_after_cleanup_acknowledgement() -> None:
    events = []
    job = _Job(events, [_progress(pending=True)])

    def peer(payload):
        error = {"rank": 1, "type": "ValueError", "message": "peer failed"}
        return {
            "error": error,
            "local_done": payload["round"] >= 1,
            "cleanup_complete": payload["round"] >= 1,
        }

    coordinator = B12xPreparationCoordinator(
        _Session(job, events),
        _batches(),
        global_rank=0,
        world_group=_PeerWorld(peer),
    )

    first = coordinator.advance()
    assert first["done"] is False
    assert first["cleanup_complete"] is True
    final = coordinator.advance()
    assert final["done"] is True
    assert final["error"] == {
        "rank": 1,
        "type": "ValueError",
        "message": "peer failed",
    }
    assert events.count("job-close") == 1


def test_abort_closes_active_job_once() -> None:
    events = []
    job = _Job(events, [_progress(pending=True)])
    coordinator = B12xPreparationCoordinator(
        _Session(job, events),
        _batches(),
        global_rank=0,
        world_group=None,
        process_local_only=True,
    )

    assert coordinator.abort()["cleanup_complete"] is True
    coordinator.abort()
    assert events.count("job-close") == 1


def test_executor_drains_later_replies_before_raising_rank_error() -> None:
    calls = []
    rounds = iter(
        (
            [
                {"native": False, "done": False},
                {"native": False, "done": False},
            ],
            [
                {
                    "done": False,
                    "cleanup_complete": True,
                    "error": {
                        "rank": 0,
                        "type": "ValueError",
                        "message": "first",
                    },
                },
                {"done": False, "cleanup_complete": False, "error": None},
            ],
            [
                {
                    "done": True,
                    "cleanup_complete": True,
                    "error": {
                        "rank": 0,
                        "type": "ValueError",
                        "message": "first",
                    },
                },
                {"done": True, "cleanup_complete": True, "error": None},
            ],
        )
    )

    def collective_rpc(method, kwargs=None):
        calls.append((method, kwargs))
        if method == "abort_b12x_preparation":
            return [{"done": True, "cleanup_complete": True}] * 2
        return next(rounds)

    executor = SimpleNamespace(
        collective_rpc=collective_rpc,
        _b12x_autotuning_cancel=threading.Event(),
    )
    with pytest.raises(RuntimeError, match="rank 0: ValueError: first"):
        Executor._run_b12x_preparation(executor, stage="weights")

    assert [method for method, _ in calls] == [
        "begin_b12x_preparation",
        "advance_b12x_preparation",
        "advance_b12x_preparation",
        "abort_b12x_preparation",
    ]


def test_executor_forwards_sticky_cancellation() -> None:
    calls = []
    responses = iter(
        (
            [{"native": False, "done": False}],
            [{"native": False, "done": True, "error": None}],
        )
    )

    def collective_rpc(method, kwargs=None):
        calls.append((method, kwargs))
        return next(responses)

    cancel = threading.Event()
    cancel.set()
    executor = SimpleNamespace(
        collective_rpc=collective_rpc,
        _b12x_autotuning_cancel=cancel,
    )
    Executor._run_b12x_preparation(executor, stage="weights")
    assert calls[0] == (
        "begin_b12x_preparation",
        {"stage": "weights"},
    )
    assert calls[-1] == (
        "advance_b12x_preparation",
        {"cancel_tuning": True},
    )


@pytest.mark.parametrize("variant", ("batched", "varlen"))
def test_attention_tuning_rendezvous_ignores_rank_local_device_ordinal(variant):
    import torch
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

    from b12x.attention import varlen
    from b12x.preparation.session import PreparationJob

    mode = FakeTensorMode()
    ranks = (0, 1, 2, 3)
    gathered = []
    for rank in ranks:
        device = torch.device("cuda", rank)

        def metadata(shape, dtype):
            return FakeTensor(mode, torch.empty(shape, device="meta", dtype=dtype), device)

        q, k, v = (metadata((9216, 16, 64), torch.bfloat16) for _ in range(3))
        if variant == "varlen":
            plan = varlen.plan(
                q, k, v, metadata((2,), torch.int32),
                max_seqlen_q=9216, max_seqlen_k=9216, causal=False,
            )
        else:
            plan = varlen.plan_batched(q, k, v, causal=False)
        request = SimpleNamespace(plan=plan, dependencies=())
        configuration = plan.contract.configure(plan.query, device=None)
        obligation = SimpleNamespace(request=request, configuration=configuration)
        key = PreparationJob._choice_key(None, obligation, {})
        gathered.append({
            "global_rank": rank,
            "tuning": ((key, ranks, {"tile_m": 128, "tile_n": 64}, 10.0 + rank, rank),),
        })
    authorized = _authorize_tuning(gathered, ranks)
    assert authorized is not None
    assert authorized[1:] == (ranks, {"tile_m": 128, "tile_n": 64}, 10.0, 0)
