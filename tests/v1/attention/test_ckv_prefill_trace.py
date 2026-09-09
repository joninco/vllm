# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU proofs for diagnostic identities without extra CUDA operations."""

from contextlib import contextmanager

import pytest
import torch

from vllm.v1.attention.backends.mla.ckv_prefetch import (
    CKVPrefetchPlan,
    CKVPrefetchRegistry,
    CKVWorkspacePool,
)
from vllm.v1.attention.backends.mla.prefill_diagnostics import (
    PrefillTrace,
    get_prefill_trace,
    install_state_trace,
)


class Event:
    def __init__(self):
        self.records = []
        self.synchronizations = 0

    def record(self, stream):
        self.records.append(stream)

    def synchronize(self):
        self.synchronizations += 1


class Stream:
    def __init__(self):
        self.events = []
        self.streams = []
        self.synchronizations = 0

    def synchronize(self):
        self.synchronizations += 1

    def wait_event(self, event):
        self.events.append(event)

    def wait_stream(self, stream):
        self.streams.append(stream)


@pytest.fixture
def scopes(monkeypatch):
    names = []

    @contextmanager
    def scope(name):
        names.append(name)
        yield

    monkeypatch.setattr(torch.profiler, "record_function", scope)
    monkeypatch.setattr(torch.autograd.profiler, "_is_profiler_enabled", True)
    return names


def test_disabled_flag_is_resolved_once_without_creating_ledger(monkeypatch):
    get_prefill_trace.cache_clear()
    monkeypatch.setenv("VLLM_DCP_PREFILL_TRACE", "0")
    try:
        assert get_prefill_trace() is None
        monkeypatch.setenv("VLLM_DCP_PREFILL_TRACE", "1")
        assert get_prefill_trace() is None
    finally:
        get_prefill_trace.cache_clear()


def test_inactive_capture_formats_nothing_and_delegates_once(monkeypatch, scopes):
    monkeypatch.setattr(torch.autograd.profiler, "_is_profiler_enabled", False)
    trace = PrefillTrace()
    event, stream = Event(), Stream()
    with trace.batch(), trace.scope("must_not_format", invalid=object()):
        trace.record(event, stream)
        trace.wait(stream, event, reason=1)
    assert scopes == []
    assert event.records == [stream] and stream.events == [event]
    assert trace._event_count == 0 and trace._stream_count == 0


def test_repeated_event_records_keep_object_identity_and_record_sequence(scopes):
    trace, event, stream = PrefillTrace(), Event(), Stream()
    with trace.batch():
        trace.record(event, stream)
        trace.wait(stream, event, reason=1)
        trace.record(event, stream)
        trace.wait(stream, event, reason=2)
    assert len(event.records) == len(stream.events) == 2
    records = [name for name in scopes if "/event_record " in name]
    waits = [name for name in scopes if "/event_wait " in name]
    assert "event=1" in records[0] and "record=1" in records[0]
    assert "event=1" in records[1] and "record=2" in records[1]
    assert "record=1" in waits[0] and "record=2" in waits[1]
    assert event.synchronizations == 0


def test_unobserved_record_is_unknown_and_object_reuse_never_reuses_id(scopes):
    import gc

    trace, stream = PrefillTrace(), Stream()
    with trace.batch():
        unobserved = Event()
        trace.wait(stream, unobserved, reason=1)
        event = Event()
        trace.record(event, stream)
        del event
        gc.collect()
        replacement = Event()
        trace.record(replacement, stream)
    assert any("event=-1" in name and "record=-1" in name for name in scopes)
    assert trace.event_identity(replacement) == (2, 1)


def test_membership_is_ordered_opaque_and_prefill_budget_is_bounded(scopes):
    trace = PrefillTrace(max_batches=1)
    with trace.batch():
        trace.set_membership([dict(request_id="decode", is_prefill=False)])
    assert trace.capture_active()
    with trace.batch():
        trace.set_membership(
            [
                dict(
                    request_id="secret-b",
                    is_prefill=True,
                    computed_tokens=8,
                    scheduled_tokens=4,
                    row_start=0,
                    row_end=4,
                ),
                dict(
                    request_id="secret-a",
                    is_prefill=True,
                    computed_tokens=0,
                    scheduled_tokens=3,
                    row_start=4,
                    row_end=7,
                ),
            ]
        )
    assert not trace.capture_active()
    manifest = trace.manifest()
    assert manifest["batch_limit_reached"]
    assert manifest["observed_prefill_batches"] == 1
    assert not manifest["annotation_complete"]
    assert "secret" not in str(manifest)
    assert [item["request"] for item in manifest["membership"]] == [0, 1, 2]
    assert [item["batch"] for item in manifest["membership"]] == [1, 2, 2]


def test_suspension_excludes_decode_and_mtp_internal_scopes(scopes):
    trace = PrefillTrace()
    with trace.batch(), trace.scope("layer", layer=4, role=0):
        with trace.suspend(), trace.scope("attention_read", role=1):
            pass
        with trace.scope("attention_read"):
            pass
    reads = [name for name in scopes if "/attention_read " in name]
    assert len(reads) == 1 and "role=0" in reads[0] and "layer=4" in reads[0]


def test_exceptions_are_exported_and_context_is_restored(scopes):
    trace = PrefillTrace()
    with (
        pytest.raises(ValueError, match="enqueue failed"),
        trace.batch(),
        trace.scope("history_gather", layer=3),
    ):
        raise ValueError("enqueue failed")
    assert any("history_gather" in name and "exceptional=1" in name for name in scopes)
    assert not trace.active


def test_capacity_exhaustion_stops_annotations_without_stopping_execution(scopes):
    trace, stream = PrefillTrace(max_objects=1), Stream()
    first, second = Event(), Event()
    with trace.batch():
        trace.record(first, stream)
        trace.record(second, stream)
    assert trace.overflow
    assert first.records == second.records == [stream]


def make_state():
    plan = CKVPrefetchPlan.create(
        requested_depth=0,
        budget_bytes=0,
        dcp_world_size=2,
        local_capacity=4,
        record_bytes=8,
    )
    pool = CKVWorkspacePool(plan, torch.device("cpu"))
    registry = CKVPrefetchRegistry(pool)
    return registry, registry.for_workspace(pool.storage, lane=(0, 0), generation=7)


def test_slot_reuse_waits_exact_writer_and_consumer_without_extra_operations(scopes):
    trace, main = PrefillTrace(), Stream()
    registry, state = make_state()
    install_state_trace(state, trace)
    events = []
    with trace.batch(), trace.scope("layer", layer=0):
        for layer in (0, 1):
            state.prepare_gather(layer, main, producer_stream=main)
            writer = Event()
            trace.record(writer, main)
            state.finish_gather(layer, writer)
            state.consume(layer, main)
            reader = Event()
            trace.record(reader, main)
            state.finish_consumer(layer, reader)
            events.extend([writer, reader])
        registry.clear()
    assert main.events == [events[0], events[0], events[1], events[2]]
    assert main.streams == [main, main]
    assert [event.synchronizations for event in events] == [0, 0, 1, 1]
    acquisitions = [name for name in scopes if "/slot_acquire " in name]
    assert "use=1" in acquisitions[0] and "use=2" in acquisitions[1]
    assert all("lease=1" in name and "generation=7" in name for name in acquisitions)
    assert any("stream_wait" in name and "event=-1" in name for name in scopes)
    waits = [name for name in scopes if "/event_wait " in name]
    assert any("reason=1" in name for name in waits)
    assert any("reason=2" in name for name in waits)
    assert any("reason=3" in name for name in waits)


def test_rebinding_workspace_has_distinct_lease_and_cache_reset_is_labelled(scopes):
    trace = PrefillTrace()
    registry, state = make_state()
    with trace.batch():
        install_state_trace(state, trace)
        state.register_cache(0, torch.empty(1))
        state.register_cache(0, torch.empty(1))
        registry.clear()
        replacement = registry.for_workspace(
            registry.pool.storage, lane=(0, 0), generation=8
        )
        install_state_trace(replacement, trace)
        replacement.close()
    assert state._prefill_trace_identity["lease"] == 1
    assert replacement._prefill_trace_identity["lease"] == 2
    assert any("reset_caches_complete" in name for name in scopes)


def test_capture_exports_are_unique_rank_bound_and_reset_membership(tmp_path, scopes):
    import json
    from pathlib import Path

    trace = PrefillTrace()
    trace.register_layer(0, "model.layers.0.self_attn", 0)
    trace.register_layer(0, "draft.layers.0.self_attn", 1)
    paths = []
    for capture in range(2):
        trace.start_capture(rank=3, name="diagnosis-dcp1", device="cuda:3")
        with trace.batch([dict(request_id=f"opaque-{capture}", is_prefill=True)]):
            pass
        paths.append(trace.export_manifest(str(tmp_path)))
    assert paths[0] != paths[1]
    for capture, path in enumerate(paths, 1):
        data = json.loads(Path(path).read_text())
        assert data["capture"]["rank"] == 3
        assert data["capture"]["capture"] == capture
        assert len(data["membership"]) == 1
        assert len(data["layers"]) == 2
        assert not data["overflow"]
        assert "opaque-" not in str(data)


@pytest.mark.parametrize(
    "enabled,export_fails", [(False, False), (True, False), (True, True)]
)
def test_worker_profile_stop_exports_only_enabled_diagnosis(
    monkeypatch, tmp_path, enabled, export_fails
):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from vllm.v1.worker import gpu_worker

    worker = object.__new__(gpu_worker.Worker)
    worker.profiler_config = SimpleNamespace(
        profiler="torch", torch_profiler_dir=str(tmp_path)
    )
    worker.profiler = SimpleNamespace(stop=Mock())
    trace = SimpleNamespace(export_manifest=Mock(return_value="manifest.json"))
    if export_fails:
        trace.export_manifest.side_effect = OSError("read-only trace directory")
    worker._prefill_trace = trace if enabled else None
    errors = []
    monkeypatch.setattr(
        gpu_worker.logger, "exception", lambda *args: errors.append(args)
    )
    worker.profile(is_start=False)
    worker.profiler.stop.assert_called_once_with()
    assert trace.export_manifest.call_count == int(enabled)
    assert bool(errors) is export_fails


@pytest.mark.parametrize("fails", [False, True])
def test_actual_backend_gather_record_association_and_exception_fence(
    monkeypatch, scopes, fails
):
    from contextlib import nullcontext
    from types import SimpleNamespace

    from vllm.v1.attention.backends.mla.b12x_mla_sparse import B12xMLASparseImpl
    from vllm.v1.attention.backends.mla.prefill_diagnostics import install_backend_trace

    trace, main = PrefillTrace(), Stream()
    registry, state = make_state()
    registry.clear()
    impl = object.__new__(B12xMLASparseImpl)
    impl._ckv_reservation = SimpleNamespace(registry=registry, collectives_warmed=True)
    impl._kernel_page_size = 2
    impl._cache_record_bytes = 8
    impl._run = lambda _: None
    impl.do_kv_cache_update = lambda *args, **kwargs: None
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: main)
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "Event", Event)

    def gather(cache, metadata, local, output, **kwargs):
        if fails:
            raise RuntimeError("native enqueue failed")
        return output.zero_()

    impl._gather_full_ckv = gather
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_dcp_group",
        lambda: SimpleNamespace(ranks=[0, 1]),
    )
    install_backend_trace(impl, trace)
    cache = torch.empty((2, 2, 8), dtype=torch.uint8)
    with trace.batch(), trace.scope("layer", layer=0):
        if fails:
            with pytest.raises(RuntimeError, match="native enqueue failed"):
                impl._consume_ckv(
                    cache,
                    SimpleNamespace(),
                    SimpleNamespace(layer_name="model.layers.0.attn"),
                    cache,
                )
        else:
            _, state, layer = impl._consume_ckv(
                cache,
                SimpleNamespace(),
                SimpleNamespace(layer_name="model.layers.0.attn"),
                cache,
            )
            impl._run(None)
            event = Event()
            trace.record(event, main)
            state.finish_consumer(layer, event)
        registry.clear()
    records = [name for name in scopes if "/event_record " in name]
    associations = [name for name in scopes if "/finish_gather " in name]
    assert len(associations) == 1 and "event=1" in associations[0]
    assert "slot=0" in records[0] and "use=1" in records[0]
    assert any("exceptional=1" in name for name in scopes) is fails
    reads = [name for name in scopes if "/attention_read " in name]
    assert len(reads) == int(not fails)
    if not fails:
        assert "slot=0" in reads[0] and "lease=1" in reads[0]


def test_prefill_batch_limit_reports_truncation_boundary(scopes):
    trace = PrefillTrace(max_batches=64)
    for _ in range(64):
        with trace.batch([dict(request_id="opaque", is_prefill=True)]):
            pass
    assert not trace.capture_active()
    manifest = trace.manifest()
    assert manifest["batch_limit_reached"]
    assert manifest["observed_prefill_batches"] == 64
    assert manifest["observed_scheduler_executions"] == 64
    assert not manifest["annotation_complete"]


def test_startup_collectives_have_separate_domain_and_exact_event_edges(
    monkeypatch, scopes
):
    from contextlib import nullcontext
    from types import SimpleNamespace

    from vllm.v1.attention.backends.mla import b12x_mla_sparse as backend
    from vllm.v1.attention.backends.mla.prefill_diagnostics import install_backend_trace

    plan = CKVPrefetchPlan.create(
        requested_depth=1,
        budget_bytes=0,
        dcp_world_size=2,
        local_capacity=4,
        record_bytes=8,
    )
    pool = CKVWorkspacePool(plan, torch.device("cpu"))
    impl = object.__new__(backend.B12xMLASparseImpl)
    impl._ckv_reservation = backend._CKVReservation(
        CKVPrefetchRegistry(pool),
        torch.empty((1, 1, 2, 8), dtype=torch.uint8),
        torch.empty((1, 1, 4, 8), dtype=torch.uint8),
    )
    impl._kernel_page_size = 2
    impl._run = lambda _: None
    main, side = Stream(), Stream()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: main)
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "Stream", lambda **_: side)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    default, dedicated = SimpleNamespace(ranks=[0, 1]), SimpleNamespace(ranks=[0, 1])
    monkeypatch.setattr(backend, "get_dcp_group", lambda: default)
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_dcp_ckv_prefetch_group", lambda: dedicated
    )
    calls = []
    monkeypatch.setattr(
        backend, "_dcp_all_gather_current_stream", lambda *args: calls.append(args)
    )
    trace = PrefillTrace()
    install_backend_trace(impl, trace)
    impl.prepare_profile_collectives()
    assert len(calls) == 13
    assert main.synchronizations == 1
    assert side.streams == [main]
    assert len(main.events) == 1
    records = [name for name in scopes if "/event_record " in name]
    waits = [name for name in scopes if "/event_wait " in name]
    assert len(records) == len(waits) == 1
    assert "event=1" in records[0] and "event=1" in waits[0]
    assert all("batch=-1" in name and "domain=1" in name for name in scopes)
    assert trace.manifest()["observed_prefill_batches"] == 0


def test_native_stream_handle_survives_python_wrapper_replacement():
    from types import SimpleNamespace

    trace = PrefillTrace()
    first = trace.stream_identity(SimpleNamespace(cuda_stream=123))
    second = trace.stream_identity(SimpleNamespace(cuda_stream=123))
    other = trace.stream_identity(SimpleNamespace(cuda_stream=456))
    assert first == second
    assert other != first


def test_worker_capture_identity_matches_reused_torch_trace_name(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from vllm.v1.worker import gpu_worker

    worker = object.__new__(gpu_worker.Worker)
    worker.profiler_config = SimpleNamespace(
        profiler="torch", torch_profiler_dir=str(tmp_path)
    )
    worker.profiler = None
    worker.rank = worker.local_rank = 3
    worker.device = "cuda:3"
    worker._prefill_trace = PrefillTrace()
    monkeypatch.setattr(
        "vllm.distributed.utils.get_worker_rank_suffix", lambda **_: "rank3-dcp1"
    )
    monkeypatch.setattr(
        gpu_worker,
        "TorchProfilerWrapper",
        lambda *args, **kwargs: SimpleNamespace(start=Mock(), stop=Mock()),
    )
    worker.profile(is_start=True, profile_prefix="first")
    worker.profile(is_start=False)
    worker.profile(is_start=True, profile_prefix="second")
    capture = worker._prefill_trace.manifest()["capture"]
    assert capture["name"] == "first_rank3-dcp1"
    assert capture["capture"] == 2
