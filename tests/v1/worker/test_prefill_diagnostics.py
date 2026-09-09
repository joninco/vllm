# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from vllm.v1.attention.backends.mla import prefill_diagnostics as diagnostics
from vllm.v1.worker.prefill_diagnostics import install_prefill_runner_diagnostics


class Trace:
    active = False
    enabled = True

    def __init__(self):
        self.records = []
        self.events = []

    def capture_active(self):
        return self.enabled

    @contextmanager
    def batch(self):
        self.active = True
        self.events.append("begin")
        try:
            yield
        finally:
            self.events.append("end")
            self.active = False

    def set_membership(self, records):
        self.records.extend(records)
        self.events.append("membership")


@pytest.mark.parametrize("request_state", [False, True])
@pytest.mark.parametrize("enabled,dcp", [(False, 4), (True, 1), (True, 4)])
def test_runner_install_preserves_disabled_callables(
    monkeypatch, request_state, enabled, dcp
):
    trace = Trace()
    monkeypatch.setattr(
        diagnostics, "get_prefill_trace", lambda: trace if enabled else None
    )
    execute = lambda: None
    prepare = lambda: None
    runner = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=dcp),
        execute_model=execute,
        prepare_inputs=prepare,
        _prepare_inputs=prepare,
    )
    install_prefill_runner_diagnostics(runner, request_state_runner=request_state)
    selected = runner.prepare_inputs if request_state else runner._prepare_inputs
    assert (runner.execute_model is execute) == (not enabled or dcp == 1)
    assert (selected is prepare) == (not enabled or dcp == 1)
    trace.enabled = False
    runner.execute_model()
    assert trace.events == []


@pytest.mark.parametrize("request_state", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_final_request_order_and_exception_scope(monkeypatch, request_state, fail):
    trace = Trace()
    monkeypatch.setattr(diagnostics, "get_prefill_trace", lambda: trace)
    runner = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=4),
        use_async_scheduling=True,
    )
    scheduler = SimpleNamespace(num_scheduled_tokens={"first": 1, "second": 5})
    batch = SimpleNamespace(
        req_ids=["second", "first"],
        num_computed_tokens_cpu=[11, 20],
        num_prompt_tokens=[30, 20],
        query_start_loc_np=[0, 5, 6],
        num_computed_tokens_np=[11, 20],
        num_scheduled_tokens=[5, 1],
        is_prefilling_np=[True, False],
    )

    def prepare(*args):
        runner.input_batch = batch
        return batch if request_state else "prepared"

    def execute(scheduler):
        method = runner.prepare_inputs if request_state else runner._prepare_inputs
        result = method(scheduler)
        trace.events.append("forward")
        if fail:
            raise ValueError("fixture forward failure")
        return result

    runner.prepare_inputs = runner._prepare_inputs = prepare
    runner.execute_model = execute
    install_prefill_runner_diagnostics(runner, request_state_runner=request_state)
    if fail:
        with pytest.raises(ValueError, match="fixture forward failure"):
            runner.execute_model(scheduler)
    else:
        assert runner.execute_model(scheduler) == (
            batch if request_state else "prepared"
        )
    assert trace.events == ["begin", "membership", "forward", "end"]
    assert not trace.active
    assert [r["request_id"] for r in trace.records] == ["second", "first"]
    assert [(r["row_start"], r["row_end"]) for r in trace.records] == [(0, 5), (5, 6)]
    assert [r["is_prefill"] for r in trace.records] == [True, False]
    assert all(r["offsets_exact"] == 0 for r in trace.records)


def test_real_trace_exports_opaque_membership_and_ignores_dummy(monkeypatch):
    import torch

    trace = diagnostics.PrefillTrace(max_batches=2)
    monkeypatch.setattr(trace, "capture_active", lambda: True)
    names = []

    @contextmanager
    def record(name):
        names.append(name)
        yield

    monkeypatch.setattr(torch.profiler, "record_function", record)
    monkeypatch.setattr(diagnostics, "get_prefill_trace", lambda: trace)
    batch = SimpleNamespace(
        req_ids=["private-request"],
        query_start_loc_np=[0, 4],
        num_computed_tokens_np=[12],
        num_scheduled_tokens=[4],
        is_prefilling_np=[True],
    )
    runner = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=4),
        prepare_inputs=lambda scheduler: batch,
    )

    def execute(scheduler, intermediate_tensors=None, dummy_run=False):
        if not dummy_run:
            runner.prepare_inputs(scheduler)
        return "output"

    runner.execute_model = execute
    install_prefill_runner_diagnostics(runner, request_state_runner=True)
    assert runner.execute_model(None, dummy_run=True) == "output"
    assert names == []
    assert runner.execute_model(None) == "output"
    assert not trace.active
    manifest = trace.manifest()
    assert "private-request" not in str(manifest)
    assert "computed_tokens" in str(manifest)
    assert names[0].startswith("dcp_prefill.v1/batch ")
    assert any(name.startswith("dcp_prefill.v1/membership ") for name in names)
    assert all("private-request" not in name for name in names)


@pytest.mark.parametrize("request_state", [False, True])
@pytest.mark.parametrize("sample_fails", [False, True])
def test_real_ticket_follows_sample_and_actual_proposer(
    monkeypatch, request_state, sample_fails
):
    import torch

    trace = diagnostics.PrefillTrace(max_batches=1)
    monkeypatch.setattr(torch.autograd.profiler, "_is_profiler_enabled", True)
    names = []

    @contextmanager
    def record(name):
        names.append(name)
        yield

    monkeypatch.setattr(torch.profiler, "record_function", record)
    monkeypatch.setattr(diagnostics, "get_prefill_trace", lambda: trace)
    runner = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=4),
        execute_model_state=None,
        use_async_scheduling=False,
    )
    batch = SimpleNamespace(
        req_ids=["request"],
        query_start_loc_np=[0, 4],
        num_computed_tokens_np=[0],
        num_scheduled_tokens=[4],
        is_prefilling_np=[True],
        num_computed_tokens_cpu=[0],
        num_prompt_tokens=[4],
    )
    runner.prepare_inputs = lambda _: batch

    def persistent_prepare(*args):
        runner.input_batch = batch

    runner._prepare_inputs = persistent_prepare
    observed = []

    def propose():
        observed.append(dict(trace.current_fields))
        assert not trace.ownership_active
        return "draft"

    setattr(
        runner,
        "speculator" if request_state else "drafter",
        SimpleNamespace(propose=propose),
    )

    def execute(scheduler):
        (runner.prepare_inputs if request_state else runner._prepare_inputs)(scheduler)
        runner.execute_model_state = tuple([object(), object()])

    def sample():
        runner.execute_model_state = None
        owner = runner.speculator if request_state else runner.drafter
        assert owner.propose() == "draft"
        if sample_fails:
            raise ValueError("sample failure")
        return "sample"

    runner.execute_model, runner.sample_tokens = execute, sample
    runner.shutdown = lambda: None
    install_prefill_runner_diagnostics(runner, request_state_runner=request_state)
    runner.execute_model(SimpleNamespace(num_scheduled_tokens={"request": 4}))
    assert not trace.active and not trace.capture_active()
    if sample_fails:
        with pytest.raises(ValueError, match="sample failure"):
            runner.sample_tokens()
    else:
        assert runner.sample_tokens() == "sample"
    assert observed[0]["batch"] == 1
    assert observed[0]["owner_role"] == 1
    assert observed[0]["ownership_enabled"] == 0
    assert not {"lease", "slot", "use", "event"} & observed[0].keys()
    assert trace.manifest()["tickets"][0]["disposition"] == (
        "sample_exception" if sample_fails else "complete"
    )
    assert not trace.active


@pytest.mark.parametrize("dcp,enabled", [(1, True), (4, False)])
def test_disabled_diagnostics_preserve_sample_and_proposer_callables(
    monkeypatch, dcp, enabled
):
    trace = diagnostics.PrefillTrace()
    monkeypatch.setattr(
        diagnostics, "get_prefill_trace", lambda: trace if enabled else None
    )
    sample, propose = lambda: None, lambda: None
    runner = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=dcp),
        execute_model=lambda: None,
        _prepare_inputs=lambda: None,
        sample_tokens=sample,
        drafter=SimpleNamespace(propose=propose),
    )
    install_prefill_runner_diagnostics(runner)
    assert runner.sample_tokens is sample
    assert runner.drafter.propose is propose


def test_runner_reset_and_execution_choice_are_boundary_only(monkeypatch):
    import torch

    from vllm.config import CUDAGraphMode

    trace = diagnostics.PrefillTrace()
    monkeypatch.setattr(torch.autograd.profiler, "_is_profiler_enabled", True)
    monkeypatch.setattr(diagnostics, "get_prefill_trace", lambda: trace)
    names = []

    @contextmanager
    def record(name):
        names.append(name)
        yield

    monkeypatch.setattr(torch.profiler, "record_function", record)
    runner = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=4),
        execute_model_state=None,
        _prepare_inputs=lambda *args: None,
        sample_tokens=lambda: None,
        shutdown=lambda: "shutdown",
        _determine_batch_execution_and_padding=lambda: (CUDAGraphMode.FULL, 16),
    )

    def execute():
        assert runner._determine_batch_execution_and_padding() == (
            CUDAGraphMode.FULL,
            16,
        )
        runner.execute_model_state = object()

    runner.execute_model = execute
    install_prefill_runner_diagnostics(runner)
    runner.execute_model()
    assert runner.shutdown() == "shutdown"
    assert trace.manifest()["tickets"][0]["disposition"] == "runner_reset"
    assert any("/execution_choice " in name for name in names)
    assert not any(
        "graph_replay" in name or "attention_dispatch" in name for name in names
    )
