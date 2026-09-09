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
