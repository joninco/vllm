# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from vllm.config import CUDAGraphMode
from vllm.models.deepseek_v32 import prefill_diagnostics as adapters
from vllm.v1.attention.backends.mla import prefill_diagnostics as diagnostics
from vllm.v1.attention.ops.dcp_prefill_policy import DCPPrefillBatch, DCPPrefillPolicy


@pytest.fixture
def trace(monkeypatch):
    trace = diagnostics.PrefillTrace()
    monkeypatch.setattr(trace, "capture_active", lambda: True)
    events = []

    @contextmanager
    def record(name):
        operation, *fields = name.split()
        events.append(
            (
                operation.removeprefix("dcp_prefill.v1/"),
                {
                    key: int(value)
                    for key, value in (field.split("=") for field in fields)
                },
            )
        )
        yield

    monkeypatch.setattr(torch.profiler, "record_function", record)
    monkeypatch.setattr(diagnostics, "get_prefill_trace", lambda: trace)
    return trace, events


def layer_fixture(
    monkeypatch,
    *,
    mtp=False,
    verification=False,
    mixed=False,
    decode=False,
    mode=CUDAGraphMode.NONE,
):
    metadata = SimpleNamespace(
        num_prefills=0 if decode else 1,
        num_decodes=int(mixed or decode),
        num_decode_tokens=int(mixed or decode),
        is_spec_decode=verification,
    )
    policy = DCPPrefillPolicy(
        dcp_world_size=4,
        max_num_tokens=8192,
        enabled=True,
        base_backend="ag_rs",
        project_before_merge=True,
        project_min_tokens=32,
    )
    calls = []

    def select(batch):
        decision = policy.select(batch)
        calls.append((batch, decision))
        return decision

    layer = SimpleNamespace(
        impl=SimpleNamespace(dcp_world_size=4),
        layer_name="model.layers.3.self_attn",
        _is_mtp_layer=mtp,
        _dcp_prefill_policy=SimpleNamespace(select=select),
    )
    context = SimpleNamespace(
        attn_metadata={layer.layer_name: metadata}, cudagraph_runtime_mode=mode
    )
    monkeypatch.setattr(adapters, "get_forward_context", lambda: context)
    producers = []

    def producer(**kwargs):
        producers.append(kwargs)
        return "native bytes"

    def forward():
        result = layer._fused_norm_rope(mla_kv_cache=object(), indexer_k_cache=object())
        decision = layer._dcp_prefill_policy.select(
            DCPPrefillBatch(
                num_tokens=64,
                num_prefills=metadata.num_prefills,
                num_decodes=metadata.num_decodes,
                is_mtp=mtp or verification,
                is_capturing=mode != CUDAGraphMode.NONE,
            )
        )
        assert decision is calls[-1][1]
        return result

    layer.forward, layer._fused_norm_rope = forward, producer
    return layer, context, calls, producers


@pytest.mark.parametrize(
    "kind", ["prefill", "mixed", "target-verification", "drafter", "decode"]
)
def test_actual_policy_dispatch_and_independent_roles(monkeypatch, trace, kind):
    ledger, events = trace
    layer, _, decisions, producers = layer_fixture(
        monkeypatch,
        mtp=kind == "drafter",
        verification=kind == "target-verification",
        mixed=kind == "mixed",
        decode=kind == "decode",
    )
    adapters.install_glm_prefill_diagnostics(layer)
    owner_role = int(kind == "drafter")
    with ledger.batch():
        ledger.set_owner(owner_role=owner_role, ticket=17, lease=91, slot=2, use=5)
        before = dict(ledger.current_fields)
        assert layer.forward() == "native bytes"
        assert ledger.current_fields == before
    assert len(producers) == len(decisions) == 1
    fields = next(fields for name, fields in events if name == "layer")
    dispatch = next(fields for name, fields in events if name == "dispatch")
    assert fields["batch"] == dispatch["batch"]
    assert fields["ticket"] == dispatch["ticket"] == 17
    assert fields["owner_role"] == owner_role
    assert fields["role"] == int(kind == "drafter")
    assert fields["verification"] == int(kind == "target-verification")
    assert dispatch["is_mtp"] == int(kind in {"drafter", "target-verification"})
    assert dispatch["num_tokens"] == 64
    if kind == "prefill":
        assert fields["ownership_enabled"] == 1
        assert dispatch["route"] == 5
        assert any(name == "cache_produce" for name, _ in events)
    else:
        assert fields["ownership_enabled"] == 0
        assert dispatch["route"] == 1
        assert not any(name == "cache_produce" for name, _ in events)
        assert not {"lease", "slot", "use"} & fields.keys()
        assert not {"lease", "slot", "use"} & dispatch.keys()
    assert not ledger.active


@pytest.mark.parametrize("excluded", ["capture", "piecewise", "ubatch", "missing"])
def test_graph_and_unknown_metadata_do_not_infer_layer_dispatch(
    monkeypatch, trace, excluded
):
    ledger, events = trace
    mode = {"capture": CUDAGraphMode.FULL, "piecewise": CUDAGraphMode.PIECEWISE}.get(
        excluded, CUDAGraphMode.NONE
    )
    layer, context, decisions, producers = layer_fixture(monkeypatch, mode=mode)
    if excluded == "ubatch":
        context.attn_metadata = [context.attn_metadata]
    elif excluded == "missing":
        context.attn_metadata = None
    adapters.install_glm_prefill_diagnostics(layer)
    with ledger.batch():
        assert layer.forward() == "native bytes"
    assert len(producers) == len(decisions) == 1
    assert [name for name, _ in events] == ["batch"]


def test_proposer_owner_does_not_relabel_target_or_enable_ownership(monkeypatch, trace):
    ledger, events = trace
    layer, _, _, _ = layer_fixture(monkeypatch)
    adapters.install_glm_prefill_diagnostics(layer)
    with ledger.batch():
        ledger.set_owner(owner_role=1, ownership_enabled=0, ticket=18)
        layer.forward()
    fields = next(fields for name, fields in events if name == "layer")
    assert fields["owner_role"] == 1
    assert fields["role"] == 0
    assert fields["verification"] == 0
    assert fields["ownership_enabled"] == 0
    assert not any(name == "cache_produce" for name, _ in events)


@pytest.mark.parametrize("enabled,dcp", [(False, 4), (True, 1)])
def test_disabled_installer_preserves_callables(monkeypatch, trace, enabled, dcp):
    ledger, _ = trace
    layer, _, _, _ = layer_fixture(monkeypatch)
    layer.impl.dcp_world_size = dcp
    original = dict(vars(layer))
    monkeypatch.setattr(
        diagnostics, "get_prefill_trace", lambda: ledger if enabled else None
    )
    adapters.install_glm_prefill_diagnostics(layer)
    assert vars(layer) == original


@pytest.mark.parametrize("graph", [False, True])
def test_forward_exception_restores_ticket_context(monkeypatch, trace, graph):
    ledger, events = trace
    mode = CUDAGraphMode.FULL if graph else CUDAGraphMode.NONE
    layer, _, _, _ = layer_fixture(monkeypatch, mtp=True, mode=mode)

    def fail():
        assert not ledger.ownership_active
        raise ValueError("fixture MTP failure")

    layer.forward = fail
    adapters.install_glm_prefill_diagnostics(layer)
    with ledger.batch():
        ledger.set_owner(owner_role=1, ticket=19, ownership_enabled=0)
        before = dict(ledger.current_fields)
        with pytest.raises(ValueError, match="fixture MTP failure"):
            layer.forward()
        assert ledger.current_fields == before
    assert not ledger.active
    assert not any(name in {"dispatch", "cache_produce"} for name, _ in events)
