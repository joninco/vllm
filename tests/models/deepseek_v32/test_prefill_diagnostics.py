# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from vllm.config import CUDAGraphMode
from vllm.models.deepseek_v32 import prefill_diagnostics as adapters
from vllm.v1.attention.backends.mla import prefill_diagnostics as diagnostics


class Trace:
    active = True

    def __init__(self):
        self.events = []

    def register_layer(self, index, name, role):
        self.registered = (index, name, role)

    @contextmanager
    def scope(self, name, **fields):
        if self.active:
            self.events.append((name, fields))
        yield

    @contextmanager
    def suspend(self):
        previous = self.active
        self.active = False
        try:
            yield
        finally:
            self.active = previous


def _layer(trace):
    decision = SimpleNamespace(route="projected_ag_rs", borrow_workspace=True)
    layer = SimpleNamespace(
        impl=SimpleNamespace(dcp_world_size=4),
        layer_name="model.layers.3.self_attn",
        _is_mtp_layer=False,
        _dcp_prefill_policy=SimpleNamespace(select=lambda batch: decision),
    )
    calls = []

    def producer(*args, **kwargs):
        calls.append(kwargs)
        return "native bytes"

    def forward():
        result = layer._fused_norm_rope(mla_kv_cache=object(), indexer_k_cache=object())
        assert layer._dcp_prefill_policy.select(None) is decision
        return result

    layer._fused_norm_rope = producer
    layer.forward = forward
    return layer, calls


@pytest.mark.parametrize(
    "excluded",
    [None, "decode", "mixed", "mtp", "verify", "capture", "ubatch", "missing"],
)
def test_layer_scope_and_actual_producer_policy(monkeypatch, excluded):
    trace = Trace()
    layer, calls = _layer(trace)
    metadata = SimpleNamespace(num_prefills=1, num_decode_tokens=0, num_decodes=0)
    context = SimpleNamespace(
        attn_metadata={layer.layer_name: metadata},
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )
    if excluded == "decode":
        metadata.num_prefills = 0
    elif excluded == "mixed":
        metadata.num_decode_tokens = 1
    elif excluded == "mtp":
        layer._is_mtp_layer = True
    elif excluded == "verify":
        metadata.is_spec_decode = True
    elif excluded == "capture":
        context.cudagraph_runtime_mode = CUDAGraphMode.FULL
    elif excluded == "ubatch":
        context.attn_metadata = [context.attn_metadata]
    elif excluded == "missing":
        context.attn_metadata = None
    monkeypatch.setattr(diagnostics, "get_prefill_trace", lambda: trace)
    monkeypatch.setattr(adapters, "get_forward_context", lambda: context)
    adapters.install_glm_prefill_diagnostics(layer)
    assert layer.forward() == "native bytes"
    assert len(calls) == 1
    assert trace.active
    if excluded:
        assert trace.events == []
    else:
        assert trace.events == [
            ("layer", {"layer": 3, "role": 0}),
            ("cache_produce", {"attention": 1, "indexer": 1}),
            ("dispatch", {"route": 5, "borrowed": 1}),
        ]


@pytest.mark.parametrize("enabled,dcp", [(False, 4), (True, 1)])
def test_disabled_installer_preserves_callables(monkeypatch, enabled, dcp):
    trace = Trace()
    layer, _ = _layer(trace)
    layer.impl.dcp_world_size = dcp
    original = dict(vars(layer))
    monkeypatch.setattr(
        diagnostics, "get_prefill_trace", lambda: trace if enabled else None
    )
    adapters.install_glm_prefill_diagnostics(layer)
    assert vars(layer) == original


def test_excluded_forward_exception_restores_trace(monkeypatch):
    trace = Trace()
    layer, _ = _layer(trace)
    layer._is_mtp_layer = True

    def fail():
        assert not trace.active
        raise ValueError("fixture MTP failure")

    layer.forward = fail
    monkeypatch.setattr(diagnostics, "get_prefill_trace", lambda: trace)
    monkeypatch.setattr(
        adapters,
        "get_forward_context",
        lambda: SimpleNamespace(
            attn_metadata=SimpleNamespace(num_prefills=1),
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
        ),
    )
    adapters.install_glm_prefill_diagnostics(layer)
    with pytest.raises(ValueError, match="fixture MTP failure"):
        layer.forward()
    assert trace.active
    assert not trace.events
