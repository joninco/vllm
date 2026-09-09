# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup-installed annotations for the GLM eager-prefill producer and route."""

from functools import wraps

from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context

_ROUTE_IDS = {
    "local": 0,
    "configured": 1,
    "full_ckv": 2,
    "a2a": 3,
    "ag_rs": 4,
    "projected_ag_rs": 5,
}


class _ObservedPolicy:
    def __init__(self, policy, trace):
        self.policy = policy
        self.trace = trace

    def select(self, batch):
        decision = self.policy.select(batch)
        with self.trace.scope(
            "dispatch",
            route=_ROUTE_IDS[decision.route],
            borrowed=int(decision.borrow_workspace),
        ):
            pass
        return decision


def install_glm_prefill_diagnostics(layer):
    """Leave disabled instances unchanged; annotate only eligible eager work."""
    from vllm.model_executor.models.utils import extract_layer_index
    from vllm.v1.attention.backends.mla.prefill_diagnostics import get_prefill_trace

    if layer.impl.dcp_world_size <= 1:
        return
    trace = get_prefill_trace()
    if trace is None:
        return
    try:
        index = extract_layer_index(layer.layer_name)
    except (AttributeError, ValueError, AssertionError, IndexError):
        index = -1
    trace.register_layer(
        index, layer.layer_name, -1 if index < 0 else int(layer._is_mtp_layer)
    )
    forward = layer.forward
    producer = layer._fused_norm_rope

    @wraps(forward)
    def observed_forward(*args, **kwargs):
        if not trace.active:
            return forward(*args, **kwargs)
        context = get_forward_context()
        batch_metadata = context.attn_metadata
        metadata = (
            batch_metadata.get(layer.layer_name)
            if isinstance(batch_metadata, dict)
            else batch_metadata
        )
        # A list requires a ubatch-specific adapter; do not assume list[0]
        # identifies the executing lane's metadata.
        eligible = (
            metadata is not None
            and not isinstance(metadata, list)
            and context.cudagraph_runtime_mode == CUDAGraphMode.NONE
            and not layer._is_mtp_layer
            and not getattr(metadata, "is_spec_decode", False)
            and getattr(metadata, "num_prefills", 0) > 0
            and getattr(metadata, "num_decode_tokens", 0) == 0
            and getattr(metadata, "num_decodes", 0) == 0
        )
        if not eligible:
            with trace.suspend():
                return forward(*args, **kwargs)
        with trace.scope("layer", layer=index, role=0 if index >= 0 else -1):
            return forward(*args, **kwargs)

    @wraps(producer)
    def observed_producer(*args, **kwargs):
        if not trace.active:
            return producer(*args, **kwargs)
        attention = kwargs.get("mla_kv_cache") is not None
        indexer = kwargs.get("indexer_k_cache") is not None
        if not attention and not indexer:
            return producer(*args, **kwargs)
        with trace.scope(
            "cache_produce", attention=int(attention), indexer=int(indexer)
        ):
            return producer(*args, **kwargs)

    layer.forward = observed_forward
    layer._fused_norm_rope = observed_producer
    if layer._dcp_prefill_policy is not None:
        layer._dcp_prefill_policy = _ObservedPolicy(layer._dcp_prefill_policy, trace)
