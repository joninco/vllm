# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Transport selection and independent ownership of captured DCP graphs."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.ops import b12x_dcp as module


def _config(**parallel_overrides):
    parallel = dict(
        decode_context_parallel_size=4,
        dcp_comm_backend="a2a",
        tensor_parallel_size=8,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        data_parallel_size=1,
        enable_dbo=False,
    )
    parallel.update(parallel_overrides)
    return SimpleNamespace(
        parallel_config=SimpleNamespace(**parallel),
        lora_config=None,
        model_config=SimpleNamespace(
            dtype=torch.bfloat16, architecture="GlmMoeDsaForCausalLM"
        ),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
            static_forward_context={},
        ),
        speculative_config=None,
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"decode_context_parallel_size": 1},
        {"decode_context_parallel_size": 2},
        {"dcp_comm_backend": "allgather"},
        {"tensor_parallel_size": 4},
        {"enable_dbo": True},
    ],
)
def test_ineligible_launch_does_not_allocate_or_initialize_collectives(
    monkeypatch, overrides
):
    def forbidden(*args, **kwargs):
        pytest.fail("Ineligible launch initialized DCP transport resources")

    monkeypatch.setattr(module, "get_dcp_group", forbidden)
    monkeypatch.setattr(torch, "empty", forbidden)
    assert module.initialize_b12x_dcp_transport(_config(**overrides), "cpu") is None


def test_direct_transport_diagnosis_switch_preserves_generic_dispatch(monkeypatch):
    monkeypatch.setenv("VLLM_USE_DIRECT_DCP_A2A", "0")
    monkeypatch.setattr(
        module, "get_dcp_group", lambda: pytest.fail("DCP registration")
    )
    assert module.initialize_b12x_dcp_transport(_config(), "cpu") is None


def test_unavailable_transport_on_any_rank_uses_generic_dispatch(monkeypatch):
    from vllm.v1.attention.backends.mla.b12x_mla_sparse import B12xMLASparseImpl

    config = _config()
    impl = object.__new__(B12xMLASparseImpl)
    impl.num_heads, impl._q_head_dim, impl.kv_lora_rank, impl._topk_tokens = (
        8,
        576,
        512,
        2048,
    )
    config.compilation_config.static_forward_context = {
        "attention": SimpleNamespace(impl=impl)
    }
    group = SimpleNamespace(world_size=4, cpu_group=object())
    monkeypatch.setattr(module, "get_dcp_group", lambda: group)

    def availability(output, value, *, group):
        output[:] = [True, False, True, True]

    monkeypatch.setattr(module.dist, "all_gather_object", availability)
    monkeypatch.setattr(
        module, "B12XDCPTransport", lambda *a: pytest.fail("Resource allocation")
    )
    assert module.initialize_b12x_dcp_transport(config, "cpu") is None


def test_target_and_draft_graphs_own_channels_before_capture(monkeypatch):
    events: list[tuple[str, int]] = []

    class Channel:
        allocated_bytes = 100
        slab_bytes = 200

        def __init__(self, **kwargs):
            self.identity = len(events)
            events.append(("allocate", self.identity))

        @contextmanager
        def capture(self):
            events.append(("capture", self.identity))
            yield

        def close(self):
            events.append(("close", self.identity))

    owner = module.B12XDCPTransport(
        SimpleNamespace(cpu_group=object()), "cpu", Channel, Channel
    )
    allocated_count = len(events)
    desc = SimpleNamespace(
        cg_mode=CUDAGraphMode.FULL,
        uniform_token_count=4,
        num_active_loras=0,
        num_tokens=16,
        num_reqs=4,
    )
    seen = []
    with monkeypatch.context() as patch:
        patch.setattr(
            torch, "empty", lambda *a, **kw: pytest.fail("Capture allocated storage")
        )
        for profiling in (True, False):
            for role in module.ROLES:
                manager = SimpleNamespace()
                owner.bind_graph_manager(manager, role, profiling=profiling)
                with manager.b12x_dcp_capture_scope(desc):
                    binding = module.active_dcp_transport()
                    assert binding is not None
                    seen.append(binding.attention)
                assert module.active_dcp_transport() is None
        assert len({id(channel) for channel in seen}) == 6
        with (
            pytest.raises(RuntimeError, match="already belongs"),
            owner.capture_scope("target", False, desc),
        ):
            pass
        desc.cg_mode = CUDAGraphMode.PIECEWISE
        with owner.capture_scope("target", False, desc):
            assert module.active_dcp_transport() is None
    assert len([event for event in events if event[0] == "allocate"]) == allocated_count
    assert owner.allocated_bytes > 0
    owner.close()
    assert len([event for event in events if event[0] == "close"]) == allocated_count
    assert owner.allocated_bytes == 0


def test_query_eligibility_requires_per_query_local_lengths():
    query = torch.zeros((4, 8, 576), dtype=torch.bfloat16)
    buffers = module._Buffers(
        torch.empty((16, 32, 576)), torch.empty(0), torch.empty(0)
    )
    binding = module._GraphTransport(None, None, buffers)
    lengths = torch.tensor([0, 1, 1, 2], dtype=torch.int32)
    assert binding.accepts(query, lengths)
    assert not binding.accepts(query, None)
    assert not binding.accepts(query, lengths[:1])
    assert not binding.accepts(query.float(), lengths)
