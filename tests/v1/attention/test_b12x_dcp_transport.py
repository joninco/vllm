# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Transport selection, launch-derived capacity and ownership of DCP graphs."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.ops import b12x_dcp as module
from vllm.v1.worker.gpu import cudagraph_utils as cgu
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    _sparse_full_capture_request_sizes,
)

# Production capture sizes: 1, 2, 4, 8, 16 and multiples of 8 up to 256.
CAPTURE_SIZES = [1, 2, 4, 8, 16, *range(24, 257, 8)]


def _speculative(num_tokens=3):
    return SimpleNamespace(
        method="mtp",
        num_speculative_tokens=num_tokens,
        uses_acceptance_length_adaptation=lambda: False,
        uses_batch_size_dynamic_speculative_decoding=lambda: False,
    )


def _config(*, max_num_seqs=16, speculative_tokens=3, **parallel_overrides):
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
            cudagraph_capture_sizes=list(CAPTURE_SIZES),
            max_cudagraph_capture_size=CAPTURE_SIZES[-1],
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        speculative_config=(
            _speculative(speculative_tokens) if speculative_tokens else None
        ),
        num_speculative_tokens=speculative_tokens,
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


def _eligible_config(**kwargs):
    from vllm.v1.attention.backends.mla.b12x_mla_sparse import B12xMLASparseImpl

    config = _config(**kwargs)
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
    return config


def test_unavailable_transport_on_any_rank_uses_generic_dispatch(monkeypatch):
    config = _eligible_config()
    group = SimpleNamespace(world_size=4, cpu_group=object())
    monkeypatch.setattr(module, "get_dcp_group", lambda: group)

    def availability(output, value, *, group):
        output[:] = [True, False, True, True]

    monkeypatch.setattr(module.dist, "all_gather_object", availability)
    monkeypatch.setattr(
        module, "B12XDCPTransport", lambda *a: pytest.fail("Resource allocation")
    )
    assert module.initialize_b12x_dcp_transport(config, "cpu") is None


def test_eligible_launch_passes_the_plan_to_the_transport(monkeypatch):
    import sys

    for name, cls in (
        ("pcie_dcp_attention", "PCIeDCPAttention"),
        ("pcie_dcp_topk_pull", "PCIeDCPTopKPull"),
    ):
        monkeypatch.setitem(
            sys.modules, f"b12x.comm.pcie.{name}", SimpleNamespace(**{cls: object()})
        )
    config = _eligible_config(max_num_seqs=16)
    group = SimpleNamespace(world_size=4, cpu_group=object())
    monkeypatch.setattr(module, "get_dcp_group", lambda: group)

    def availability(output, value, *, group):
        output[:] = [True] * 4

    monkeypatch.setattr(module.dist, "all_gather_object", availability)
    created = {}

    def transport(*args):
        created["args"] = args
        return "transport"

    monkeypatch.setattr(module, "B12XDCPTransport", transport)
    assert module.initialize_b12x_dcp_transport(config, "cpu") == "transport"
    plan = created["args"][-1]
    assert isinstance(plan, module.TransportPlan)
    assert plan.capacity == 64


@pytest.mark.parametrize(
    ("max_num_seqs", "speculative_tokens", "capacity", "expected"),
    [
        (
            16,
            3,
            64,
            {
                "target": (4, 8, 16, 24, 32, 40, 48, 56, 64),
                "draft_prefill": (4, 8, 16, 32, 64),
                "draft_decode": (1, 2, 4, 8, 16),
            },
        ),
        (
            64,
            0,
            64,
            {"target": (1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64)},
        ),
        (
            4,
            3,
            16,
            {
                "target": (4, 8, 16),
                "draft_prefill": (4, 8, 16),
                "draft_decode": (1, 2, 4),
            },
        ),
        (
            64,
            3,
            256,
            {
                "target": (4, 8, 16, *range(24, 257, 8)),
                "draft_prefill": (4, 8, 16, 32, 64, 128, 256),
                "draft_decode": (1, 2, 4, 8, 16, *range(24, 65, 8)),
            },
        ),
    ],
)
def test_plan_derives_capacity_and_graph_rows_from_the_launch(
    max_num_seqs, speculative_tokens, capacity, expected
):
    plan = module.plan_transport(
        _config(max_num_seqs=max_num_seqs, speculative_tokens=speculative_tokens)
    )
    assert plan.capacity == capacity
    assert dict(plan.rows_by_role) == expected
    assert plan.channel_count == 2 * sum(len(rows) for rows in expected.values())
    assert not hasattr(module, "MAX_ROWS")


def _real_manager(monkeypatch, config, decode_query_len, request_sizes=None):
    """Build a CudaGraphManager on CPU from the same configuration."""
    monkeypatch.setattr(
        cgu,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(
        cgu, "current_platform", SimpleNamespace(get_global_graph_pool=lambda: None)
    )
    monkeypatch.setattr(cgu, "is_breakable_cudagraph_enabled", lambda: False)
    vllm_config = SimpleNamespace(
        scheduler_config=config.scheduler_config,
        compilation_config=config.compilation_config,
        parallel_config=config.parallel_config,
        speculative_config=config.speculative_config,
        num_speculative_tokens=config.num_speculative_tokens,
    )
    return cgu.CudaGraphManager(
        vllm_config,
        torch.device("cpu"),
        CUDAGraphMode.FULL_AND_PIECEWISE,
        decode_query_len,
        full_capture_request_sizes=request_sizes,
    )


@pytest.mark.parametrize(
    ("max_num_seqs", "speculative_tokens"), [(16, 3), (64, 0), (64, 3)]
)
def test_plan_matches_the_graph_managers_full_decode_descriptors(
    monkeypatch, max_num_seqs, speculative_tokens
):
    config = _config(max_num_seqs=max_num_seqs, speculative_tokens=speculative_tokens)
    plan = module.plan_transport(config)
    query_len = speculative_tokens + 1
    managers = {"target": _real_manager(monkeypatch, config, query_len)}
    if speculative_tokens:
        managers["draft_prefill"] = _real_manager(
            monkeypatch,
            config,
            query_len,
            request_sizes=_sparse_full_capture_request_sizes(max_num_seqs),
        )
        managers["draft_decode"] = _real_manager(monkeypatch, config, 1)
    for role, manager in managers.items():
        assert manager.uniform_full_decode_token_counts() == plan.rows_by_role[role]
        for desc in manager._capture_descs[CUDAGraphMode.FULL]:
            if desc.uniform_token_count is not None:
                assert desc.num_tokens <= plan.capacity


class _Channel:
    """Fake channel recording the requested capacity and graph ownership."""

    events: list[tuple[str, int]] = []

    def __init__(self, *, max_rows, **kwargs):
        self.identity = len(_Channel.events)
        self.max_rows = max_rows
        self.allocated_bytes = 1000 * max_rows
        self.slab_bytes = 10 * max_rows
        _Channel.events.append(("allocate", self.identity))

    @contextmanager
    def capture(self):
        _Channel.events.append(("capture", self.identity))
        yield

    def close(self):
        _Channel.events.append(("close", self.identity))


def _plan(**rows_by_role):
    return module.TransportPlan(
        capacity=max(rows[-1] for rows in rows_by_role.values()),
        rows_by_role=rows_by_role,
    )


def _manager(rows):
    return SimpleNamespace(uniform_full_decode_token_counts=lambda: tuple(rows))


def _desc(num_tokens, num_reqs, cg_mode=CUDAGraphMode.FULL):
    return SimpleNamespace(
        cg_mode=cg_mode,
        uniform_token_count=num_tokens // num_reqs,
        num_active_loras=0,
        num_tokens=num_tokens,
        num_reqs=num_reqs,
    )


def test_channels_match_each_captured_graph_and_are_sized_per_graph(monkeypatch):
    _Channel.events = []
    plan = _plan(
        target=(4, 8, 16, 24, 32, 64),
        draft_prefill=(4, 8, 16, 32, 64),
        draft_decode=(1, 2, 4, 8, 16),
    )
    owner = module.B12XDCPTransport(
        SimpleNamespace(cpu_group=object()), "cpu", _Channel, _Channel, plan
    )
    assert len(owner._channels) == plan.channel_count == 32
    for (role, _profiling, rows), binding in owner._channels.items():
        assert binding.attention.max_rows == rows
        assert binding.candidates.max_rows == rows
        assert binding.max_rows == rows
        assert binding.buffers.query.shape == (plan.rows_by_role[role][-1], 32, 576)
    channel_bytes = sum(
        2 * (1000 * rows + 10 * rows)
        for rows_list in plan.rows_by_role.values()
        for rows in rows_list
    )
    buffer_bytes = sum(
        rows_list[-1] * (32 * 576 * 2 + 8 * 512 * 2 + 32 * 4)
        for rows_list in plan.rows_by_role.values()
    )
    assert owner.allocated_bytes == channel_bytes + buffer_bytes
    assert sum(owner.role_bytes.values()) == owner.allocated_bytes
    assert set(owner.role_bytes) == {"target", "draft_prefill", "draft_decode"}
    owner.close()
    assert owner.allocated_bytes == 0 and owner.role_bytes == {}
    # Each channel owns an attention exchange and a candidate exchange.
    assert len([e for e in _Channel.events if e[0] == "close"]) == 2 * 32


@pytest.mark.parametrize(
    ("role", "num_tokens", "num_reqs"),
    [
        ("target", 24, 6),
        ("target", 32, 8),
        ("target", 64, 16),
        ("draft_prefill", 32, 8),
        ("draft_prefill", 64, 16),
        ("draft_decode", 24, 24),
        ("draft_decode", 32, 32),
        ("draft_decode", 64, 64),
    ],
)
def test_capture_scope_binds_the_channel_of_the_exchanged_row_count(
    monkeypatch, role, num_tokens, num_reqs
):
    _Channel.events = []
    plan = _plan(
        target=(4, 8, 16, 24, 32, 64),
        draft_prefill=(4, 8, 16, 32, 64),
        draft_decode=(1, 2, 4, 8, 16, 24, 32, 64),
    )
    owner = module.B12XDCPTransport(
        SimpleNamespace(cpu_group=object()), "cpu", _Channel, _Channel, plan
    )
    query = torch.zeros((num_tokens, 8, 576), dtype=torch.bfloat16)
    lengths = torch.ones(num_tokens, dtype=torch.int32)
    for profiling in (True, False):
        manager = _manager(plan.rows_by_role[role])
        owner.bind_graph_manager(manager, role, profiling=profiling)
        with manager.b12x_dcp_capture_scope(_desc(num_tokens, num_reqs)):
            binding = module.active_dcp_transport()
            assert binding is not None
            assert binding.max_rows == num_tokens
            assert binding.attention.max_rows == num_tokens
            assert binding.accepts(query, lengths)
            assert not binding.accepts(
                torch.zeros((num_tokens + 1, 8, 576), dtype=torch.bfloat16),
                torch.ones(num_tokens + 1, dtype=torch.int32),
            )
        assert module.active_dcp_transport() is None
    with (
        pytest.raises(RuntimeError, match="already belongs"),
        owner.capture_scope(role, False, _desc(num_tokens, num_reqs)),
    ):
        pass
    owner.close()


def test_profiling_and_serving_graphs_own_distinct_channels(monkeypatch):
    _Channel.events = []
    plan = _plan(target=(4, 16), draft_prefill=(4, 16), draft_decode=(1, 4))
    owner = module.B12XDCPTransport(
        SimpleNamespace(cpu_group=object()), "cpu", _Channel, _Channel, plan
    )
    allocated = len(_Channel.events)
    seen = []
    with monkeypatch.context() as patch:
        patch.setattr(
            torch, "empty", lambda *a, **kw: pytest.fail("Capture allocated storage")
        )
        for profiling in (True, False):
            for role in module.ROLES:
                manager = _manager(plan.rows_by_role[role])
                owner.bind_graph_manager(manager, role, profiling=profiling)
                with manager.b12x_dcp_capture_scope(_desc(4, 1)):
                    seen.append(module.active_dcp_transport().attention)
        assert len({id(channel) for channel in seen}) == 6
        with owner.capture_scope("target", False, _desc(4, 1, CUDAGraphMode.PIECEWISE)):
            assert module.active_dcp_transport() is None
        with owner.capture_scope("target", False, _desc(8, 2)):
            assert module.active_dcp_transport() is None
    assert len([e for e in _Channel.events if e[0] == "allocate"]) == allocated
    owner.close()
    assert len([e for e in _Channel.events if e[0] == "close"]) == allocated


def test_binding_rejects_a_manager_whose_graph_set_differs_from_the_plan():
    _Channel.events = []
    plan = _plan(target=(4, 8, 16))
    owner = module.B12XDCPTransport(
        SimpleNamespace(cpu_group=object()), "cpu", _Channel, _Channel, plan
    )
    with pytest.raises(RuntimeError, match="planned rows \\[4, 8, 16\\]"):
        owner.bind_graph_manager(_manager((4, 8, 16, 32)), "target", profiling=False)
    with pytest.raises(RuntimeError, match="draft_decode"):
        owner.bind_graph_manager(_manager((1, 2)), "draft_decode", profiling=False)
    owner.bind_graph_manager(_manager(()), "draft_decode", profiling=False)
    with pytest.raises(ValueError, match="Unknown DCP graph role"):
        owner.bind_graph_manager(_manager(()), "verifier", profiling=False)
    owner.close()


def test_query_eligibility_requires_per_query_local_lengths():
    query = torch.zeros((4, 8, 576), dtype=torch.bfloat16)
    buffers = module._Buffers(
        torch.empty((16, 32, 576)), torch.empty(0), torch.empty(0)
    )
    binding = module._GraphTransport(None, None, buffers, 16)
    lengths = torch.tensor([0, 1, 1, 2], dtype=torch.int32)
    assert binding.accepts(query, lengths)
    assert not binding.accepts(query, None)
    assert not binding.accepts(query, lengths[:1])
    assert not binding.accepts(query.float(), lengths)
    wide = torch.zeros((17, 8, 576), dtype=torch.bfloat16)
    assert not binding.accepts(wide, torch.ones(17, dtype=torch.int32))
    assert not binding.merge_candidates(torch.zeros((17, 2048, 2)), None)
