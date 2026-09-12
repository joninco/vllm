# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU storage and ordering contracts for eager projected DCP prefill.

Real workspace allocation and BF16 BMM are exercised with CPU collectives.
These tests do not validate NCCL, CUDA kernel precision or asynchronous joins.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.ops import dcp_prefill_workspace as module
from vllm.v1.worker.workspace import WorkspaceManager


@pytest.fixture
def workspace_manager(monkeypatch):
    manager = WorkspaceManager(torch.device("cpu"))
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(module, "current_workspace_manager", lambda: manager)
    return manager


def _plan(borrowed):
    return module.DCPPrefillWorkspace(
        (((9, 8, 64), torch.bfloat16), ((2048,), torch.uint8)),
        world_size=4,
        latent_dim=5,
        value_dim=3,
        borrowed=borrowed,
    )


@pytest.mark.parametrize("borrowed", [False, True])
@pytest.mark.parametrize("rows", [3, 7])
def test_reserved_storage_handles_query_chunks_and_pitched_projection(
    monkeypatch, workspace_manager, borrowed, rows
):
    plan = _plan(borrowed)
    plan.reserve()
    workspace_manager.lock()
    buffers = plan.borrow(rows)
    group = SimpleNamespace(world_size=4, rank_in_group=2, device_group=object())
    query = torch.arange(rows * 2 * 64).reshape(rows, 2, 64).to(torch.bfloat16)
    calls = []

    def gather(out, local, *, group):
        calls.append(tuple(local.shape))
        for rank in range(4):
            out.view(4, *local.shape)[rank].copy_(local + rank * 32)

    monkeypatch.setattr(module.dist, "all_gather_into_tensor", gather)
    gathered = buffers.gather_query((query[..., :5], query[..., 5:]), group)
    expected = torch.cat([query + rank * 32 for rank in range(4)], dim=1)
    torch.testing.assert_close(gathered, expected, atol=0, rtol=0)
    assert len(calls) == (rows + 1) // 2
    assert gathered.data_ptr() == buffers.query.data_ptr()
    assert (gathered.data_ptr() == buffers.backend_query.data_ptr()) is borrowed

    partial_storage = buffers.scratch[: 9 * 8 * 5 * 2].view(torch.bfloat16)
    partials = partial_storage.as_strided((rows, 8, 5), (5, 9 * 5, 1))
    generator = torch.Generator().manual_seed(301)
    partials.copy_(torch.randn(rows, 8, 5, generator=generator).to(torch.bfloat16))
    partials[0] = float("nan")
    original = partials.clone()
    lse = (
        buffers.scratch[9 * 8 * 5 * 2 : 9 * 8 * 5 * 2 + rows * 8 * 4]
        .view(torch.float32)
        .view(rows, 8)
    )
    lse.fill_(0)
    lengths = torch.ones(rows, dtype=torch.int32)
    lengths[0] = 0
    weights = torch.randn(2, 5, 3, generator=generator).to(torch.bfloat16)
    preparation = []

    def prepare(source, counts, out):
        assert source.data_ptr() == lse.data_ptr()
        out.copy_(torch.where(counts[:, None] > 0, source, -float("inf")))
        preparation.append(out.clone())

    def gather_weights(out, local, *, group):
        assert preparation
        for rank in range(4):
            out.view(4, *local.shape)[rank].copy_(local * (rank + 1))

    monkeypatch.setattr(module.dist, "all_gather_into_tensor", gather_weights)
    projected = buffers.project(partials, lse, lengths, weights, group, prepare)
    expected_weights = torch.cat([weights * (rank + 1) for rank in range(4)])
    expected_projected = torch.bmm(
        original.transpose(0, 1), expected_weights
    ).transpose(0, 1)
    torch.testing.assert_close(
        projected, expected_projected, equal_nan=True, atol=0, rtol=0
    )
    assert projected.stride() == (3, rows * 3, 1)
    assert projected.data_ptr() == buffers.projected.data_ptr()
    assert (projected.data_ptr() == buffers.scratch.data_ptr()) is borrowed
    torch.testing.assert_close(buffers.local_lse, preparation[0])

    def gather_lse(out, local, *, group):
        for rank in range(4):
            out.view(4, rows, 8)[rank].copy_(local)

    def correct(out, lses, rank, ctx, *, is_lse_base_on_e, lse_output):
        assert lse_output.data_ptr() == buffers.final_lse.data_ptr()
        assert rank == 2 and is_lse_base_on_e
        out[0].zero_()
        out.div_(4)
        lse_output.copy_(torch.logsumexp(lses, dim=0))
        return out, lse_output

    def reduce(out, local, *, group):
        assert local.is_contiguous()
        assert local.data_ptr() == buffers.projected.data_ptr()
        out.copy_(local[4:6] * 4)

    monkeypatch.setattr(module.dist, "all_gather_into_tensor", gather_lse)
    monkeypatch.setattr(module.dist, "reduce_scatter_tensor", reduce)
    result = buffers.combine(projected, group, correct, is_lse_base_on_e=True)
    reference = expected_projected[:, 4:6].clone()
    reference[0].zero_()
    torch.testing.assert_close(result, reference, atol=0, rtol=0)
    assert result.data_ptr() == buffers.result.data_ptr()
    assert result.stride() == (3, rows * 3, 1)
    assert (result.data_ptr() == buffers.backend_query.data_ptr()) is borrowed
    assert workspace_manager.is_locked()


def test_projection_rejects_output_outside_backend_scratch(workspace_manager):
    plan = _plan(True)
    plan.reserve()
    buffers = plan.borrow(3)
    with pytest.raises(ValueError, match="backend scratch"):
        buffers.project(
            torch.zeros(3, 8, 5, dtype=torch.bfloat16),
            torch.zeros(3, 8),
            torch.ones(3, dtype=torch.int32),
            torch.zeros(2, 5, 3, dtype=torch.bfloat16),
            None,
            None,
        )


def test_locked_workspace_rejects_unreserved_projection_capacity(workspace_manager):
    plan = _plan(True)
    plan.reserve()
    workspace_manager.lock()
    with pytest.raises(AssertionError, match="locked"):
        _plan(False).reserve()
    with pytest.raises(ValueError, match="capacity"):
        plan.borrow(10)


def test_projected_workspace_rejects_capture(monkeypatch, workspace_manager):
    plan = _plan(True)
    plan.reserve()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="eager-only"):
        plan.borrow(3)


@pytest.mark.parametrize("borrowed", [False, True])
@pytest.mark.parametrize("rows", [33, 64])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires one CUDA GPU")
def test_gpu_projected_merge_matches_float64_projection_reference(
    monkeypatch, borrowed, rows
):
    """Real BF16 projection and LSE kernels; peer collectives are simulated."""
    from b12x.comm.prefill import prepare_prefill_lse

    from vllm.v1.attention.ops.dcp import correct_attn_out

    capacity, heads, latent, value_dim = 64, 32, 512, 256
    manager = WorkspaceManager(torch.device("cuda"))
    monkeypatch.setattr(module, "current_workspace_manager", lambda: manager)
    scratch_bytes = capacity * heads * (latent * 2 + 4)
    plan = module.DCPPrefillWorkspace(
        (((capacity, heads, 576), torch.bfloat16), ((scratch_bytes,), torch.uint8)),
        world_size=4,
        latent_dim=latent,
        value_dim=value_dim,
        borrowed=borrowed,
    )
    plan.reserve()
    manager.lock()
    buffers = plan.borrow(rows)
    group = SimpleNamespace(world_size=4, rank_in_group=0, device_group=object())
    output_bytes = capacity * heads * latent * 2
    output = (
        buffers.scratch[:output_bytes]
        .view(torch.bfloat16)
        .as_strided((rows, heads, latent), (latent, capacity * latent, 1))
    )
    lse = (
        buffers.scratch[output_bytes:]
        .view(torch.float32)[: rows * heads]
        .view(rows, heads)
    )
    generator = torch.Generator().manual_seed(903)
    original = (torch.randn(rows, heads, latent, generator=generator) / 8).bfloat16()
    local_weights = (
        torch.randn(8, latent, value_dim, generator=generator) / latent**0.5
    ).bfloat16()
    expected = torch.einsum(
        "bhl,hlv->bhv", original.double(), local_weights.repeat(4, 1, 1).double()
    )
    expected[0].zero_()
    output.copy_(original)
    output[0] = float("nan")
    lse.fill_(0)
    lengths = torch.ones(rows, dtype=torch.int32, device="cuda")
    lengths[0] = 0

    def gather(out, local, *, group):
        for rank in range(4):
            out.view(4, *local.shape)[rank].copy_(local)

    def reduce(out, local, *, group):
        # Equal shard LSE gives weight 1/4; each simulated shard has the same
        # normalized latent output and uses the same gathered head weights.
        torch.testing.assert_close(
            local.transpose(0, 1).float().cpu(),
            expected.float() / 4,
            atol=0.002,
            rtol=0.012,
        )
        out.copy_(local[:8] * 4)

    monkeypatch.setattr(module.dist, "all_gather_into_tensor", gather)
    monkeypatch.setattr(module.dist, "reduce_scatter_tensor", reduce)
    projected = buffers.project(
        output, lse, lengths, local_weights.cuda(), group, prepare_prefill_lse
    )
    result = buffers.combine(projected, group, correct_attn_out, is_lse_base_on_e=True)
    torch.testing.assert_close(
        result.float().cpu(), expected[:, :8].float(), atol=0.008, rtol=0.012
    )
    assert result.data_ptr() == buffers.result.data_ptr()
    assert result.stride() == (value_dim, rows * value_dim, 1)


@pytest.mark.parametrize("borrowed", [False, True])
def test_startup_prewarms_selected_geometry_in_reserved_storage(
    monkeypatch, workspace_manager, borrowed
):
    import sys

    from vllm.v1.attention.ops import dcp

    plan = _plan(borrowed)
    plan.reserve()
    workspace_manager.lock()
    transport = dcp.MLADCPManager.__new__(dcp.MLADCPManager)
    transport.group = SimpleNamespace(
        world_size=4, rank_in_group=0, ranks=[0, 1, 2, 3], device_group=object()
    )
    transport.is_lse_base_on_e = True
    transport.prefill_workspaces = {borrowed: plan}
    calls = []

    def gather(out, local, *, group):
        calls.append(("gather", tuple(local.shape)))
        for rank in range(4):
            out.view(4, *local.shape)[rank].copy_(local)

    def reduce(out, local, *, group):
        calls.append(("reduce", tuple(local.shape)))
        out.copy_(local[:2])

    def prepare(source, counts, out):
        assert torch.all(counts == 1)
        out.copy_(source)

    def correct(out, lses, rank, ctx, *, is_lse_base_on_e, lse_output):
        lse_output.fill_(0)
        return out, lse_output

    monkeypatch.setitem(
        sys.modules, "b12x.comm.prefill", SimpleNamespace(prepare_prefill_lse=prepare)
    )
    monkeypatch.setattr(module.dist, "all_gather_into_tensor", gather)
    monkeypatch.setattr(module.dist, "reduce_scatter_tensor", reduce)
    monkeypatch.setattr(dcp, "correct_attn_out", correct)
    key = transport.prefill_warmup_key
    transport.prewarm_prefill(
        torch.ones(2, 5, 3, dtype=torch.bfloat16), backend_specs=plan.backend_specs
    )
    assert transport.prefill_warmup_record == {
        "group_ranks": (0, 1, 2, 3),
        "device": "cpu",
        "layouts": (
            (borrowed, 1, plan.reserved_bytes),
            (borrowed, 9, plan.reserved_bytes),
        ),
        "collective_calls": 8,
        "completed": True,
    }
    assert len(calls) == 8
    assert key == transport.prefill_warmup_key
    assert workspace_manager.is_locked()


def test_prefill_geometry_change_is_rejected_before_collectives(workspace_manager):
    plan = _plan(True)
    plan.reserve()
    changed = (((9, 8, 64), torch.bfloat16), ((4096,), torch.uint8))
    with pytest.raises(RuntimeError, match="geometry changed"):
        plan.borrow(3, backend_specs=changed)


def test_query_source_cannot_alias_gather_destination(workspace_manager):
    plan = _plan(True)
    plan.reserve()
    buffers = plan.borrow(3)
    with pytest.raises(ValueError, match="overlaps gather workspace"):
        buffers.gather_query(buffers.query[:3, :2], None)
