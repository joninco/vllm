# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.models.kimi_k3.nvidia import tp_projection


def test_projection_group_uses_matching_dcp_coordinator(monkeypatch):
    tp_group = SimpleNamespace(world_size=8, ranks=list(range(8)))
    dcp_group = SimpleNamespace(world_size=8, ranks=list(range(8)))
    monkeypatch.setattr(
        tp_projection, "get_tensor_model_parallel_world_size", lambda: 8
    )
    monkeypatch.setattr(tp_projection, "get_tp_group", lambda: tp_group)
    monkeypatch.setattr(tp_projection, "get_dcp_group", lambda: dcp_group)

    assert tp_projection._get_kimi_projection_group() is dcp_group


def test_projection_group_uses_tp_for_different_dcp_ranks(monkeypatch):
    tp_group = SimpleNamespace(world_size=8, ranks=list(range(8)))
    dcp_group = SimpleNamespace(world_size=4, ranks=list(range(4)))
    monkeypatch.setattr(
        tp_projection, "get_tensor_model_parallel_world_size", lambda: 8
    )
    monkeypatch.setattr(tp_projection, "get_tp_group", lambda: tp_group)
    monkeypatch.setattr(tp_projection, "get_dcp_group", lambda: dcp_group)

    assert tp_projection._get_kimi_projection_group() is tp_group


def test_projection_group_rejects_incomplete_tp_coordinator(monkeypatch):
    monkeypatch.setattr(
        tp_projection, "get_tensor_model_parallel_world_size", lambda: 8
    )
    monkeypatch.setattr(
        tp_projection,
        "get_tp_group",
        lambda: SimpleNamespace(world_size=4, ranks=list(range(4))),
    )
    monkeypatch.setattr(
        tp_projection,
        "get_dcp_group",
        lambda: SimpleNamespace(world_size=4, ranks=list(range(4))),
    )

    with pytest.raises(RuntimeError, match="does not span"):
        tp_projection._get_kimi_projection_group()


@pytest.mark.skipif(torch.accelerator.device_count() < 1, reason="CUDA is required.")
def test_b12x_projection_gather_removes_each_rank_padding(monkeypatch):
    tp_size = 2
    local_width = 132
    local = torch.arange(local_width, dtype=torch.bfloat16, device="cuda").view(1, -1)
    group = SimpleNamespace(world_size=tp_size, ranks=list(range(tp_size)))
    received: dict[str, object] = {}
    monkeypatch.setattr(tp_projection.envs, "VLLM_USE_B12X_DCP_A2A", True)
    monkeypatch.setattr(
        tp_projection, "get_tensor_model_parallel_world_size", lambda: tp_size
    )
    monkeypatch.setattr(tp_projection, "_get_kimi_projection_group", lambda: group)

    def gather(transport, projection_group, *, max_batch_size):
        received.update(
            transport=transport,
            projection_group=projection_group,
            max_batch_size=max_batch_size,
        )
        padded_width = transport.shape[-1]
        result = torch.full(
            (1, tp_size, padded_width),
            -1,
            dtype=transport.dtype,
            device=transport.device,
        )
        result[0, 0, :local_width] = torch.arange(
            local_width, dtype=transport.dtype, device=transport.device
        )
        result[0, 1, :local_width] = torch.arange(
            local_width,
            2 * local_width,
            dtype=transport.dtype,
            device=transport.device,
        )
        return result

    monkeypatch.setattr(tp_projection, "dcp_b12x_all_gather_heads", gather)

    actual = tp_projection.gather_kimi_sharded_projection(local)

    expected = torch.arange(
        2 * local_width, dtype=local.dtype, device=local.device
    ).view(1, -1)
    torch.testing.assert_close(actual, expected)
    assert received["projection_group"] is group
    assert received["max_batch_size"] == 1
    assert received["transport"].shape == (1, 1, 136)


@pytest.mark.skipif(torch.accelerator.device_count() < 1, reason="CUDA is required.")
def test_b12x_projection_gather_preserves_fp32_payload_bits(monkeypatch):
    tp_size = 2
    local = torch.tensor([[1.25, -2.5]], dtype=torch.float32, device="cuda")
    other = torch.tensor([[3.75, -4.5]], dtype=torch.float32, device="cuda")
    group = SimpleNamespace(world_size=tp_size, ranks=list(range(tp_size)))
    received: dict[str, object] = {}
    monkeypatch.setattr(tp_projection.envs, "VLLM_USE_B12X_DCP_A2A", True)
    monkeypatch.setattr(
        tp_projection, "get_tensor_model_parallel_world_size", lambda: tp_size
    )
    monkeypatch.setattr(tp_projection, "_get_kimi_projection_group", lambda: group)

    def gather(transport, projection_group, *, max_batch_size):
        received["transport"] = transport
        result = torch.empty(
            (1, tp_size, transport.shape[-1]),
            dtype=transport.dtype,
            device=transport.device,
        )
        result[0, 0].copy_(transport[0, 0])
        result[0, 1].copy_(other.view(torch.float8_e4m3fn).flatten())
        return result

    monkeypatch.setattr(tp_projection, "dcp_b12x_all_gather_heads", gather)

    actual = tp_projection.gather_kimi_sharded_projection(local)

    torch.testing.assert_close(actual, torch.cat((local, other), dim=-1))
    assert actual.dtype == torch.float32
    assert received["transport"].dtype == torch.float8_e4m3fn
    assert received["transport"].shape == (1, 1, 8)


@pytest.mark.skipif(torch.accelerator.device_count() < 1, reason="CUDA is required.")
def test_b12x_projection_gather_preserves_fp8_payload(monkeypatch):
    tp_size = 2
    local = (
        torch.arange(16, dtype=torch.float32, device="cuda")
        .to(torch.float8_e4m3fn)
        .view(1, -1)
    )
    other = (
        torch.arange(16, 32, dtype=torch.float32, device="cuda")
        .to(torch.float8_e4m3fn)
        .view(1, -1)
    )
    group = SimpleNamespace(world_size=tp_size, ranks=list(range(tp_size)))
    monkeypatch.setattr(tp_projection.envs, "VLLM_USE_B12X_DCP_A2A", True)
    monkeypatch.setattr(
        tp_projection, "get_tensor_model_parallel_world_size", lambda: tp_size
    )
    monkeypatch.setattr(tp_projection, "_get_kimi_projection_group", lambda: group)

    def gather(transport, projection_group, *, max_batch_size):
        return torch.stack((transport[:, 0], other), dim=1)

    monkeypatch.setattr(tp_projection, "dcp_b12x_all_gather_heads", gather)

    actual = tp_projection.gather_kimi_sharded_projection(local)

    torch.testing.assert_close(
        actual.float(),
        torch.cat((local, other), dim=-1).float(),
    )
    assert actual.dtype == torch.float8_e4m3fn


def test_projection_gather_uses_standard_collective_outside_decode(monkeypatch):
    local = torch.arange(12).view(3, 4)
    expected = torch.cat((local, local), dim=-1)
    monkeypatch.setattr(
        tp_projection, "get_tensor_model_parallel_world_size", lambda: 2
    )
    monkeypatch.setattr(
        tp_projection,
        "tensor_model_parallel_all_gather",
        lambda output, dim: expected,
    )

    actual = tp_projection.gather_kimi_sharded_projection(local)

    assert actual is expected
