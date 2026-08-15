# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.distributed.device_communicators.custom_all_reduce import (
    CustomAllreduce,
    get_active_b12x_pcie_allreduce,
)


def test_accessor_accepts_active_runtime_without_fused_rms_norm(monkeypatch):
    custom_allreduce = object.__new__(CustomAllreduce)
    custom_allreduce.disabled = False
    custom_allreduce._pcie_runtime = object()
    group = SimpleNamespace(
        device_communicator=SimpleNamespace(ca_comm=custom_allreduce)
    )
    monkeypatch.setattr("vllm.distributed.parallel_state.get_tp_group", lambda: group)

    assert get_active_b12x_pcie_allreduce() is custom_allreduce


def test_accessor_rejects_disabled_runtime(monkeypatch):
    custom_allreduce = object.__new__(CustomAllreduce)
    custom_allreduce.disabled = True
    custom_allreduce._pcie_runtime = object()
    group = SimpleNamespace(
        device_communicator=SimpleNamespace(ca_comm=custom_allreduce)
    )
    monkeypatch.setattr("vllm.distributed.parallel_state.get_tp_group", lambda: group)

    assert get_active_b12x_pcie_allreduce() is None


def test_accessor_rejects_missing_runtime(monkeypatch):
    custom_allreduce = object.__new__(CustomAllreduce)
    custom_allreduce.disabled = False
    custom_allreduce._pcie_runtime = None
    custom_allreduce._pcie_dma = None
    custom_allreduce._ptr = 0
    group = SimpleNamespace(
        device_communicator=SimpleNamespace(ca_comm=custom_allreduce)
    )
    monkeypatch.setattr("vllm.distributed.parallel_state.get_tp_group", lambda: group)

    assert get_active_b12x_pcie_allreduce() is None
