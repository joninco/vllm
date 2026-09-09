# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.v1.worker.memory_diagnostics import MemoryDiagnostics


@pytest.mark.parametrize("reset_peak", ["false", "true"])
def test_worker_memory_observation_preserves_bytes_and_pre_reset_peaks(
    monkeypatch, reset_peak
):
    stats = {"reserved_bytes.all.current": 2**33, "allocated_bytes.all.peak": 2**32}
    initial = SimpleNamespace(
        **{
            key: 2**34 + index
            for index, key in enumerate(
                (
                    "free_memory",
                    "total_memory",
                    "cuda_memory",
                    "torch_memory",
                    "torch_allocated",
                    "non_torch_memory",
                    "torch_peak",
                )
            )
        }
    )
    worker = SimpleNamespace(
        device=torch.device("cuda:3"),
        rank=7,
        requested_memory=99463834829,
        init_snapshot=initial,
    )
    get_info = Mock(return_value=(2**35, 2**36))
    get_stats = Mock(return_value=stats)
    reset = Mock(
        side_effect=lambda device: stats.update({"allocated_bytes.all.peak": 0})
    )
    monkeypatch.setattr(torch.accelerator, "get_memory_info", get_info)
    monkeypatch.setattr(torch.accelerator, "memory_stats", get_stats)
    monkeypatch.setattr(torch.accelerator, "reset_peak_memory_stats", reset)
    result = MemoryDiagnostics.capture_memory_diagnostics(worker, reset_peak)
    assert json.loads(json.dumps(result)) == result
    assert result["rank"] == 7
    assert result["requested_bytes"] == 99463834829
    assert result["initial_snapshot"] == vars(initial)
    assert result["allocator_stats"]["allocated_bytes.all.peak"] == 2**32
    assert result["device_used_minus_torch_reserved_bytes"] == 2**35 - 2**33
    get_info.assert_called_once_with(worker.device)
    get_stats.assert_called_once_with(worker.device)
    if reset_peak == "true":
        reset.assert_called_once_with(worker.device)
    else:
        reset.assert_not_called()


@pytest.mark.parametrize("invalid", [True, "yes", "", None])
def test_invalid_peak_reset_rejected_before_device_access(invalid):
    with pytest.raises(ValueError, match="reset_peak"):
        MemoryDiagnostics.capture_memory_diagnostics(object(), invalid)
