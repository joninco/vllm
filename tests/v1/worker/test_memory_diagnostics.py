# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gzip
import hashlib
import json
from pathlib import Path
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


def test_stopped_cpu_profiler_exports_distinct_raw_memory_timelines(tmp_path):
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as profiler:
        values = torch.ones(32)
        values = values.square()
    worker = SimpleNamespace(
        rank=3,
        device=torch.device("cpu"),
        profiler=SimpleNamespace(is_running=False, profiler=profiler),
        profiler_config=SimpleNamespace(
            profiler="torch",
            torch_profiler_with_memory=True,
            torch_profiler_with_stack=True,
            torch_profiler_record_shapes=True,
            torch_profiler_dir=str(tmp_path),
        ),
    )
    observations = [
        MemoryDiagnostics.export_memory_diagnostics_timeline(worker) for _ in range(2)
    ]
    assert observations[0]["path"] != observations[1]["path"]
    for result in observations:
        path = Path(result["path"])
        assert path.parent.parent == tmp_path
        assert result["rank"] == 3
        assert result["device"] == "cpu"
        assert result["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        with gzip.open(path, "rt") as stream:
            events = json.load(stream)
        assert events and all(len(event) == 4 for event in events)
        assert any(
            event[2] == values.numel() * values.element_size() for event in events
        )


@pytest.mark.parametrize("fault", ["running", "memory", "stack", "shapes", "missing"])
def test_memory_timeline_rejects_incomplete_profiler_before_file_creation(
    tmp_path, fault
):
    config = SimpleNamespace(
        profiler="torch",
        torch_profiler_with_memory=fault != "memory",
        torch_profiler_with_stack=fault != "stack",
        torch_profiler_record_shapes=fault != "shapes",
        torch_profiler_dir=str(tmp_path),
    )
    wrapper = (
        None if fault == "missing" else SimpleNamespace(is_running=fault == "running")
    )
    worker = SimpleNamespace(profiler_config=config, profiler=wrapper)
    with pytest.raises(RuntimeError, match="stopped torch profiler"):
        MemoryDiagnostics.export_memory_diagnostics_timeline(worker)
    assert not list(tmp_path.iterdir())
