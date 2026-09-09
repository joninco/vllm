# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit worker extension for allocator accounting on diagnosis launches.

Select with ``--worker-extension-cls
vllm.v1.worker.memory_diagnostics.MemoryDiagnostics``. Call
``capture_memory_diagnostics`` through collective RPC only after requests drain.
The extension does not load by default or allocate GPU tensors. Development
HTTP RPC, when used, requires a diagnosis server bound to a trusted interface.
"""

import os
import time
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from vllm.utils.mem_utils import MemorySnapshot


class MemoryDiagnostics:
    device: torch.device
    init_snapshot: "MemorySnapshot"
    rank: int
    requested_memory: int

    def capture_memory_diagnostics(self, reset_peak: str = "false") -> dict[str, Any]:
        """Read byte-valued device and allocator observations on this worker.

        Args:
            reset_peak: ``"true"`` resets allocator peak counters after reading
                them, to measure the following stress phase. ``"false"`` only
                observes. The string interface matches development HTTP RPC.

        Returns:
            Worker identity, startup budget, device-wide free/total bytes, and
            this process's allocator counters. Device use minus Torch reserved
            includes CUDA libraries and other processes; it is not an allocator
            leak measurement. No synchronization or cache eviction is performed.
        """
        if reset_peak not in ("true", "false"):
            raise ValueError("reset_peak must be 'true' or 'false'")
        device = self.device
        free_bytes, total_bytes = torch.accelerator.get_memory_info(device)
        stats = dict(torch.accelerator.memory_stats(device))
        reserved_bytes = stats["reserved_bytes.all.current"]
        initial = self.init_snapshot
        observation = {
            "schema_version": 1,
            "captured_unix_ns": time.time_ns(),
            "worker_pid": os.getpid(),
            "rank": self.rank,
            "device": str(device),
            "requested_bytes": self.requested_memory,
            "initial_snapshot": {
                key: getattr(initial, key)
                for key in (
                    "free_memory",
                    "total_memory",
                    "cuda_memory",
                    "torch_memory",
                    "torch_allocated",
                    "non_torch_memory",
                    "torch_peak",
                )
            },
            "device_free_bytes": free_bytes,
            "device_total_bytes": total_bytes,
            "device_used_minus_torch_reserved_bytes": (
                total_bytes - free_bytes - reserved_bytes
            ),
            "allocator_stats": stats,
            "peak_counters_reset_after_observation": reset_peak == "true",
        }
        if reset_peak == "true":
            torch.accelerator.reset_peak_memory_stats(device)
        return observation
