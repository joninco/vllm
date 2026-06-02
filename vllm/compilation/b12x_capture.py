# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterator
from contextlib import contextmanager

import vllm.envs as envs


def b12x_cuda_graph_prewarm_enabled() -> bool:
    return (
        envs.VLLM_USE_B12X_SPARSE_INDEXER
        or envs.VLLM_USE_B12X_MHC
        or envs.VLLM_USE_B12X_FP8_GEMM
        or envs.VLLM_USE_B12X_WO_PROJECTION
        or envs.VLLM_USE_B12X_MOE
    )


@contextmanager
def guard_b12x_kernel_resolution(reason: str) -> Iterator[None]:
    if not b12x_cuda_graph_prewarm_enabled():
        yield
        return

    try:
        from b12x import (
            freeze_kernel_resolution,
            kernel_resolution_frozen,
            unfreeze_kernel_resolution,
        )
    except ImportError:
        yield
        return

    if kernel_resolution_frozen():
        yield
        return

    freeze_kernel_resolution(reason)
    try:
        yield
    finally:
        unfreeze_kernel_resolution()
