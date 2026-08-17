# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tensor-parallel collectives for Kimi-K3 projection outputs."""

import torch

import vllm.envs as envs
from vllm.distributed import (
    get_dcp_group,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_gather,
)
from vllm.v1.attention.ops.dcp_alltoall import dcp_b12x_all_gather_heads


def _get_kimi_projection_group():
    """Return the coordinator that spans every projection weight shard.

    Projection weights are sharded across the full tensor-parallel group. The
    DCP coordinator is valid only when its ordered rank list matches the
    tensor-parallel coordinator.
    """
    tp_size = get_tensor_model_parallel_world_size()
    dcp_group = get_dcp_group()
    tp_group = get_tp_group()
    if tp_group.world_size != tp_size:
        raise RuntimeError(
            "Kimi projection group does not span tensor-parallel ranks: "
            f"group={tp_group.world_size}, TP={tp_size}"
        )
    if dcp_group.world_size == tp_size and list(dcp_group.ranks) == list(
        tp_group.ranks
    ):
        return dcp_group
    return tp_group


def _try_b12x_kimi_projection_gather(
    output_parallel: torch.Tensor,
) -> torch.Tensor | None:
    """Gather one decode projection over the lossless B12X copy channel."""
    if (
        not envs.VLLM_USE_B12X_DCP_A2A
        or output_parallel.ndim != 2
        or output_parallel.shape[0] != 1
        or not output_parallel.is_cuda
        or not output_parallel.is_contiguous()
    ):
        return None

    tp_size = get_tensor_model_parallel_world_size()
    if tp_size <= 1:
        return None
    projection_group = _get_kimi_projection_group()

    local_width = output_parallel.shape[1]
    restore_dtype: torch.dtype | None = None
    strip_local_width: int | None = None
    if output_parallel.dtype in (torch.float16, torch.bfloat16):
        if local_width % 8 == 0:
            transport = output_parallel.view(1, 1, local_width)
        else:
            padded_width = (local_width + 7) // 8 * 8
            transport = torch.nn.functional.pad(
                output_parallel, (0, padded_width - local_width)
            ).view(1, 1, padded_width)
            strip_local_width = local_width
    elif output_parallel.dtype == torch.float32:
        raw_width = local_width * output_parallel.element_size()
        if raw_width % 8 != 0:
            return None
        # The FP8 view exposes one-byte transport lanes without converting the
        # FP32 payload. The gathered result is restored to the original dtype.
        transport = output_parallel.view(torch.float8_e4m3fn).view(1, 1, raw_width)
        restore_dtype = torch.float32
    elif output_parallel.dtype == torch.float8_e4m3fn:
        if local_width % 16 != 0:
            return None
        transport = output_parallel.view(1, 1, local_width)
    else:
        return None

    gathered = dcp_b12x_all_gather_heads(
        transport,
        projection_group,
        max_batch_size=1,
    )
    if strip_local_width is not None:
        gathered = gathered.narrow(-1, 0, strip_local_width).contiguous()
    gathered = gathered.flatten(1)
    if restore_dtype is not None:
        gathered = gathered.view(restore_dtype)
    return gathered


def gather_kimi_sharded_projection(output_parallel: torch.Tensor) -> torch.Tensor:
    """Gather a rank-major Kimi-K3 projection through a lossless fast path."""
    if get_tensor_model_parallel_world_size() <= 1:
        return output_parallel
    gathered = _try_b12x_kimi_projection_gather(output_parallel)
    if gathered is not None:
        return gathered
    return tensor_model_parallel_all_gather(output_parallel, dim=-1)
