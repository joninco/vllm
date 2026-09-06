# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Overlap GLM decode collectives with next-layer weight loads.

Small tensor-parallel decode steps leave high-bandwidth memory underused while
the final MoE or dense-MLP all-reduce waits on PCIe synchronization. This
module uses that interval to stream the next decoder layer's BF16 attention
projection weights into L2 on a side stream. The consumer does not wait for
the prefetch: weights that have arrived are served from L2 and remaining
weights are fetched from memory normally.

The side stream is used only in full CUDA graphs whose padded token count is
bounded by ``VLLM_L2_PREFETCH_MAX_TOKENS``. Every forward joins the side stream
before returning, which closes the graph fork. Eager and piecewise-graph
forwards only warm the Triton kernel; creating an event for every eager layer
would retain driver allocations while the host runs ahead of the GPU.

``register_attention_layers`` must run during model construction. It attaches
fixed weight lists to each decoder layer whose entry reduction can overlap the
loads and allocates a persistent sink before graph tracing. Live request
quantities never enter a compile or cache key.
"""

from __future__ import annotations

import os

import torch

from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

ENABLED = os.environ.get("VLLM_L2_PREFETCH", "1") == "1"
MAX_TOKENS = int(os.environ.get("VLLM_L2_PREFETCH_MAX_TOKENS", "64"))
NUM_PROGRAMS = int(os.environ.get("VLLM_L2_PREFETCH_CTAS", "32"))

_BLOCK = 2048
_UNROLL = 8
_streams: dict[int, torch.cuda.Stream] = {}
_sinks: dict[int, torch.Tensor] = {}
_pending_devices: set[int] = set()
_warmed_devices: set[int] = set()


@triton.jit
def _l2_prefetch_kernel(
    ptr,
    n,
    out_ptr,
    BLOCK: tl.constexpr,
    NPROG: tl.constexpr,
    UNROLL: tl.constexpr,
):
    pid = tl.program_id(0)
    acc = tl.zeros([BLOCK], dtype=tl.int64)
    for start in range(pid * BLOCK * UNROLL, n, NPROG * BLOCK * UNROLL):
        for offset in tl.static_range(UNROLL):
            indices = start + offset * BLOCK + tl.arange(0, BLOCK)
            acc += tl.load(ptr + indices, mask=indices < n, other=0)
    tl.store(out_ptr + pid, tl.sum(acc))


def _device_index(device: torch.device) -> int:
    if device.type != "cuda":
        raise ValueError("L2 weight prefetch requires a CUDA device")
    return (
        device.index
        if device.index is not None
        else torch.accelerator.current_device_index()
    )


def _stream(device: torch.device) -> torch.cuda.Stream:
    device_index = _device_index(device)
    stream = _streams.get(device_index)
    if stream is None:
        stream = torch.cuda.Stream(device=device)
        _streams[device_index] = stream
    return stream


def _sink(device: torch.device) -> torch.Tensor:
    device_index = _device_index(device)
    value = _sinks.get(device_index)
    if value is None:
        value = torch.zeros(1 << 16, dtype=torch.int64, device=device)
        _sinks[device_index] = value
    return value


def _launch(weight: torch.Tensor, sink: torch.Tensor) -> None:
    packed_weight = weight.view(-1).view(torch.int64)
    _l2_prefetch_kernel[(NUM_PROGRAMS,)](
        packed_weight,
        packed_weight.numel(),
        sink,
        BLOCK=_BLOCK,
        NPROG=NUM_PROGRAMS,
        UNROLL=_UNROLL,
        num_warps=16,
    )


def _active(num_tokens: int) -> bool:
    if not ENABLED or num_tokens > MAX_TOKENS or not is_forward_context_available():
        return False
    mode = get_forward_context().cudagraph_runtime_mode
    if mode == CUDAGraphMode.PIECEWISE:
        return False
    return mode == CUDAGraphMode.FULL or torch.cuda.is_current_stream_capturing()


def _warm_up(weights: list[torch.Tensor], sink: torch.Tensor) -> None:
    device_index = _device_index(sink.device)
    if device_index in _warmed_devices or torch.cuda.is_current_stream_capturing():
        return
    _launch(weights[0], sink)
    _warmed_devices.add(device_index)


def _prefetch_impl(
    hidden_states: torch.Tensor,
    weights: list[torch.Tensor],
    sink: torch.Tensor,
) -> None:
    if not _active(hidden_states.shape[0]):
        _warm_up(weights, sink)
        return

    device_index = _device_index(hidden_states.device)
    main_stream = torch.cuda.current_stream(hidden_states.device)
    prefetch_stream = _stream(hidden_states.device)
    prefetch_stream.wait_stream(main_stream)
    with torch.cuda.stream(prefetch_stream):
        for weight in weights:
            _launch(weight, sink)
    _pending_devices.add(device_index)
    logger.info_once(
        "L2 attention-weight prefetch is active in full CUDA graphs "
        "through %d padded tokens.",
        MAX_TOKENS,
    )


def _prefetch_fake(
    hidden_states: torch.Tensor,
    weights: list[torch.Tensor],
    sink: torch.Tensor,
) -> None:
    return None


def _join_impl(sink: torch.Tensor) -> None:
    device_index = _device_index(sink.device)
    if device_index not in _pending_devices:
        return
    torch.cuda.current_stream(sink.device).wait_stream(_stream(sink.device))
    _pending_devices.remove(device_index)


def _join_fake(sink: torch.Tensor) -> None:
    return None


direct_register_custom_op(
    op_name="l2_weight_prefetch",
    op_func=_prefetch_impl,
    mutates_args=["sink"],
    fake_impl=_prefetch_fake,
)
direct_register_custom_op(
    op_name="l2_weight_prefetch_join",
    op_func=_join_impl,
    mutates_args=["sink"],
    fake_impl=_join_fake,
)


def issue(hidden_states: torch.Tensor, weights: list[torch.Tensor] | None) -> None:
    """Issue a prefetch ordered after ``hidden_states`` when targets exist."""
    if weights and hidden_states.device.type == "cuda":
        torch.ops.vllm.l2_weight_prefetch(
            hidden_states,
            weights,
            _sink(hidden_states.device),
        )


def join(hidden_states: torch.Tensor) -> None:
    """Join an existing device prefetch stream before the model returns."""
    if hidden_states.device.type != "cuda":
        return
    sink = _sinks.get(_device_index(hidden_states.device))
    if sink is not None:
        torch.ops.vllm.l2_weight_prefetch_join(sink)


def _bf16_weight(module: object, name: str) -> torch.Tensor | None:
    submodule = getattr(module, name, None)
    weight = getattr(submodule, "weight", None)
    if (
        isinstance(weight, torch.Tensor)
        and weight.dtype == torch.bfloat16
        and weight.is_contiguous()
    ):
        return weight
    return None


def register_attention_layers(layers: torch.nn.ModuleList, start: int, end: int) -> int:
    """Attach fixed attention projection weights to decoder layers.

    Each local layer after the first receives its own fused query/key/value
    input projection, query output projection, and sparse indexer query
    projection when that layer computes an index selection. The preceding
    layer leaves its MLP output unreduced, so these weights can load alongside
    the entry fused all-reduce and RMSNorm. The returned count is the number of
    decoder transitions with a non-empty target list.
    """
    target_count = 0
    for layer_index in range(start, end):
        layer = layers[layer_index]
        attention = getattr(layer, "self_attn", None)
        layer._l2_prefetch_weights = None
        if attention is None or layer_index == start:
            continue

        weights = [
            _bf16_weight(attention, "fused_qkv_a_proj"),
            _bf16_weight(attention, "q_b_proj"),
        ]
        indexer = getattr(attention, "indexer", None)
        if indexer is not None and not getattr(attention, "skip_topk", True):
            weights.append(_bf16_weight(indexer, "wq_b"))
        targets = [weight for weight in weights if weight is not None]
        if not targets:
            continue

        layer._l2_prefetch_weights = targets
        target_count += 1
        if targets[0].device.type == "cuda":
            _sink(targets[0].device)

    logger.info_once(
        "Registered %d decoder transitions for L2 attention-weight prefetch.",
        target_count,
    )
    return target_count
