# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deterministic ordering of the B12X sparse-indexer selection.

The B12X DSA indexer packs the selected KV rows of a query in the order its
CTAs reserve output slots, an atomic arrival order that follows kernel
timing. The sparse MLA decode kernel sums the selected rows in that order in
fp32, so a change in the timing of the kernels around the indexer (a
concurrent weight prefetch, a different launch sequence) moves the summation
order and, through the BF16 rounding of the layer output, the deep-tail log
probabilities of long contexts. Sorting each row makes the order a function
of the selected set alone.

The sort is the b12x op ``attention.topk_sort`` (``sort_convert``): the
indexer emits logical positions, and the sort rewrites each row ascending and
converts the positions to physical cache slots in place. This module owns
the dispatch: the run-time gates, the side stream inside full CUDA graphs and
the join before the first consumer.

Inside full CUDA graphs the decode-path sort runs on its own stream
(``sort_convert_async``: the stream waits for the indexer and the sort is
enqueued there; ``join`` makes the main stream wait for it). The join sits in
the sparse MLA backend's ``forward_mqa`` right before the first read of the
selection, after the KV-cache write and the query projections (which this
model issues before the indexer) and after the query concatenation that
``forward_mqa`` runs on the main stream between the indexer and the
attention kernels; that concatenation and the launch latency of the sort are
what the side stream hides. A join placed before the attention op would
leave nothing to overlap: the captured chain would be linear and the driver
would replay it on one stream. The sort stream is not the L2-prefetch side
stream: that one carries the output-projection prefetch, issued at the start
of the attention block, and a sort enqueued behind it would make the
attention kernels wait for the prefetch. Eager forwards sort in line: a fork
records an event per call, and events pending behind a host that runs ahead
retain device memory.

``VLLM_TOPK_SORT=0`` disables the sort and ``VLLM_TOPK_SORT_MAX_TOKENS``
(default 8) bounds the row count that sorts. Both are read once per process
and decide, at model construction, which indexer plans emit logical
positions; they never enter a compile or cache key.
"""

from __future__ import annotations

import os

import torch

from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context, is_forward_context_available

ENABLED = os.environ.get("VLLM_TOPK_SORT", "1") != "0"
#: Largest row count (tokens of the step) that is sorted. The sort exists so
#: that loads overlapping the indexer and the attention kernels (the L2
#: prefetch attention window, gated by ``VLLM_L2_PREFETCH_ATTN_MAX_TOKENS``)
#: cannot change the attention's summation order; steps beyond that window
#: do not pay for it. 0 sorts every row count.
MAX_TOKENS = int(os.environ.get("VLLM_TOPK_SORT_MAX_TOKENS", "8"))

_pending_devices: set[int] = set()
_streams: dict[int, torch.cuda.Stream] = {}


def active(rows: int) -> bool:
    """Whether a step of ``rows`` tokens sorts its selection."""
    return ENABLED and (MAX_TOKENS <= 0 or rows <= MAX_TOKENS)


def _op():
    from b12x.attention import topk_sort

    return topk_sort


def is_supported(device: torch.device) -> bool:
    return ENABLED and _op().is_supported(device)


def precompile(max_positions: int, device: torch.device) -> None:
    """Compile and warm-run the sort before any CUDA-graph capture."""
    _op().precompile(max_positions, device)


def _device_index(device: torch.device) -> int:
    return (
        device.index
        if device.index is not None
        else torch.accelerator.current_device_index()
    )


def _stream(device: torch.device) -> torch.cuda.Stream:
    index = _device_index(device)
    stream = _streams.get(index)
    if stream is None:
        stream = torch.cuda.Stream(device=device)
        _streams[index] = stream
    return stream


def _graph_mode() -> bool:
    if torch.cuda.is_current_stream_capturing():
        return True
    if not is_forward_context_available():
        return False
    return get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.FULL


def sort_convert(
    indices: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    max_positions: int,
) -> torch.Tensor:
    """Sort each row of logical positions ascending and convert them in place
    to physical slots, on the current stream."""
    _op().sort_convert(
        indices,
        seq_lens.to(torch.int32),
        block_table,
        block_size,
        max_positions,
    )
    return indices


def sort_convert_async(
    indices: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    max_positions: int,
) -> torch.Tensor:
    """``sort_convert`` on the sort stream inside full CUDA graphs, in line
    otherwise. The caller must run ``join`` on the main stream before the
    first consumer of ``indices``."""
    if not _graph_mode():
        return sort_convert(indices, seq_lens, block_table, block_size, max_positions)
    device = indices.device
    main = torch.cuda.current_stream(device)
    side = _stream(device)
    side.wait_stream(main)
    with torch.cuda.stream(side):
        sort_convert(indices, seq_lens, block_table, block_size, max_positions)
    _pending_devices.add(_device_index(device))
    return indices


def join(device: torch.device) -> None:
    """Main-stream wait for a pending ``sort_convert_async``; a no-op when
    nothing is pending."""
    index = _device_index(device)
    if index not in _pending_devices:
        return
    torch.cuda.current_stream(device).wait_stream(_stream(device))
    _pending_devices.remove(index)
