# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Reader for the canonical TP-independent Kimi-K3 QSRT atom container.

The checkpoint stores one safetensors tensor with 96 padded physical atom
rows.  Tensor parallelism selects a contiguous whole-row extent at load time;
the serialized file never names a TP size or rank.  Runtime loading narrows
InstantTensor's unopened I/O layout to that extent so no unrelated atom bytes
cross the host/device boundary.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

SCHEMA = "kquant_kimi_k3_qsrt_atoms_v1"
ENCODING = "qsrt_sqg_e4m3"
VERSION = 1
PROFILE_ID = 1
EXPERTS = 896
HIDDEN_SIZE = 3584
INTERMEDIATE_SIZE = 3072
ATOM_CHANNELS = 32
ATOM_SLOTS = 96
ATOM_BUNDLE_BYTES = 129216
FORMAT_SECTION_BYTES = 4096
FORMAT_TABLE_BYTES = EXPERTS
SHARED_SCALE_SECTION_BYTES = 24576
SHARED_SCALE_BYTES = 3 * HIDDEN_SIZE * torch.float16.itemsize
FORMAT_X4T = 0xFF

FORMAT_TENSOR = "_qsrt_format_section"
SHARED_SCALE_TENSOR = "_qsrt_shared_scale_section"
ATOM_TENSOR = "qsrt_atoms"
TENSOR_INVENTORY = {FORMAT_TENSOR, SHARED_SCALE_TENSOR, ATOM_TENSOR}

# InstantTensor's buffered-AIO default can reserve 4 GiB per open reader
# (512 x 8 MiB). Keep its storage-native 8 MiB chunks, but cap the queue at
# 128 MiB and allow it to use the small amount of free memory left late in a
# full-model load. Changing the native chunk geometry can abort its async
# executor on this storage stack.
_INSTANTTENSOR_IO_DEPTH = 16
_INSTANTTENSOR_MAX_FREE_MEM_USAGE = 0.9


def _concrete_device(device: torch.device | str) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None:
        return torch.device("cuda", torch.accelerator.current_device_index())
    return resolved


def _metadata_int(metadata: dict[str, str], name: str) -> int:
    try:
        value = int(metadata[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"QSRT metadata {name!r} is missing or invalid") from exc
    return value


@dataclass(frozen=True)
class QSRTAtomLayerMetadata:
    path: Path
    layer: int
    format_codes: torch.Tensor
    compressed_expert_ids: torch.Tensor
    x4t_expert_ids: torch.Tensor
    gate_suh: torch.Tensor
    up_suh: torch.Tensor
    down_svh: torch.Tensor
    atom_slot_stride_bytes: int

    @property
    def compressed_experts(self) -> int:
        return int(self.compressed_expert_ids.numel())

    @property
    def x4t_experts(self) -> int:
        return int(self.x4t_expert_ids.numel())

    @property
    def atom_slot_payload_bytes(self) -> int:
        return self.compressed_experts * ATOM_BUNDLE_BYTES


def read_qsrt_atom_layer_metadata(
    path: str | Path,
    *,
    layer: int,
    expected_bits: Sequence[int] | None = None,
) -> QSRTAtomLayerMetadata:
    """Validate the small metadata tensors without reading the atom slab."""

    path = Path(path)
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        if set(handle.keys()) != TENSOR_INVENTORY:
            raise ValueError("QSRT atom tensor inventory is noncanonical")
        expected_metadata: dict[str, str] = {
            "format": "pt",
            "schema": SCHEMA,
            "version": str(VERSION),
            "encoding": ENCODING,
            "layer": str(layer),
            "profile_id": str(PROFILE_ID),
            "experts": str(EXPERTS),
            "intermediate_channels": str(INTERMEDIATE_SIZE),
            "latent_channels": str(HIDDEN_SIZE),
            "atom_channels": str(ATOM_CHANNELS),
            "atom_slots": str(ATOM_SLOTS),
            "atom_bundle_bytes": str(ATOM_BUNDLE_BYTES),
            "alignment_bytes": "4096",
        }
        for name, expected in expected_metadata.items():
            if metadata.get(name) != expected:
                raise ValueError(
                    f"QSRT metadata {name} mismatch: "
                    f"{metadata.get(name)!r} != {expected!r}"
                )

        compressed_count = _metadata_int(metadata, "compressed_experts")
        x4t_count = _metadata_int(metadata, "x4t_experts")
        slot_payload = _metadata_int(metadata, "atom_slot_payload_bytes")
        slot_stride = _metadata_int(metadata, "atom_slot_stride_bytes")
        if compressed_count + x4t_count != EXPERTS:
            raise ValueError("QSRT tier counts do not cover all experts")
        if slot_payload != compressed_count * ATOM_BUNDLE_BYTES:
            raise ValueError("QSRT atom-slot payload byte count is invalid")
        if slot_stride < slot_payload or slot_stride % 4096:
            raise ValueError("QSRT atom-slot stride is invalid")

        format_section = handle.get_tensor(FORMAT_TENSOR)
        if (
            format_section.dtype != torch.uint8
            or tuple(format_section.shape) != (FORMAT_SECTION_BYTES,)
            or bool(torch.any(format_section[FORMAT_TABLE_BYTES:] != 0))
        ):
            raise ValueError("QSRT format section is malformed")
        format_codes = format_section[:FORMAT_TABLE_BYTES].clone().contiguous()
        r13 = format_codes >> 4
        r2 = format_codes & 0xF
        compressed_mask = format_codes != FORMAT_X4T
        if bool(torch.any(compressed_mask & ((r13 > 2) | (r2 > 2)))):
            raise ValueError("QSRT format table contains an invalid rate code")
        compressed_ids = torch.nonzero(compressed_mask, as_tuple=False).flatten()
        x4t_ids = torch.nonzero(~compressed_mask, as_tuple=False).flatten()
        if (
            int(compressed_ids.numel()) != compressed_count
            or int(x4t_ids.numel()) != x4t_count
        ):
            raise ValueError("QSRT format table tier counts disagree with metadata")
        if expected_bits is not None:
            if len(expected_bits) != EXPERTS:
                raise ValueError("hybrid_bit_map must describe all 896 experts")
            expected_x4t = torch.tensor(expected_bits, dtype=torch.int16) == 4
            if not torch.equal(expected_x4t, ~compressed_mask):
                raise ValueError("QSRT format table disagrees with hybrid_bit_map")

        shared = handle.get_tensor(SHARED_SCALE_TENSOR)
        if (
            shared.dtype != torch.uint8
            or tuple(shared.shape) != (SHARED_SCALE_SECTION_BYTES,)
            or bool(torch.any(shared[SHARED_SCALE_BYTES:] != 0))
        ):
            raise ValueError("QSRT shared-scale section is malformed")
        vectors = (
            shared[:SHARED_SCALE_BYTES]
            .clone()
            .view(torch.float16)
            .reshape(3, HIDDEN_SIZE)
            .contiguous()
        )
        if not bool(torch.all(torch.isfinite(vectors))):
            raise ValueError("QSRT shared scales contain non-finite values")

        atom_shape = handle.get_slice(ATOM_TENSOR).get_shape()
        if atom_shape != [ATOM_SLOTS, slot_stride]:
            raise ValueError(
                f"QSRT atom slab shape {atom_shape} != {[ATOM_SLOTS, slot_stride]}"
            )

    return QSRTAtomLayerMetadata(
        path=path,
        layer=layer,
        format_codes=format_codes,
        compressed_expert_ids=compressed_ids.to(torch.int32).contiguous(),
        x4t_expert_ids=x4t_ids.to(torch.int32).contiguous(),
        gate_suh=vectors[0].contiguous(),
        up_suh=vectors[1].contiguous(),
        down_svh=vectors[2].contiguous(),
        atom_slot_stride_bytes=slot_stride,
    )


def balanced_atom_partition(shard_count: int, shard_index: int) -> tuple[int, int]:
    if not 1 <= shard_count <= ATOM_SLOTS:
        raise ValueError(f"shard_count must lie in 1..{ATOM_SLOTS}")
    if not 0 <= shard_index < shard_count:
        raise ValueError(f"shard_index must lie in 0..{shard_count - 1}")
    quotient, remainder = divmod(ATOM_SLOTS, shard_count)
    count = quotient + int(shard_index < remainder)
    first = shard_index * quotient + min(shard_index, remainder)
    return first, count


def _select_instanttensor_extent(
    handle: Any,
    *,
    tensor_name: str,
    first_row: int,
    rows: int,
    row_bytes: int,
) -> None:
    """Restrict an unopened InstantTensor handle to one tensor row extent."""

    required = (
        "ordered_tensor_metadatas",
        "tensor_offsets",
        "tensor_sizes",
        "total_tensor_size",
        "tensor_name_to_index",
        "loader_handle",
        "_determine_buffer_size",
    )
    missing = [name for name in required if not hasattr(handle, name)]
    if missing:
        raise RuntimeError(
            "Installed InstantTensor cannot select a QSRT atom extent "
            f"(missing: {', '.join(missing)})"
        )
    if handle.loader_handle is not None:
        raise RuntimeError("QSRT extent selection must occur before InstantTensor I/O")
    try:
        tensor_index = handle.tensor_name_to_index[tensor_name]
    except KeyError as exc:
        raise ValueError(f"QSRT file omits tensor {tensor_name!r}") from exc
    file_index, tensor_start = handle.tensor_offsets[tensor_index]
    end_file_index, tensor_end = handle.tensor_offsets[tensor_index + 1]
    if (
        file_index != end_file_index
        or tensor_end - tensor_start != ATOM_SLOTS * row_bytes
    ):
        raise RuntimeError("InstantTensor QSRT atom offsets disagree with metadata")
    begin = tensor_start + first_row * row_bytes
    end = begin + rows * row_bytes
    metadata = {
        "dtype": "U8",
        "shape": [rows, row_bytes],
        "data_offsets": [0, end - begin],
    }
    handle.ordered_tensor_metadatas = [(tensor_name, metadata)]
    handle.tensor_name_to_index = {tensor_name: 0}
    handle.tensor_offsets = [(file_index, begin), (file_index, end)]
    handle.tensor_sizes = [end - begin]
    handle.total_tensor_size = end - begin
    handle._determine_buffer_size(None)


@contextmanager
def open_qsrt_atom_extent(
    metadata: QSRTAtomLayerMetadata,
    *,
    shard_count: int,
    shard_index: int,
    device: torch.device | str | None,
) -> Iterator[tuple[int, torch.Tensor]]:
    """Yield ``(first_atom_slot, [A,E,129216]u8)`` for one shard.

    CPU mode exists for focused tests.  CUDA mode uses InstantTensor and keeps
    the returned tensor valid only for the context lifetime; callers must
    complete preparation before leaving the context.
    """

    first, rows = balanced_atom_partition(shard_count, shard_index)
    if device is None or torch.device(device).type == "cpu":
        with safe_open(metadata.path, framework="pt", device="cpu") as handle:
            padded = handle.get_slice(ATOM_TENSOR)[first : first + rows]
            compact = padded[:, : metadata.atom_slot_payload_bytes].contiguous()
            yield (
                first,
                compact.reshape(rows, metadata.compressed_experts, ATOM_BUNDLE_BYTES),
            )
        return

    import instanttensor

    opener = instanttensor.safe_open(
        str(metadata.path),
        framework="pt",
        device=_concrete_device(device),
        concurrency=1,
        io_depth=_INSTANTTENSOR_IO_DEPTH,
        max_free_mem_usage=_INSTANTTENSOR_MAX_FREE_MEM_USAGE,
        load_now=False,
        copy=False,
    )
    _select_instanttensor_extent(
        opener,
        tensor_name=ATOM_TENSOR,
        first_row=first,
        rows=rows,
        row_bytes=metadata.atom_slot_stride_bytes,
    )
    with opener as handle:
        loaded = dict(handle.tensors())
        padded = loaded[ATOM_TENSOR]
        compact = padded[:, : metadata.atom_slot_payload_bytes]
        # Preserve the padded row stride as a zero-copy view. B12X extracts
        # each matrix component into its prepared owner before this context
        # exits, so carrying padding through the read avoids a second full
        # atom-slab allocation.
        atoms = compact.as_strided(
            (rows, metadata.compressed_experts, ATOM_BUNDLE_BYTES),
            (
                metadata.atom_slot_stride_bytes,
                ATOM_BUNDLE_BYTES,
                1,
            ),
        )
        yield first, atoms


__all__ = [
    "QSRTAtomLayerMetadata",
    "balanced_atom_partition",
    "open_qsrt_atom_extent",
    "read_qsrt_atom_layer_metadata",
]
