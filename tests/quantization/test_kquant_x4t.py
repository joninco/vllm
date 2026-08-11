# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from vllm.model_executor.layers.quantization import kquant_x4t as x4t


def _scale_components(scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows, columns = scale.shape
    source = scale.numpy()
    indexed = (np.arange(rows, dtype=np.int64)[:, None] << 8) + source
    histogram = np.bincount(indexed.ravel(), minlength=rows * 256).reshape(rows, 256)
    bases = np.argmax(histogram[:, :-1] + histogram[:, 1:], axis=1).astype(np.uint8)
    source_i16 = source.astype(np.int16)
    base_i16 = bases.astype(np.int16)
    low = source_i16 == base_i16[:, None]
    high = source_i16 == base_i16[:, None] + 1
    selectors = np.packbits(high, axis=1, bitorder="little")
    tile_bytes = x4t.X4T_TILE_ROWS * (1 + math.ceil(columns / 8))
    fixed = np.empty((rows // x4t.X4T_TILE_ROWS, tile_bytes), dtype=np.uint8)
    fixed[:, : x4t.X4T_TILE_ROWS] = bases.reshape(-1, x4t.X4T_TILE_ROWS)
    fixed[:, x4t.X4T_TILE_ROWS :] = selectors.reshape(rows // 16, -1)
    coordinates = np.argwhere(~(low | high))
    if coordinates.size:
        positions = coordinates[:, 0].astype(np.uint32) * np.uint32(
            columns
        ) + coordinates[:, 1].astype(np.uint32)
        values = source[coordinates[:, 0], coordinates[:, 1]].astype(np.uint32)
        exceptions = positions | (values << np.uint32(x4t.X4T_POSITION_BITS))
    else:
        exceptions = np.empty((0,), dtype=np.uint32)
    return torch.from_numpy(fixed.reshape(-1).copy()), torch.from_numpy(
        exceptions.copy()
    )


def _patterned_scale(matrix: str, value: int) -> torch.Tensor:
    rows, features = x4t._MATRIX_SHAPES[matrix]
    columns = features // x4t.MXFP4_BLOCK
    row = torch.arange(rows, dtype=torch.int16)[:, None]
    column = torch.arange(columns, dtype=torch.int16)[None, :]
    scale = (value + (row % 4) + ((row + 3 * column) & 1)).to(torch.uint8)
    for position, replacement in (
        (3, 97),
        (columns * (rows // 2) + columns // 2, 151),
        (rows * columns - 1, 211),
    ):
        scale.view(-1)[position] = replacement
    return scale.contiguous()


def _write_layer(path, *, layer: int = 24, expert: int = 17) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {
        "expert_ids": torch.tensor([expert], dtype=torch.int32)
    }
    scales: dict[str, torch.Tensor] = {}
    for matrix_index, matrix in enumerate(x4t.X4T_MATRIX_ORDER):
        rows, features = x4t._MATRIX_SHAPES[matrix]
        if matrix in ("w1", "w3"):
            packed = (
                torch.arange(rows, dtype=torch.int64)[:, None]
                .div(256, rounding_mode="floor")
                .to(torch.uint8)
                .expand(rows, features // 2)
                .contiguous()
                .unsqueeze(0)
            )
        else:
            packed = (
                torch.arange(features // x4t.MXFP4_BLOCK, dtype=torch.int64)
                .div(8, rounding_mode="floor")
                .to(torch.uint8)[:, None, None]
                .expand(features // x4t.MXFP4_BLOCK, rows, x4t.MXFP4_BLOCK // 2)
                .contiguous()
                .unsqueeze(0)
            )
        scale = _patterned_scale(matrix, 120 + matrix_index)
        fixed, exceptions = _scale_components(scale)
        tensors[f"{matrix}.packed"] = packed
        tensors[f"{matrix}.scale_fixed"] = fixed.unsqueeze(0)
        tensors[f"{matrix}.scale_exceptions"] = exceptions.view(torch.uint8)
        tensors[f"{matrix}.scale_exception_offsets"] = torch.tensor(
            [0, exceptions.numel()], dtype=torch.int64
        )
        scales[matrix] = scale
    save_file(
        tensors,
        path,
        metadata={
            "format": "pt",
            "schema": x4t.X4T_SCHEMA,
            "version": str(x4t.X4T_VERSION),
            "layer": str(layer),
            "experts": "1",
            "expert_capacity": str(x4t.X4T_EXPERTS_PER_LAYER),
            "matrix_order": ",".join(x4t.X4T_MATRIX_ORDER),
            "scale_codec": "x4t-adjacent-pair-fixed-stream-v1",
            "w2_packed_layout": "group-major-32-channel-v1",
            "exact_mxfp4_reconstruction": "true",
        },
    )
    return scales


def _decode_scale(component: x4t.X4TScaleComponents) -> torch.Tensor:
    rows, columns = component.rows, component.columns
    selector_bytes = math.ceil(columns / 8)
    tile_bytes = x4t.X4T_TILE_ROWS * (1 + selector_bytes)
    fixed = component.fixed.cpu().reshape(rows // x4t.X4T_TILE_ROWS, tile_bytes)
    bases = fixed[:, : x4t.X4T_TILE_ROWS].reshape(rows)
    selectors = fixed[:, x4t.X4T_TILE_ROWS :].reshape(rows, selector_bytes)
    column = torch.arange(columns, dtype=torch.int64)
    selected = (
        selectors[:, column // 8].to(torch.int16) >> (column % 8).to(torch.int16)
    ) & 1
    result = (bases.to(torch.int16)[:, None] + selected).to(torch.uint8)
    for word in component.exceptions.cpu().to(torch.int64).tolist():
        result.view(-1)[word & x4t.X4T_POSITION_MASK] = word >> x4t.X4T_POSITION_BITS
    return result


def test_x4t_safetensors_reader_recovers_balanced_shard(tmp_path) -> None:
    path = tmp_path / "x4t-layer-00024.safetensors"
    scales = _write_layer(path)
    with x4t.X4TLayerReader(
        path, shard_count=12, shard_index=5, device="cpu"
    ) as reader:
        w1, w3, w2 = reader.read_shard_triplet(17)

    assert reader.layer == 24
    assert reader.expert_ids == (17,)
    assert not reader.has(16, "w1")
    assert tuple(w1.packed.shape) == (256, 1792)
    assert tuple(w2.packed.shape) == (3584, 128)
    assert bool(torch.all(w1.packed == 5))
    assert bool(torch.all(w3.packed == 5))
    assert bool(torch.all(w2.packed == 5))
    assert torch.equal(_decode_scale(w1.scale), scales["w1"][1280:1536])
    assert torch.equal(_decode_scale(w3.scale), scales["w3"][1280:1536])
    assert torch.equal(_decode_scale(w2.scale), scales["w2"][:, 40:48])


def test_x4t_extent_plan_reads_only_selected_tp_ranges(tmp_path) -> None:
    path = tmp_path / "x4t-layer-00024.safetensors"
    _write_layer(path)
    reader = x4t.X4TLayerReader(path, shard_count=12, shard_index=5)
    extents = reader._extent_plan()
    names = {name for name, *_ in extents}
    assert "w1.17.packed" in names
    assert "w2.17.packed" in names
    w1 = next(item for item in extents if item[0] == "w1.17.packed")
    w2 = next(item for item in extents if item[0] == "w2.17.packed")
    assert w1[3] == 256 * 1792
    assert w2[3] == 8 * 3584 * 16


def test_x4t_extent_batches_bound_selected_bytes() -> None:
    extents = [
        (f"part.{index}", "source", index * 40, size)
        for index, size in enumerate((40, 50, 20, 100))
    ]
    batches = x4t._batch_extents(extents, max_bytes=100)

    assert batches == [extents[:2], extents[2:3], extents[3:]]
    sizes = [sum(extent[3] for extent in batch) for batch in batches]
    assert sizes == [90, 20, 100]


def test_x4t_extent_batches_reject_invalid_limits() -> None:
    with pytest.raises(ValueError, match="batch size"):
        x4t._batch_extents([], max_bytes=0)
    with pytest.raises(ValueError, match="extent size"):
        x4t._batch_extents([("bad", "source", 0, -1)])
    with pytest.raises(ValueError, match="exceeds"):
        x4t._batch_extents([("large", "source", 0, 101)], max_bytes=100)


def test_x4t_rejects_noncanonical_metadata(tmp_path) -> None:
    path = tmp_path / "bad.safetensors"
    save_file(
        {"expert_ids": torch.empty((0,), dtype=torch.int32)},
        path,
        metadata={"schema": "not-x4t", "version": "2"},
    )
    with pytest.raises(ValueError, match="schema"):
        x4t.read_x4t_layer_metadata(path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_x4t_instanttensor_shard_matches_cpu(tmp_path) -> None:
    path = tmp_path / "x4t-layer-00024.safetensors"
    _write_layer(path)
    with x4t.X4TLayerReader(path, shard_count=12, shard_index=3) as cpu:
        expected = cpu.read_shard_triplet(17)
    with x4t.X4TLayerReader(path, shard_count=12, shard_index=3, device="cuda") as gpu:
        actual = gpu.read_shard_triplet(17)
    for left, right in zip(expected, actual, strict=True):
        assert torch.equal(left.packed, right.packed.cpu())
        assert torch.equal(_decode_scale(left.scale), _decode_scale(right.scale))
