# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.quantization.mxfp4 import (
    _ceil_div,
    _e8m0_bytes_to_float,
    _e8m0_scale_bytes_from_amax,
    _mxfp4_decode_packed,
    _mxfp4_encode_values,
    _mxfp4_realign_w2_fp4_e8m0_to_local_k32,
    _mxfp4_w2_scale_cols_for_rank,
)


def _pack_codes(codes: torch.Tensor) -> torch.Tensor:
    if codes.shape[-1] % 2:
        pad = torch.zeros(*codes.shape[:-1], 1, dtype=torch.uint8)
        codes = torch.cat((codes, pad), dim=-1)
    return codes[..., 0::2] | (codes[..., 1::2] << 4)


def _dequant_w2(
    w2: torch.Tensor,
    scale: torch.Tensor,
    *,
    logical_k: int,
    source_k_offset: int,
) -> torch.Tensor:
    flat_w2 = w2.view(-1, w2.shape[-1])
    flat_scale = scale.view(-1, scale.shape[-1])
    raw = _mxfp4_decode_packed(flat_w2, logical_k)
    cols = torch.arange(logical_k)
    source_groups = ((source_k_offset + cols) // 32).to(torch.long)
    scale_f32 = _e8m0_bytes_to_float(flat_scale.index_select(1, source_groups))
    return raw * scale_f32


def test_mxfp4_w2_scale_cols_cover_virtual_tp_alignment_8() -> None:
    assert [
        _mxfp4_w2_scale_cols_for_rank(logical_k=312, tp_rank=rank)
        for rank in range(10)
    ] == [10, 11, 11, 10, 10, 11, 11, 10, 10, 11]


def test_mxfp4_w2_realign_requantizes_crossing_scale_groups() -> None:
    logical_k = 40
    source_k_offset = 24
    rows = 5
    raw_scale_cols = _ceil_div(source_k_offset + logical_k, 32)
    local_scale_cols = _ceil_div(logical_k, 32)

    codes = (torch.arange(rows * logical_k, dtype=torch.uint8) % 16).view(
        rows,
        logical_k,
    )
    w2 = _pack_codes(codes).view(1, rows, logical_k // 2)
    raw_scale = torch.tensor(
        [
            [126, 129],
            [124, 127],
            [128, 126],
            [125, 130],
            [127, 128],
        ],
        dtype=torch.uint8,
    ).view(1, rows, raw_scale_cols)

    source_vals = _dequant_w2(
        w2,
        raw_scale,
        logical_k=logical_k,
        source_k_offset=source_k_offset,
    )

    _mxfp4_realign_w2_fp4_e8m0_to_local_k32(
        w2,
        raw_scale,
        logical_k=logical_k,
        source_k_offset=source_k_offset,
        row_chunk=2,
    )

    local_scale = torch.empty(rows, local_scale_cols, dtype=torch.uint8)
    expected_codes = torch.empty(rows, logical_k, dtype=torch.uint8)
    for group_idx in range(local_scale_cols):
        k_start = group_idx * 32
        k_end = min(k_start + 32, logical_k)
        group_vals = source_vals[:, k_start:k_end]
        scale_bytes = _e8m0_scale_bytes_from_amax(group_vals.abs().amax(dim=1))
        local_scale[:, group_idx] = scale_bytes
        scale = _e8m0_bytes_to_float(scale_bytes).unsqueeze(1)
        expected_codes[:, k_start:k_end] = _mxfp4_encode_values(
            group_vals / scale.clamp(min=1e-30)
        )

    expected_w2 = _pack_codes(expected_codes).view_as(w2)
    assert torch.equal(w2, expected_w2)

    dequant_after = _dequant_w2(
        w2,
        local_scale.view(1, rows, local_scale_cols),
        logical_k=logical_k,
        source_k_offset=0,
    )
    assert torch.isfinite(dequant_after).all()
