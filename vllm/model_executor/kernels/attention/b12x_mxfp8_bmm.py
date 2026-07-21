# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib
from collections.abc import Iterable
from typing import Any, Literal

import torch

from vllm.utils.torch_utils import direct_register_custom_op

_BMM_SPEC = {
    "a_dtype": "bfloat16",
    "b_dtype": "float8_e4m3fn",
    "sf_dtype": "float8_e8m0fnu",
    "c_dtype": "bfloat16",
    "sf_vec_size": 32,
}
_B12X_BMM: Any | None = None
_B12X_BMM_MISSING = False


def _import_b12x_bmm() -> Any | None:
    global _B12X_BMM, _B12X_BMM_MISSING
    if _B12X_BMM is not None:
        return _B12X_BMM
    if _B12X_BMM_MISSING:
        return None
    try:
        gemm = importlib.import_module("sparkinfer.gemm")
    except ImportError:
        _B12X_BMM_MISSING = True
        return None
    required = ("bmm", "can_implement_bmm", "prewarm_bmm")
    if not all(callable(getattr(gemm, name, None)) for name in required):
        _B12X_BMM_MISSING = True
        return None
    _B12X_BMM = gemm
    return gemm


def can_implement_b12x_mxfp8_bmm(
    *,
    batch: int,
    max_m: int,
    n: int,
    k: int,
    b_major: Literal["k", "n"],
    device: torch.device,
) -> bool:
    gemm = _import_b12x_bmm()
    if gemm is None:
        return False
    return bool(
        gemm.can_implement_bmm(
            batch=batch,
            max_m=max_m,
            n=n,
            k=k,
            b_major=b_major,
            sf_axis=b_major,
            device=device,
            **_BMM_SPEC,
        )
    )


def _b12x_mxfp8_bmm_impl(
    lhs: torch.Tensor,
    b_values: torch.Tensor,
    b_scales: torch.Tensor,
    out: torch.Tensor,
    b_major: int,
) -> None:
    gemm = _import_b12x_bmm()
    if gemm is None:
        raise ImportError("sparkinfer.gemm.bmm is not available")
    major = "k" if b_major == 0 else "n"
    gemm.bmm(
        lhs,
        (b_values, b_scales),
        out,
        b_major=major,
        sf_axis=major,
        **_BMM_SPEC,
    )


def _b12x_mxfp8_bmm_fake(
    lhs: torch.Tensor,
    b_values: torch.Tensor,
    b_scales: torch.Tensor,
    out: torch.Tensor,
    b_major: int,
) -> None:
    del lhs, b_values, b_scales, out, b_major


direct_register_custom_op(
    op_name="b12x_mxfp8_bmm",
    op_func=_b12x_mxfp8_bmm_impl,
    mutates_args=["out"],
    fake_impl=_b12x_mxfp8_bmm_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


def run_b12x_mxfp8_bmm(
    lhs: torch.Tensor,
    rhs: tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    *,
    b_major: Literal["k", "n"],
) -> torch.Tensor:
    b_values, b_scales = rhs
    torch.ops.vllm.b12x_mxfp8_bmm(
        lhs,
        b_values,
        b_scales,
        out,
        0 if b_major == "k" else 1,
    )
    return out


def warmup_b12x_mla_mxfp8_bmm(
    model: torch.nn.Module,
    *,
    m_values: Iterable[int] = range(1, 33),
) -> int:
    gemm = _import_b12x_bmm()
    if gemm is None:
        return 0

    values = tuple(dict.fromkeys(int(m) for m in m_values if int(m) > 0))
    seen: set[tuple[str, tuple[int, ...], tuple[int, ...], torch.device]] = set()
    warmed = 0
    for module in model.modules():
        for attr, major in (
            ("_b12x_absorb_uk_rhs", "n"),
            ("_b12x_absorb_uv_rhs", "k"),
        ):
            rhs = getattr(module, attr, None)
            if rhs is None:
                continue
            b_values, b_scales = rhs
            signature = (
                major,
                tuple(b_values.shape),
                tuple(b_scales.shape),
                b_values.device,
            )
            if signature in seen:
                continue
            seen.add(signature)
            warmed += gemm.prewarm_bmm(
                rhs,
                values,
                b_major=major,
                sf_axis=major,
                **_BMM_SPEC,
            )
    return warmed


__all__ = [
    "can_implement_b12x_mxfp8_bmm",
    "run_b12x_mxfp8_bmm",
    "warmup_b12x_mla_mxfp8_bmm",
]
