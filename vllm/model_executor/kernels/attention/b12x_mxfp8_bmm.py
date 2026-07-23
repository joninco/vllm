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
_MXFP8_MLA_QUERY: Any | None = None
_MXFP8_MLA_QUERY_MISSING = False


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


def _import_mxfp8_mla_query() -> Any | None:
    global _MXFP8_MLA_QUERY, _MXFP8_MLA_QUERY_MISSING
    if _MXFP8_MLA_QUERY is not None:
        return _MXFP8_MLA_QUERY
    if _MXFP8_MLA_QUERY_MISSING:
        return None
    try:
        mla_query = importlib.import_module("sparkinfer.attention.mla_query")
    except ImportError:
        _MXFP8_MLA_QUERY_MISSING = True
        return None
    required = ("run", "can_implement", "prewarm")
    if not all(callable(getattr(mla_query, name, None)) for name in required):
        _MXFP8_MLA_QUERY_MISSING = True
        return None
    _MXFP8_MLA_QUERY = mla_query
    return mla_query


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


def can_implement_mxfp8_mla_query(
    *,
    num_heads: int,
    max_m: int,
    nope_dim: int,
    latent_dim: int,
    output_dtype: torch.dtype,
    device: torch.device,
) -> bool:
    mla_query = _import_mxfp8_mla_query()
    if mla_query is None:
        return False
    return bool(
        mla_query.can_implement(
            num_heads=num_heads,
            max_m=max_m,
            nope_dim=nope_dim,
            latent_dim=latent_dim,
            output_dtype=output_dtype,
            device=device,
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


def _mxfp8_mla_query_impl(
    lhs: torch.Tensor,
    b_values: torch.Tensor,
    b_scales: torch.Tensor,
    q_pe: torch.Tensor,
    q_scale: torch.Tensor,
    out: torch.Tensor,
) -> None:
    mla_query = _import_mxfp8_mla_query()
    if mla_query is None:
        raise ImportError("sparkinfer.attention.mla_query is not available")
    mla_query.run(lhs, (b_values, b_scales), q_pe, q_scale, out)


def _mxfp8_mla_query_fake(
    lhs: torch.Tensor,
    b_values: torch.Tensor,
    b_scales: torch.Tensor,
    q_pe: torch.Tensor,
    q_scale: torch.Tensor,
    out: torch.Tensor,
) -> None:
    del lhs, b_values, b_scales, q_pe, q_scale, out


direct_register_custom_op(
    op_name="mxfp8_mla_query",
    op_func=_mxfp8_mla_query_impl,
    mutates_args=["out"],
    fake_impl=_mxfp8_mla_query_fake,
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


def run_mxfp8_mla_query(
    lhs: torch.Tensor,
    rhs: tuple[torch.Tensor, torch.Tensor],
    q_pe: torch.Tensor,
    q_scale: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    b_values, b_scales = rhs
    torch.ops.vllm.mxfp8_mla_query(
        lhs,
        b_values,
        b_scales,
        q_pe,
        q_scale,
        out,
    )
    return out


def _module_uses_mxfp8_mla_query(module: torch.nn.Module) -> bool:
    """Return whether fused query assembly replaces this module's UK BMM."""
    mla_query = _import_mxfp8_mla_query()
    rhs = getattr(module, "_b12x_absorb_uk_rhs", None)
    output_dtype = getattr(module, "_mxfp8_mla_query_output_dtype", None)
    if (
        mla_query is None
        or rhs is None
        or output_dtype
        not in (
            torch.bfloat16,
            torch.float8_e4m3fn,
        )
    ):
        return False

    b_values, _ = rhs
    backend_gate = getattr(
        getattr(module, "impl", None),
        "supports_mxfp8_mla_query_output",
        None,
    )
    if callable(backend_gate) and not backend_gate(
        int(b_values.shape[0]), output_dtype
    ):
        return False

    return bool(
        mla_query.can_implement(
            num_heads=int(b_values.shape[0]),
            max_m=1,
            nope_dim=int(b_values.shape[1]),
            latent_dim=int(b_values.shape[2]),
            output_dtype=output_dtype,
            device=b_values.device,
        )
    )


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
            if attr == "_b12x_absorb_uk_rhs" and _module_uses_mxfp8_mla_query(module):
                continue
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


def warmup_mxfp8_mla_query(
    model: torch.nn.Module,
    *,
    m_values: Iterable[int] = range(1, 33),
) -> int:
    mla_query = _import_mxfp8_mla_query()
    if mla_query is None:
        return 0

    values = tuple(dict.fromkeys(int(m) for m in m_values if int(m) > 0))
    seen: set[tuple[tuple[int, ...], tuple[int, ...], torch.dtype, torch.device]] = (
        set()
    )
    warmed = 0
    for module in model.modules():
        if not _module_uses_mxfp8_mla_query(module):
            continue
        rhs = getattr(module, "_b12x_absorb_uk_rhs", None)
        output_dtype = getattr(module, "_mxfp8_mla_query_output_dtype", None)
        if rhs is None or output_dtype not in (
            torch.bfloat16,
            torch.float8_e4m3fn,
        ):
            continue
        b_values, b_scales = rhs
        signature = (
            tuple(b_values.shape),
            tuple(b_scales.shape),
            output_dtype,
            b_values.device,
        )
        if signature in seen:
            continue
        seen.add(signature)
        warmed += mla_query.prewarm(
            rhs,
            values,
            output_dtype=output_dtype,
        )
    return warmed


__all__ = [
    "can_implement_b12x_mxfp8_bmm",
    "can_implement_mxfp8_mla_query",
    "run_b12x_mxfp8_bmm",
    "run_mxfp8_mla_query",
    "warmup_b12x_mla_mxfp8_bmm",
    "warmup_mxfp8_mla_query",
]
