# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate the activation dtype selected for packed MLA projections."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.attention.mla_attention import (
    _get_kv_b_proj_input_dtype,
)
from vllm.platforms import current_platform


@pytest.mark.parametrize(
    ("weight_dtype", "use_fp8_prefill", "expected"),
    [
        (torch.uint8, False, None),
        (torch.int32, False, torch.bfloat16),
        (torch.bfloat16, False, torch.bfloat16),
        (torch.float16, False, torch.float16),
        (current_platform.fp8_dtype(), False, None),
        (current_platform.fp8_dtype(), True, current_platform.fp8_dtype()),
    ],
)
def test_kv_b_proj_input_dtype_for_packed_weights(
    weight_dtype: torch.dtype,
    use_fp8_prefill: bool,
    expected: torch.dtype | None,
) -> None:
    projection = SimpleNamespace(
        weight=SimpleNamespace(dtype=weight_dtype),
        params_dtype=torch.bfloat16,
    )

    assert _get_kv_b_proj_input_dtype(projection, use_fp8_prefill) is expected


def test_kv_b_proj_input_dtype_without_weight_uses_parameter_dtype() -> None:
    projection = SimpleNamespace(params_dtype=torch.bfloat16)

    assert _get_kv_b_proj_input_dtype(projection, False) is torch.bfloat16
