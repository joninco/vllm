# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.envs as envs
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    _can_alias_fused_moe_output,
)


def test_fused_moe_output_alias_can_be_disabled(monkeypatch) -> None:
    allocated = torch.empty((2, 8), dtype=torch.bfloat16)
    caller_output = torch.empty_like(allocated)

    monkeypatch.setattr(envs, "VLLM_DISABLE_FUSED_MOE_OUTPUT_ALIAS", False)
    assert _can_alias_fused_moe_output(caller_output, allocated)

    monkeypatch.setattr(envs, "VLLM_DISABLE_FUSED_MOE_OUTPUT_ALIAS", True)
    assert not _can_alias_fused_moe_output(caller_output, allocated)


def test_fused_moe_output_alias_requires_matching_storage() -> None:
    allocated = torch.empty((2, 8), dtype=torch.bfloat16)

    assert not _can_alias_fused_moe_output(None, allocated)
    assert not _can_alias_fused_moe_output(
        torch.empty((1, 8), dtype=torch.bfloat16), allocated
    )
    assert not _can_alias_fused_moe_output(
        torch.empty((2, 8), dtype=torch.float32), allocated
    )
