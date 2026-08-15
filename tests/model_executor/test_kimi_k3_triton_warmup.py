# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.model_executor.warmup.kimi_k3_triton_warmup import (
    _warm_recurrent_kda,
)
from vllm.models.kimi_k3.nvidia.ops.third_party.kda import fused_recurrent


def test_speculative_kda_warmup_before_kv_cache_binding(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        fused_recurrent,
        "get_fused_recurrent_kda_fwd_warmup_profiles",
        lambda _num_heads: (2,),
    )
    monkeypatch.setattr(
        fused_recurrent,
        "fused_recurrent_kda",
        lambda **kwargs: calls.append(kwargs),
    )

    layer = SimpleNamespace(
        num_spec=7,
        local_num_heads=2,
        head_dim=4,
        A_log=torch.empty(2, dtype=torch.float32),
        dt_bias=torch.empty(8, dtype=torch.float32),
        gate_lower_bound=-10.0,
        get_state_shape=lambda: ((10, 4), (2, 4, 4)),
        get_state_dtype=lambda: (torch.bfloat16, torch.float32),
    )

    _warm_recurrent_kda(layer, torch.bfloat16)

    assert len(calls) == 1
    call = calls[0]
    assert call["initial_state"].shape == (1, 2, 4, 4)
    assert call["initial_state"].dtype == torch.float32
    assert call["q"].shape == (1, 16, 2, 4)
    assert call["ssm_state_indices"].shape == (2, 8)
