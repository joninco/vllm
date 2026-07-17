# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import MethodType, SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v4.nvidia.model import DeepseekV4DecoderLayer


def _make_b12x_layer() -> DeepseekV4DecoderLayer:
    return object.__new__(DeepseekV4DecoderLayer)


def test_b12x_mhc_requires_fused_norm_weight() -> None:
    layer = _make_b12x_layer()

    with pytest.raises(RuntimeError, match="requires fused RMSNorm"):
        layer._require_b12x_mhc_norm_weight(None)

    norm_weight = torch.ones(4)

    assert layer._require_b12x_mhc_norm_weight(norm_weight) is norm_weight


def test_b12x_forward_only_passes_bf16_fn_to_post_pre() -> None:
    layer = _make_b12x_layer()
    layer._use_b12x_mhc = True
    layer.attn_norm = SimpleNamespace(
        weight=SimpleNamespace(data=torch.ones(4)),
        variance_epsilon=1e-6,
    )
    layer.ffn_norm = SimpleNamespace(
        weight=SimpleNamespace(data=torch.ones(4)),
        variance_epsilon=1e-6,
    )
    layer.hc_attn_fn = torch.ones(24, 16)
    layer.hc_attn_scale = torch.ones(3)
    layer.hc_attn_base = torch.zeros(24)
    layer.hc_ffn_fn = torch.ones(24, 16)
    layer.hc_ffn_scale = torch.ones(3)
    layer.hc_ffn_base = torch.zeros(24)
    layer.hc_ffn_fn_bf16 = torch.ones(24, 16, dtype=torch.bfloat16)
    calls: list[tuple[str, torch.Tensor | None]] = []

    def hc_pre(
        self: DeepseekV4DecoderLayer,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        calls.append(("pre", None))
        post_mix = torch.zeros(x.shape[0], 4)
        res_mix = torch.zeros(x.shape[0], 4, 4)
        return x, post_mix, res_mix

    def hc_post_pre(
        self: DeepseekV4DecoderLayer,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_eps: float,
        hc_fn_bf16: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        calls.append(("post_pre", hc_fn_bf16))
        return residual, post, comb, x

    layer.hc_pre = MethodType(hc_pre, layer)
    layer.hc_post_pre = MethodType(hc_post_pre, layer)
    layer.attn = lambda positions, x, kv_cache: x
    layer.ffn = lambda x, input_ids: x

    x = torch.ones(2, 4)
    out = DeepseekV4DecoderLayer.forward(layer, x, torch.arange(2), None)

    assert [tuple(t.shape) for t in out] == [(2, 4), (2, 4), (2, 4), (2, 4, 4)]
    assert calls == [("pre", None), ("post_pre", layer.hc_ffn_fn_bf16)]
