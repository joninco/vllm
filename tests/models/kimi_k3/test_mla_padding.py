# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch


def test_kimi_mla_defines_graph_padding_before_output_projection(monkeypatch):
    from vllm.models.kimi_k3.nvidia import mla

    attention = object.__new__(mla.MultiHeadLatentAttention)
    torch.nn.Module.__init__(attention)
    attention.layer_name = "model.layers.0.self_attn"
    attention.rotary_emb = None

    metadata = SimpleNamespace(num_actual_tokens=2, num_decode_tokens=0)
    context = SimpleNamespace(
        attn_metadata={attention.layer_name: metadata},
        slot_mapping={attention.layer_name: torch.arange(4)},
    )
    monkeypatch.setattr(mla, "get_forward_context", lambda: context)

    def write_active_prefill(*args):
        args[-1].fill_(3)

    attention._forward_prefill_fused = write_active_prefill
    output = torch.full((4, 8), 9, dtype=torch.bfloat16)
    attention_method = type(attention)._attention
    invoke_attention = getattr(attention_method, "__wrapped__", attention_method)
    invoke_attention(
        attention,
        torch.arange(4),
        torch.zeros((4, 1, 8), dtype=torch.bfloat16),
        torch.zeros((4, 8), dtype=torch.bfloat16),
        torch.zeros((4, 8), dtype=torch.bfloat16),
        output,
    )

    torch.testing.assert_close(output[:2], torch.full_like(output[:2], 3))
    torch.testing.assert_close(output[2:], torch.zeros_like(output[2:]))
