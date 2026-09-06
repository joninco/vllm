# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.config import CUDAGraphMode
from vllm.model_executor.layers import l2_prefetch


class _Linear(nn.Module):
    def __init__(self, rows: int = 4, columns: int = 4) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(rows, columns, dtype=torch.bfloat16),
            requires_grad=False,
        )


class _AttentionWrapper(nn.Module):
    def __init__(self, *, skip_topk: bool) -> None:
        super().__init__()
        self.indexer = nn.Module()
        self.indexer.wq_b = _Linear()
        self.skip_topk = skip_topk


class _Attention(nn.Module):
    def __init__(self, *, skip_topk: bool) -> None:
        super().__init__()
        self.fused_qkv_a_proj = _Linear()
        self.q_b_proj = _Linear()
        wrapper = _AttentionWrapper(skip_topk=skip_topk)
        self.indexer = wrapper.indexer
        self.skip_topk = wrapper.skip_topk


class _Mlp(nn.Module):
    def __init__(self, *, moe: bool) -> None:
        super().__init__()
        if moe:
            self.experts = nn.Module()


class _DecoderLayer(nn.Module):
    def __init__(self, *, moe: bool, skip_topk: bool) -> None:
        super().__init__()
        self.self_attn = _Attention(skip_topk=skip_topk)
        self.mlp = _Mlp(moe=moe)


@pytest.mark.parametrize(
    ("mode", "capturing", "expected"),
    [
        (CUDAGraphMode.FULL, False, True),
        (CUDAGraphMode.PIECEWISE, True, False),
        (CUDAGraphMode.NONE, True, True),
        (CUDAGraphMode.NONE, False, False),
    ],
)
def test_active_requires_full_graph_or_capture(
    monkeypatch: pytest.MonkeyPatch,
    mode: CUDAGraphMode,
    capturing: bool,
    expected: bool,
) -> None:
    monkeypatch.setattr(l2_prefetch, "ENABLED", True)
    monkeypatch.setattr(l2_prefetch, "MAX_TOKENS", 16)
    monkeypatch.setattr(l2_prefetch, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        l2_prefetch,
        "get_forward_context",
        lambda: SimpleNamespace(cudagraph_runtime_mode=mode),
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)

    assert l2_prefetch._active(16) is expected
    assert not l2_prefetch._active(17)


def test_register_decoder_layers_attaches_fixed_next_layer_weights() -> None:
    layers = nn.ModuleList(
        [
            _DecoderLayer(moe=False, skip_topk=True),
            _DecoderLayer(moe=True, skip_topk=False),
            _DecoderLayer(moe=True, skip_topk=True),
        ]
    )

    assert l2_prefetch.register_attention_layers(layers, 0, len(layers)) == 2

    assert layers[0]._l2_prefetch_weights is None

    dense_targets = layers[1]._l2_prefetch_weights
    assert dense_targets == [
        layers[1].self_attn.fused_qkv_a_proj.weight,
        layers[1].self_attn.q_b_proj.weight,
        layers[1].self_attn.indexer.wq_b.weight,
    ]

    moe_targets = layers[2]._l2_prefetch_weights
    assert moe_targets == [
        layers[2].self_attn.fused_qkv_a_proj.weight,
        layers[2].self_attn.q_b_proj.weight,
    ]


def test_issue_and_join_ignore_cpu_tensors(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args, **kwargs):
        raise AssertionError("CPU tensors must not enter CUDA prefetch operations")

    monkeypatch.setattr(torch.ops.vllm, "l2_weight_prefetch", fail)
    monkeypatch.setattr(torch.ops.vllm, "l2_weight_prefetch_join", fail)
    hidden_states = torch.empty(4, 8, dtype=torch.bfloat16)

    l2_prefetch.issue(hidden_states, [hidden_states])
    l2_prefetch.join(hidden_states)
