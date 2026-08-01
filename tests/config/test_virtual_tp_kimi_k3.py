# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any, cast

import pytest

from vllm.config import ParallelConfig
from vllm.config.virtual_tp import (
    VIRTUAL_TP_PLAN_ATTR,
    maybe_apply_b12x_virtual_tp_padding,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum


class FakeKimiK3ModelConfig:
    def __init__(self):
        self.hf_text_config = SimpleNamespace(
            model_type="kimi_linear",
            num_attention_heads=96,
            moe_intermediate_size=3072,
            vocab_size=163840,
        )
        self.hf_config = SimpleNamespace(
            model_type="kimi_k3",
            text_config=self.hf_text_config,
        )
        self.model_arch_config = self.get_model_arch_config()

    def get_model_arch_config(self):
        return SimpleNamespace(
            total_num_attention_heads=self.hf_text_config.num_attention_heads,
        )


@pytest.mark.parametrize("tp_size", [6, 12])
def test_b12x_virtual_tp_padding_kimi_k3_vocab(tp_size: int):
    model_config = FakeKimiK3ModelConfig()
    vllm_config = SimpleNamespace(
        model_config=model_config,
        parallel_config=ParallelConfig(tensor_parallel_size=tp_size),
        kernel_config=SimpleNamespace(moe_backend="b12x"),
        attention_config=SimpleNamespace(
            backend=AttentionBackendEnum.B12X_MLA_SPARSE,
        ),
    )

    maybe_apply_b12x_virtual_tp_padding(cast(Any, vllm_config))

    text_config = model_config.hf_text_config
    plan = getattr(text_config, VIRTUAL_TP_PLAN_ATTR)
    assert text_config.num_attention_heads == 96
    assert text_config.moe_intermediate_size == 3072
    assert text_config.vocab_size == 163840
    assert plan["attention_heads"]["padded_size"] == 96
    assert plan["moe_intermediate_size"]["padded_size"] == 3072
    assert plan["vocab_size"] == {
        "original_size": 163840,
        "padded_size": 163968,
        "tp_size": tp_size,
        "local_size": 163968 // tp_size,
        "padding_size": 192,
    }
