# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

import vllm.config.virtual_tp as virtual_tp
from vllm.config import ParallelConfig, set_current_vllm_config
from vllm.config.speculative import SpeculativeConfig
from vllm.config.virtual_tp import (
    VIRTUAL_TP_PLAN_ATTR,
    maybe_apply_b12x_virtual_tp_padding,
)
from vllm.model_executor.virtual_tp import (
    get_virtual_tp_axis_local_size,
    get_virtual_tp_axis_shard_size,
    get_virtual_tp_vocab_padding_size,
    pad_or_narrow_weight,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum


class FakeModelConfig:
    def __init__(self):
        self.hf_text_config = SimpleNamespace(
            model_type="deepseek_v4",
            num_attention_heads=128,
            o_groups=16,
            moe_intermediate_size=3072,
            n_routed_experts=384,
            n_shared_experts=1,
            vocab_size=129280,
        )
        self.hf_config = self.hf_text_config
        self.model_arch_config = self.get_model_arch_config()

    def get_model_arch_config(self):
        return SimpleNamespace(
            total_num_attention_heads=self.hf_text_config.num_attention_heads,
        )


class FakeGlmDsaModelConfig:
    def __init__(self):
        self.hf_text_config = SimpleNamespace(
            model_type="glm_moe_dsa",
            architectures=["GlmMoeDsaForCausalLM"],
            hidden_size=8192,
            num_attention_heads=64,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            kv_lora_rank=512,
            index_topk=256,
            moe_intermediate_size=2048,
            n_routed_experts=256,
            n_shared_experts=1,
            vocab_size=129280,
        )
        self.hf_config = self.hf_text_config
        self.model_arch_config = self.get_model_arch_config()

    def get_model_arch_config(self):
        return SimpleNamespace(
            total_num_attention_heads=self.hf_text_config.num_attention_heads,
        )


class FakeWrappedGlmDsaModelConfig(FakeGlmDsaModelConfig):
    def __init__(self):
        super().__init__()
        self.hf_config = SimpleNamespace(
            model_type="wrapper",
            architectures=["GlmMoeDsaForCausalLM"],
            text_config=self.hf_text_config,
            num_attention_heads=self.hf_text_config.num_attention_heads,
            moe_intermediate_size=self.hf_text_config.moe_intermediate_size,
            vocab_size=self.hf_text_config.vocab_size,
        )
        self.model_arch_config = self.get_model_arch_config()


class FakeUnsupportedModelConfig:
    def __init__(self):
        self.hf_text_config = SimpleNamespace(
            model_type="llama",
            num_attention_heads=64,
            moe_intermediate_size=2048,
            n_routed_experts=8,
            vocab_size=32000,
        )
        self.hf_config = self.hf_text_config
        self.model_arch_config = self.get_model_arch_config()

    def get_model_arch_config(self):
        return SimpleNamespace(
            total_num_attention_heads=self.hf_text_config.num_attention_heads,
        )


class FakeAlignedGlmDsaModelConfig(FakeGlmDsaModelConfig):
    def __init__(self):
        super().__init__()
        self.hf_text_config.num_attention_heads = 96
        self.hf_text_config.moe_intermediate_size = 2112
        self.hf_text_config.vocab_size = 129408
        self.model_arch_config = self.get_model_arch_config()


def _fake_vllm_config(
    *,
    model_config: Any | None = None,
    moe_backend: str = "b12x",
    tensor_parallel_size: int = 10,
) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=model_config or FakeModelConfig(),
        parallel_config=ParallelConfig(
            tensor_parallel_size=tensor_parallel_size,
        ),
        kernel_config=SimpleNamespace(moe_backend=moe_backend),
        attention_config=SimpleNamespace(
            backend=AttentionBackendEnum.B12X_MLA_SPARSE,
        ),
    )


def test_b12x_virtual_tp_padding_deepseek_v4_pro_tp10():
    vllm_config = _fake_vllm_config()

    maybe_apply_b12x_virtual_tp_padding(cast(Any, vllm_config))

    text_config = vllm_config.model_config.hf_text_config
    assert text_config.num_attention_heads == 160
    assert text_config.o_groups == 20
    assert text_config.moe_intermediate_size == 3200
    assert text_config.vocab_size == 129280
    assert vllm_config.model_config.model_arch_config.total_num_attention_heads == 160

    plan = getattr(text_config, VIRTUAL_TP_PLAN_ATTR)
    assert plan["attention_heads"] == {
        "original_size": 128,
        "padded_size": 160,
        "tp_size": 10,
        "local_size": 16,
    }
    assert plan["output_groups"] == {
        "original_size": 16,
        "padded_size": 20,
        "tp_size": 10,
        "local_size": 2,
        "heads_per_group": 8,
    }
    assert plan["moe_intermediate_size"] == {
        "original_size": 3072,
        "padded_size": 3200,
        "tp_size": 10,
        "local_size": 320,
    }
    assert plan["moe_intermediate_size"]["local_size"] % 32 == 0
    assert plan["shared_expert_intermediate_size"] == {
        "original_size": 3072,
        "padded_size": 3840,
        "tp_size": 10,
        "local_size": 384,
    }
    assert plan["shared_expert_intermediate_size"]["local_size"] % 128 == 0
    assert plan["vocab_size"] == {
        "original_size": 129280,
        "padded_size": 129280,
        "tp_size": 10,
        "local_size": 12928,
        "padding_size": 320,
    }


def test_b12x_virtual_tp_padding_logs_when_triggered(
    monkeypatch: pytest.MonkeyPatch,
):
    vllm_config = _fake_vllm_config()
    logs: list[str] = []

    def warning(message: str, *args: Any) -> None:
        logs.append(message % args)

    monkeypatch.setattr(virtual_tp.logger, "warning", warning)
    maybe_apply_b12x_virtual_tp_padding(cast(Any, vllm_config))

    assert any("Automatically enabled B12X virtual TP padding" in log for log in logs)
    assert any("attention heads 128 -> 160" in log for log in logs)
    assert any("MoE intermediate size 3072 -> 3200" in log for log in logs)


def test_b12x_virtual_tp_vocab_padding_deepseek_v4_pro_tp3():
    vllm_config = _fake_vllm_config(tensor_parallel_size=3)

    maybe_apply_b12x_virtual_tp_padding(cast(Any, vllm_config))

    text_config = vllm_config.model_config.hf_text_config
    assert text_config.num_attention_heads == 144
    assert text_config.vocab_size == 129280

    plan = getattr(text_config, VIRTUAL_TP_PLAN_ATTR)
    assert plan["attention_heads"] == {
        "original_size": 128,
        "padded_size": 144,
        "tp_size": 3,
        "local_size": 48,
    }
    assert plan["vocab_size"] == {
        "original_size": 129280,
        "padded_size": 129408,
        "tp_size": 3,
        "local_size": 43136,
        "padding_size": 192,
    }
    assert plan["output_groups"] == {
        "original_size": 16,
        "padded_size": 18,
        "tp_size": 3,
        "local_size": 6,
        "heads_per_group": 8,
    }


def test_b12x_virtual_tp_moe_padding_deepseek_v4_flash_tp3():
    vllm_config = _fake_vllm_config(tensor_parallel_size=3)
    vllm_config.model_config.hf_text_config.moe_intermediate_size = 2048

    maybe_apply_b12x_virtual_tp_padding(cast(Any, vllm_config))

    text_config = vllm_config.model_config.hf_text_config
    assert text_config.moe_intermediate_size == 2112

    plan = getattr(text_config, VIRTUAL_TP_PLAN_ATTR)
    assert plan["moe_intermediate_size"] == {
        "original_size": 2048,
        "padded_size": 2112,
        "tp_size": 3,
        "local_size": 704,
    }


def test_b12x_virtual_tp_padding_glm_dsa_tp6():
    vllm_config = _fake_vllm_config(
        model_config=FakeGlmDsaModelConfig(),
        tensor_parallel_size=6,
    )

    maybe_apply_b12x_virtual_tp_padding(cast(Any, vllm_config))

    text_config = vllm_config.model_config.hf_text_config
    assert text_config.num_attention_heads == 96
    assert text_config.moe_intermediate_size == 2112
    assert text_config.vocab_size == 129280
    assert text_config.original_num_attention_heads == 64
    assert text_config.original_moe_intermediate_size == 2048
    assert vllm_config.model_config.model_arch_config.total_num_attention_heads == 96

    plan = getattr(text_config, VIRTUAL_TP_PLAN_ATTR)
    assert "output_groups" not in plan
    assert "shared_expert_intermediate_size" not in plan
    assert plan["attention_heads"] == {
        "original_size": 64,
        "padded_size": 96,
        "tp_size": 6,
        "local_size": 16,
    }
    assert plan["moe_intermediate_size"] == {
        "original_size": 2048,
        "padded_size": 2112,
        "tp_size": 6,
        "local_size": 352,
    }
    assert plan["vocab_size"] == {
        "original_size": 129280,
        "padded_size": 129408,
        "tp_size": 6,
        "local_size": 21568,
        "padding_size": 192,
    }


def test_b12x_virtual_tp_padding_glm_dsa_draft_tp6():
    target_model_config = FakeGlmDsaModelConfig()
    draft_model_config = FakeGlmDsaModelConfig()
    spec_config = SimpleNamespace(
        method="mtp",
        target_model_config=target_model_config,
        draft_model_config=draft_model_config,
        draft_parallel_config=ParallelConfig(
            tensor_parallel_size=6,
        ),
    )

    SpeculativeConfig._maybe_apply_virtual_tp_to_draft(cast(Any, spec_config))

    text_config = draft_model_config.hf_text_config
    assert text_config.num_attention_heads == 96
    assert text_config.moe_intermediate_size == 2112
    assert draft_model_config.model_arch_config.total_num_attention_heads == 96


def test_b12x_virtual_tp_padding_updates_distinct_hf_configs():
    vllm_config = _fake_vllm_config(
        model_config=FakeWrappedGlmDsaModelConfig(),
        tensor_parallel_size=6,
    )

    maybe_apply_b12x_virtual_tp_padding(cast(Any, vllm_config))

    root_config = vllm_config.model_config.hf_config
    text_config = vllm_config.model_config.hf_text_config
    plan = getattr(text_config, VIRTUAL_TP_PLAN_ATTR)

    assert root_config.num_attention_heads == 96
    assert root_config.moe_intermediate_size == 2112
    assert root_config.vocab_size == 129280
    assert getattr(root_config, VIRTUAL_TP_PLAN_ATTR) is plan
    assert getattr(root_config.text_config, VIRTUAL_TP_PLAN_ATTR) is plan


def test_b12x_virtual_tp_padding_skips_aligned_config():
    vllm_config = _fake_vllm_config(
        model_config=FakeAlignedGlmDsaModelConfig(),
        tensor_parallel_size=6,
    )

    maybe_apply_b12x_virtual_tp_padding(cast(Any, vllm_config))

    text_config = vllm_config.model_config.hf_text_config
    assert not hasattr(text_config, VIRTUAL_TP_PLAN_ATTR)
    assert text_config.num_attention_heads == 96
    assert text_config.moe_intermediate_size == 2112
    assert text_config.vocab_size == 129408


def test_b12x_virtual_tp_padding_rejects_flashinfer_moe():
    vllm_config = _fake_vllm_config(moe_backend="flashinfer_b12x")

    with pytest.raises(ValueError, match="native B12X MoE"):
        maybe_apply_b12x_virtual_tp_padding(cast(Any, vllm_config))


def test_b12x_virtual_tp_padding_skips_unsupported_models():
    vllm_config = _fake_vllm_config(model_config=FakeUnsupportedModelConfig())

    maybe_apply_b12x_virtual_tp_padding(cast(Any, vllm_config))

    text_config = vllm_config.model_config.hf_text_config
    assert not hasattr(text_config, VIRTUAL_TP_PLAN_ATTR)


def test_virtual_tp_pad_or_narrow_weight_zero_fills_tail():
    current_config = _fake_vllm_config()
    maybe_apply_b12x_virtual_tp_padding(cast(Any, current_config))
    loaded_weight = torch.arange(6).reshape(3, 2)

    with set_current_vllm_config(cast(Any, current_config)):
        padded = pad_or_narrow_weight(loaded_weight, 0, 2, 3)
        local_moe_size = get_virtual_tp_axis_local_size("moe_intermediate_size", -1)
        vocab_padding_size = get_virtual_tp_vocab_padding_size(-1)

    expected = torch.tensor([[4, 5], [0, 0], [0, 0]])
    assert torch.equal(padded, expected)
    assert local_moe_size == 320
    assert vocab_padding_size == 320


def test_virtual_tp_axis_shard_size_uses_stored_tensor_units():
    current_config = _fake_vllm_config()
    maybe_apply_b12x_virtual_tp_padding(cast(Any, current_config))

    with set_current_vllm_config(cast(Any, current_config)):
        assert get_virtual_tp_axis_shard_size("moe_intermediate_size", 320) == 320
        assert get_virtual_tp_axis_shard_size("moe_intermediate_size", 160) == 160
        assert get_virtual_tp_axis_shard_size("moe_intermediate_size", 512) == 320


def test_virtual_tp_pad_or_narrow_weight_is_strict_without_plan():
    loaded_weight = torch.arange(6).reshape(3, 2)

    with pytest.raises(RuntimeError):
        pad_or_narrow_weight(loaded_weight, 0, 2, 3)
