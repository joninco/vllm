# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models.qwen3_dspark import DSparkMarkovHead
from vllm.model_executor.models.registry import ModelRegistry
from vllm.model_executor.warmup.kimi_k3_triton_warmup import _get_kda_layer
from vllm.models.kimi_k3.nvidia import dspark_mla
from vllm.models.kimi_k3.nvidia import kda as kimi_k3_kda
from vllm.models.kimi_k3.nvidia.dspark_mla import K3DSparkForCausalLM, K3DSparkModel


def test_dspark_mla_uses_compile_free_model_entrypoint():
    assert ModelRegistry._try_load_model_cls("K3DSparkModel") is K3DSparkForCausalLM
    assert not issubclass(K3DSparkModel, TorchCompileWithNoGuardsWrapper)


@pytest.mark.parametrize(
    ("checkpoint_name", "runtime_name", "shard_id"),
    [
        (
            "layers.0.self_attn.q_a_proj.weight",
            "model.layers.0.self_attn.fused_qkv_a_proj.weight",
            0,
        ),
        (
            "layers.0.self_attn.kv_a_proj_with_mqa.weight",
            "model.layers.0.self_attn.fused_qkv_a_proj.weight",
            1,
        ),
        (
            "layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.gate_up_proj.weight",
            0,
        ),
        (
            "layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.gate_up_proj.weight",
            1,
        ),
        ("context_proj.weight", "model.context_proj.weight", None),
    ],
)
def test_dspark_mla_checkpoint_weight_mapping(checkpoint_name, runtime_name, shard_id):
    assert K3DSparkForCausalLM.hf_to_vllm_mapper._map_name_with_shard(
        checkpoint_name
    ) == (runtime_name, shard_id)


def test_dspark_mla_shares_frozen_target_weights_and_skips_training_head():
    assert not K3DSparkForCausalLM.has_own_embed_tokens
    assert not K3DSparkForCausalLM.has_own_lm_head
    assert set(K3DSparkForCausalLM.checkpoint_skip_substrs) == {
        "confidence_head",
        "embed_tokens",
        "lm_head",
    }


@pytest.mark.cpu_test
def test_dspark_markov_head_is_replicated(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.model_executor.layers import logits_processor, vocab_parallel_embedding

    monkeypatch.setattr(
        vocab_parallel_embedding, "get_tensor_model_parallel_rank", lambda: 3
    )
    monkeypatch.setattr(
        vocab_parallel_embedding,
        "get_tensor_model_parallel_world_size",
        lambda: 8,
    )
    monkeypatch.setattr(
        logits_processor,
        "get_current_vllm_config",
        lambda: SimpleNamespace(model_config=None),
    )
    monkeypatch.delenv("VLLM_DSPARK_SHARD_MARKOV_HEAD", raising=False)
    monkeypatch.delenv("VLLM_DSPARK_REPLICATE_MARKOV_W1", raising=False)

    head = DSparkMarkovHead(128, 128, 8, prefix="markov_head")
    assert head.markov_w2.tp_size == 1
    assert head.markov_w1.weight.shape == (128, 8)
    assert head.markov_w2.weight.shape == (128, 8)

    def fail_collective(*args, **kwargs):
        raise AssertionError("replicated Markov head must not invoke TP collectives")

    monkeypatch.setattr(
        vocab_parallel_embedding,
        "tensor_model_parallel_all_reduce",
        fail_collective,
    )
    logits_processor = LogitsProcessor(128)
    monkeypatch.setattr(logits_processor, "_gather_logits", fail_collective)

    markov_embed = head.embed(torch.tensor([1, 2]))
    bias = head.bias(markov_embed, logits_processor)
    assert markov_embed.shape == (2, 8)
    assert bias.shape == (2, 128)


@pytest.mark.cpu_test
@pytest.mark.parametrize("replicate_w1", [False, True])
def test_dspark_markov_head_shards_vocabulary_weights(
    monkeypatch: pytest.MonkeyPatch,
    replicate_w1: bool,
) -> None:
    from vllm.model_executor.layers import logits_processor, vocab_parallel_embedding

    monkeypatch.setenv("VLLM_DSPARK_SHARD_MARKOV_HEAD", "1")
    monkeypatch.setenv(
        "VLLM_DSPARK_REPLICATE_MARKOV_W1",
        "1" if replicate_w1 else "0",
    )
    monkeypatch.setattr(
        vocab_parallel_embedding, "get_tensor_model_parallel_rank", lambda: 3
    )
    monkeypatch.setattr(
        vocab_parallel_embedding,
        "get_tensor_model_parallel_world_size",
        lambda: 8,
    )
    monkeypatch.setattr(
        logits_processor,
        "get_current_vllm_config",
        lambda: SimpleNamespace(model_config=None),
    )

    head = DSparkMarkovHead(128, 128, 8, prefix="markov_head")

    assert head.shard_across_tp
    assert head.replicate_w1 is replicate_w1
    assert head.markov_w2.tp_size == 8
    assert head.markov_w2.weight.shape == (16, 8)
    if replicate_w1:
        assert isinstance(head.markov_w1, nn.Embedding)
        assert head.markov_w1.weight.shape == (128, 8)
    else:
        assert isinstance(
            head.markov_w1,
            vocab_parallel_embedding.VocabParallelEmbedding,
        )
        assert head.markov_w1.weight.shape == (16, 8)

    processor = LogitsProcessor(128)
    local_bias = head.local_bias(torch.ones((2, 8)), processor)
    assert local_bias.shape == (2, 16)


@pytest.mark.cpu_test
def test_replicated_markov_w1_requires_sharded_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VLLM_DSPARK_SHARD_MARKOV_HEAD", raising=False)
    monkeypatch.setenv("VLLM_DSPARK_REPLICATE_MARKOV_W1", "1")

    with pytest.raises(ValueError, match="requires VLLM_DSPARK_SHARD_MARKOV_HEAD"):
        DSparkMarkovHead(128, 128, 8, prefix="markov_head")


@pytest.mark.cpu_test
def test_k3_dspark_uses_replicated_markov_head(monkeypatch: pytest.MonkeyPatch):
    markov_head_calls = []
    context_kv_proj_calls = []

    class DummyModule(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    def make_markov_head(*args, **kwargs):
        markov_head_calls.append((args, kwargs))
        return DummyModule()

    def make_context_kv_proj(*args, **kwargs):
        context_kv_proj_calls.append((args, kwargs))
        return DummyModule()

    monkeypatch.setattr(dspark_mla, "get_draft_quant_config", lambda _: None)
    monkeypatch.setattr(dspark_mla, "ReplicatedLinear", DummyModule)
    monkeypatch.setattr(dspark_mla, "MergedColumnParallelLinear", make_context_kv_proj)
    monkeypatch.setattr(dspark_mla, "RMSNorm", DummyModule)
    monkeypatch.setattr(dspark_mla, "K3DSparkDecoderLayer", DummyModule)
    monkeypatch.setattr(dspark_mla, "DSparkMarkovHead", make_markov_head)

    config = SimpleNamespace(
        target_hidden_size=16,
        num_target_layers=2,
        hidden_size=8,
        kv_lora_rank=3,
        qk_rope_head_dim=1,
        rms_norm_eps=1e-6,
        num_hidden_layers=1,
        vocab_size=128,
        draft_vocab_size=128,
        markov_rank=4,
    )
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=config)
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
    )

    K3DSparkModel(vllm_config=vllm_config, start_layer_id=0, prefix="model")

    assert len(markov_head_calls) == 1
    assert context_kv_proj_calls == [
        (
            (8, [4]),
            {
                "bias": False,
                "return_bias": False,
                "quant_config": None,
                "prefix": "model.layers.0.self_attn.fused_qkv_a_proj",
                "disable_tp": True,
            },
        )
    ]


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("tp_size", "expect_sharded"),
    [(1, False), (8, True), (12, False), (16, True)],
)
def test_k3_dspark_context_projection_uses_divisible_tp_geometry(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    expect_sharded: bool,
) -> None:
    context_projection_calls = []

    class DummyModule(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    def make_sharded_projection(*args, **kwargs):
        context_projection_calls.append((args, kwargs))
        return DummyModule()

    monkeypatch.setattr(dspark_mla, "get_draft_quant_config", lambda _: None)
    monkeypatch.setattr(dspark_mla, "ColumnParallelLinear", make_sharded_projection)
    monkeypatch.setattr(dspark_mla, "ReplicatedLinear", DummyModule)
    monkeypatch.setattr(dspark_mla, "MergedColumnParallelLinear", DummyModule)
    monkeypatch.setattr(dspark_mla, "RMSNorm", DummyModule)
    monkeypatch.setattr(dspark_mla, "K3DSparkDecoderLayer", DummyModule)
    monkeypatch.setattr(dspark_mla, "DSparkMarkovHead", DummyModule)

    config = SimpleNamespace(
        target_hidden_size=7168,
        num_target_layers=5,
        hidden_size=7168,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        rms_norm_eps=1e-6,
        num_hidden_layers=1,
        vocab_size=163840,
        draft_vocab_size=163840,
        markov_rank=256,
    )
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=config)
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=tp_size),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
    )

    model = K3DSparkModel(
        vllm_config=vllm_config,
        start_layer_id=0,
        prefix="model",
    )

    assert model.context_proj_sharded is expect_sharded
    assert bool(context_projection_calls) is expect_sharded
    if expect_sharded:
        args, kwargs = context_projection_calls[0]
        assert args == (7168 * 5, 7168)
        assert kwargs["gather_output"] is True


def test_context_kv_weights_are_loaded_as_merged_linear_shards():
    weights = [
        (
            "layers.0.self_attn.kv_a_proj_with_mqa.weight_packed",
            torch.arange(4),
        ),
        (
            "layers.1.self_attn.kv_a_proj_with_mqa.weight_scale",
            torch.tensor(0.5),
        ),
    ]

    duplicated = dspark_mla._duplicate_context_kv_weights(weights, 2)
    mapped = list(K3DSparkForCausalLM.hf_to_vllm_mapper.apply(duplicated))

    assert [name for name, _ in mapped] == [
        "model.layers.0.self_attn.fused_qkv_a_proj.weight_packed",
        "model.context_kv_proj.weight_packed",
        "model.layers.1.self_attn.fused_qkv_a_proj.weight_scale",
        "model.context_kv_proj.weight_scale",
    ]
    assert [weight.shard_id for _, weight in mapped] == [1, 0, 1, 1]
    assert mapped[0][1].data_ptr() == mapped[1][1].data_ptr()
    assert mapped[2][1].data_ptr() == mapped[3][1].data_ptr()


def test_kda_warmup_ignores_cacheless_dspark_layers(
    monkeypatch: pytest.MonkeyPatch,
):
    class FakeKDA:
        kv_cache: list[object]

    draft_layer = FakeKDA()
    target_layer = FakeKDA()
    target_layer.kv_cache = []
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            compilation_config=SimpleNamespace(
                static_forward_context={
                    "draft.layers.0": draft_layer,
                    "target.layers.0": target_layer,
                }
            )
        )
    )
    monkeypatch.setattr(kimi_k3_kda, "KimiK3DeltaAttention", FakeKDA)

    assert _get_kda_layer(worker) is target_layer
