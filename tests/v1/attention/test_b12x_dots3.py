# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from vllm.models.dots3_note.nvidia import b12x_attention
from vllm.models.dots3_note.nvidia import model as dots3_model
from vllm.models.dots3_note.nvidia.b12x_attention import (
    B12xHybridMLABackend,
    B12xHybridMLASlidingBackend,
)
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.registry import AttentionBackendEnum


def _dots3_hf_config() -> SimpleNamespace:
    return SimpleNamespace(
        model_type="dots3_note",
        num_attention_heads=128,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        index_topk=2048,
        swa_num_attention_heads=64,
        swa_kv_lora_rank=1024,
        swa_qk_nope_head_dim=192,
        swa_qk_rope_head_dim=64,
        swa_v_head_dim=128,
        sliding_window_size=513,
    )


def _vllm_config(*, mtp: bool = False) -> SimpleNamespace:
    target = SimpleNamespace(
        hf_text_config=_dots3_hf_config(),
        dtype=torch.bfloat16,
    )
    model = target
    speculative = None
    if mtp:
        model = SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type="deepseek_mtp"),
            dtype=torch.bfloat16,
        )
        speculative = SimpleNamespace(target_model_config=target)
    return SimpleNamespace(
        model_config=model,
        speculative_config=speculative,
        parallel_config=SimpleNamespace(
            tensor_parallel_size=8,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        cache_config=SimpleNamespace(block_size=64, cache_dtype="fp8"),
    )


@pytest.mark.parametrize(
    ("enabled", "pipeline_parallel_size", "expected"),
    [(False, 1, False), (True, 1, True), (True, 2, False)],
)
def test_dots3_model_initializes_inherited_sequence_parallel_contract(
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
    pipeline_parallel_size: int,
    expected: bool,
) -> None:
    class StopConstruction(Exception):
        pass

    def stop_construction(*args, **kwargs) -> None:
        raise StopConstruction

    monkeypatch.setattr(dots3_model.torch, "empty", stop_construction)
    instance = dots3_model.Dots3NoteModel.__new__(dots3_model.Dots3NoteModel)
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(vocab_size=1, index_topk=2048)
        ),
        quant_config=None,
        parallel_config=SimpleNamespace(
            use_sequence_parallel_moe=enabled,
            pipeline_parallel_size=pipeline_parallel_size,
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=1),
    )
    with pytest.raises(StopConstruction):
        dots3_model.Dots3NoteModel.__init__(instance, vllm_config=config)
    assert instance.use_sequence_parallel is expected


@pytest.mark.parametrize(
    ("enabled", "pipeline_parallel_size", "expected"),
    [(False, 1, False), (True, 1, True), (True, 2, False)],
)
def test_dots3_layer_initializes_inherited_sequence_parallel_contract(
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
    pipeline_parallel_size: int,
    expected: bool,
) -> None:
    class StopConstruction(Exception):
        pass

    def stop_construction(*args, **kwargs) -> None:
        raise StopConstruction

    monkeypatch.setattr(dots3_model, "Dots3NoteSlidingAttention", stop_construction)
    instance = dots3_model.Dots3NoteDecoderLayer.__new__(
        dots3_model.Dots3NoteDecoderLayer
    )
    config = SimpleNamespace(hidden_size=1, layer_types=["sliding_attention"])
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=config),
        quant_config=None,
        parallel_config=SimpleNamespace(
            use_sequence_parallel_moe=enabled,
            pipeline_parallel_size=pipeline_parallel_size,
        ),
    )
    with pytest.raises(StopConstruction):
        dots3_model.Dots3NoteDecoderLayer.__init__(
            instance,
            vllm_config=vllm_config,
            prefix="layers.0",
        )
    assert instance.use_sequence_parallel is expected


def test_b12x_dots3_backend_is_registered_for_both_layer_types() -> None:
    assert AttentionBackendEnum.B12X_HYBRID_MLA.get_class() is B12xHybridMLABackend
    assert B12xHybridMLABackend.get_name() == "B12X_HYBRID_MLA"
    assert B12xHybridMLABackend.get_supported_head_sizes() == [576]
    assert B12xHybridMLASlidingBackend.get_supported_head_sizes() == [1088]
    assert B12xHybridMLABackend.supports_block_size(64)
    assert not B12xHybridMLABackend.supports_block_size(16)
    assert B12xHybridMLABackend.supports_compute_capability(DeviceCapability(12, 0))
    assert B12xHybridMLABackend.supports_compute_capability(DeviceCapability(12, 1))
    assert not B12xHybridMLABackend.supports_compute_capability(DeviceCapability(10, 0))


@pytest.mark.parametrize("mtp", [False, True])
def test_b12x_dots3_exact_config_is_accepted(mtp: bool) -> None:
    assert b12x_attention._validate_b12x_hybrid_config(_vllm_config(mtp=mtp)) is None


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (
            lambda cfg: setattr(cfg.model_config.hf_text_config, "model_type", "llama"),
            "supports only dots3_note",
        ),
        (
            lambda cfg: setattr(cfg.parallel_config, "tensor_parallel_size", 4),
            "qualified only for TP8",
        ),
        (
            lambda cfg: setattr(cfg.parallel_config, "decode_context_parallel_size", 2),
            "does not support decode context parallelism",
        ),
        (
            lambda cfg: setattr(
                cfg.parallel_config, "prefill_context_parallel_size", 2
            ),
            "does not support prefill context parallelism",
        ),
        (
            lambda cfg: setattr(cfg.cache_config, "block_size", 16),
            "requires block size 64",
        ),
        (
            lambda cfg: setattr(cfg.cache_config, "cache_dtype", "bfloat16"),
            "requires E4M3 FP8",
        ),
        (
            lambda cfg: setattr(cfg.model_config, "dtype", torch.float16),
            "requires BF16 activations",
        ),
        (
            lambda cfg: setattr(cfg.model_config.hf_text_config, "index_topk", 1024),
            "requires the exact DSA/SWA geometry",
        ),
    ],
)
def test_b12x_dots3_config_rejects_every_contract_mismatch(
    mutation, expected: str
) -> None:
    config = deepcopy(_vllm_config())
    mutation(config)
    reason = b12x_attention._validate_b12x_hybrid_config(config)
    assert reason is not None
    assert expected in reason
