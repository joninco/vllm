# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.distributed.device_communicators import custom_all_reduce
from vllm.v1.worker.gpu.spec_decode.dspark import speculator as speculator_module
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator


def _captured_speculator() -> SimpleNamespace:
    return SimpleNamespace(
        use_draft_token_capacity=False,
        draft_logits=None,
        _draft_topk=None,
        _use_local_draft_argmax=False,
        _capture_sharded_markov=True,
        _markov_outside_cudagraph=False,
        max_num_reqs=8,
        num_speculative_steps=7,
        vllm_config=object(),
        draft_model_config=SimpleNamespace(hf_config=SimpleNamespace(markov_rank=128)),
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )


def test_captured_markov_allreduce_probe_covers_every_draft_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe_shapes = []

    class ActiveAllreduce:
        def should_custom_ar(self, probe):
            probe_shapes.append(tuple(probe.shape))
            return True

    model = SimpleNamespace(
        supports_local_draft_argmax=lambda: True,
        draft_id_to_target_id=None,
        _b12x_dspark_argmax_enabled=True,
        model=SimpleNamespace(markov_head=SimpleNamespace(replicate_w1=False)),
    )
    speculator = _captured_speculator()
    monkeypatch.setenv("VLLM_DSPARK_SHARD_MARKOV_HEAD", "1")
    monkeypatch.setattr(
        speculator_module,
        "load_dspark_model",
        lambda _target_model, _config: model,
    )
    monkeypatch.setattr(
        custom_all_reduce,
        "get_active_b12x_pcie_allreduce",
        lambda: ActiveAllreduce(),
        raising=False,
    )

    loaded = DSparkSpeculator.load_draft_model(
        speculator,
        target_model=object(),
        target_attn_layer_names=set(),
    )

    assert loaded is model
    assert speculator._use_local_draft_argmax
    assert probe_shapes == [(56, 128)]


def test_captured_markov_with_replicated_w1_needs_no_allreduce_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = SimpleNamespace(
        supports_local_draft_argmax=lambda: True,
        draft_id_to_target_id=None,
        _b12x_dspark_argmax_enabled=True,
        model=SimpleNamespace(markov_head=SimpleNamespace(replicate_w1=True)),
    )
    speculator = _captured_speculator()
    monkeypatch.setenv("VLLM_DSPARK_SHARD_MARKOV_HEAD", "1")
    monkeypatch.setattr(
        speculator_module,
        "load_dspark_model",
        lambda _target_model, _config: model,
    )
    monkeypatch.setattr(
        custom_all_reduce,
        "get_active_b12x_pcie_allreduce",
        lambda: pytest.fail("replicated W1 must not inspect TP all-reduce"),
        raising=False,
    )

    loaded = DSparkSpeculator.load_draft_model(
        speculator,
        target_model=object(),
        target_attn_layer_names=set(),
    )

    assert loaded is model
    assert speculator._use_local_draft_argmax
