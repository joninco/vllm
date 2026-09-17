# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DeepseekV32 MTP draft model declares the checkpoint shards it needs.

The draft model loads only the MTP layers (``model.layers.{num_hidden_layers
+ i}``); the embedding and the lm_head are shared with the target model after
loading. Declaring those prefixes lets the default loader select the shards
holding them from ``model.safetensors.index.json`` instead of reading every
shard of the target model a second time.
"""

import json
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader import weight_utils
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.models.deepseek_v32.nvidia.mtp import DeepseekV32MTP


def _prefixes(num_hidden_layers: int, num_nextn_predict_layers: int) -> tuple[str, ...]:
    mtp = SimpleNamespace(
        config=SimpleNamespace(
            num_hidden_layers=num_hidden_layers,
            num_nextn_predict_layers=num_nextn_predict_layers,
        )
    )
    return DeepseekV32MTP._checkpoint_weight_name_prefixes(mtp)


def test_mtp_prefixes_name_only_the_mtp_layers() -> None:
    assert _prefixes(78, 1) == ("model.layers.78.", "layers.78.")
    assert _prefixes(61, 2) == (
        "model.layers.61.",
        "layers.61.",
        "model.layers.62.",
        "layers.62.",
    )


def test_mtp_loading_opens_only_the_shards_holding_the_mtp_layer(
    tmp_path, monkeypatch
) -> None:
    """A checkpoint with a target shard and an MTP shard: only the latter is opened."""
    target_path = tmp_path / "model-00001-of-00002.safetensors"
    draft_path = tmp_path / "model-00002-of-00002.safetensors"
    target_weights = {
        "model.layers.0.mlp.down_proj.weight": torch.zeros(1),
        "model.embed_tokens.weight": torch.zeros(1),
        "lm_head.weight": torch.zeros(1),
    }
    draft_weights = {
        "model.layers.78.eh_proj.weight": torch.tensor([1.0]),
        "model.layers.78.enorm.weight": torch.tensor([2.0]),
        "model.layers.78.shared_head.norm.weight": torch.tensor([3.0]),
    }
    # The MTP shard also holds the tail of the target model, as in a real
    # checkpoint; the loader yields only the declared prefixes from it.
    tail_weight = {"model.layers.77.post_attention_layernorm.weight": torch.zeros(1)}
    save_file(target_weights, target_path)
    save_file({**draft_weights, **tail_weight}, draft_path)
    weight_map = {name: target_path.name for name in target_weights}
    weight_map.update(
        {name: draft_path.name for name in (*draft_weights, *tail_weight)}
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )

    opened = []
    safe_open = weight_utils.safe_open

    def record_open(filename, **kwargs):
        opened.append(filename)
        return safe_open(filename, **kwargs)

    monkeypatch.setattr(weight_utils, "safe_open", record_open)
    loader = DefaultModelLoader(LoadConfig(load_format="safetensors"))
    loaded = dict(
        loader.get_all_weights(
            SimpleNamespace(model=str(tmp_path), revision=None),
            SimpleNamespace(checkpoint_weight_name_prefixes=_prefixes(78, 1)),
        )
    )

    assert opened == [str(draft_path)]
    assert loaded.keys() == draft_weights.keys()
    for name, weight in loaded.items():
        torch.testing.assert_close(weight, draft_weights[name])
