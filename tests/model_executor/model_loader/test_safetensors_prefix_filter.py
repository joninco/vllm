# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
import torch
from safetensors.torch import save_file

from vllm.model_executor.model_loader.weight_utils import (
    filter_safetensors_files_by_weight_name_prefixes,
    safetensors_weights_iterator,
)


def test_safetensors_prefix_filter_uses_index_and_skips_other_tensors(tmp_path):
    shard_a = tmp_path / "model-00001-of-00002.safetensors"
    shard_b = tmp_path / "model-00002-of-00002.safetensors"
    save_file(
        {
            "model.layers.0.weight": torch.ones(1),
        },
        shard_a,
    )
    save_file(
        {
            "mtp.0.weight": torch.ones(1),
            "model.layers.1.weight": torch.ones(1),
        },
        shard_b,
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {
                    "model.layers.0.weight": shard_a.name,
                    "mtp.0.weight": shard_b.name,
                    "model.layers.1.weight": shard_b.name,
                },
            }
        )
    )

    selected_files = filter_safetensors_files_by_weight_name_prefixes(
        [str(shard_a), str(shard_b)],
        str(tmp_path),
        "model.safetensors.index.json",
        ("mtp.",),
    )

    assert selected_files == [str(shard_b)]
    weights = dict(
        safetensors_weights_iterator(
            selected_files,
            use_tqdm_on_load=False,
            weight_name_prefixes=("mtp.",),
        )
    )
    assert set(weights) == {"mtp.0.weight"}


def test_safetensors_prefix_filter_fails_when_index_has_no_matches(tmp_path):
    shard = tmp_path / "model-00001-of-00001.safetensors"
    save_file({"model.layers.0.weight": torch.ones(1)}, shard)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {
                    "model.layers.0.weight": shard.name,
                },
            }
        )
    )

    with pytest.raises(RuntimeError, match="matching prefixes"):
        filter_safetensors_files_by_weight_name_prefixes(
            [str(shard)],
            str(tmp_path),
            "model.safetensors.index.json",
            ("mtp.",),
        )
