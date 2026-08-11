# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import glob
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

import vllm.model_executor.model_loader.weight_utils as weight_utils
from vllm.model_executor.model_loader.weight_utils import (
    download_weights_from_hf,
    instanttensor_weights_iterator,
    safetensors_weights_iterator,
)
from vllm.platforms import current_platform


@pytest.mark.parametrize(
    ("setting", "expected_copy", "expected_borrowed"),
    [("0", False, True), ("1", True, False)],
)
def test_instanttensor_copy_contract(
    setting, expected_copy, expected_borrowed, monkeypatch
):
    tensor = torch.ones(4)
    observed: dict[str, object] = {}

    class FakeReader:
        total_tensor_size = tensor.numel() * tensor.element_size()

        def keys(self):
            return ["weight"]

        def tensors(self):
            yield "weight", tensor

    @contextmanager
    def fake_safe_open(files, *, framework, device, process_group, copy):
        observed.update(
            files=files,
            framework=framework,
            device=device,
            process_group=process_group,
            copy=copy,
        )
        yield FakeReader()

    def no_world_group():
        raise AssertionError

    monkeypatch.setenv("INSTANTTENSOR_COPY", setting)
    monkeypatch.setattr(
        weight_utils,
        "current_platform",
        SimpleNamespace(is_cuda=lambda: True, current_device=lambda: 0),
    )
    monkeypatch.setattr(weight_utils, "get_world_group", no_world_group)
    monkeypatch.setitem(
        sys.modules, "instanttensor", SimpleNamespace(safe_open=fake_safe_open)
    )

    loaded = list(instanttensor_weights_iterator(["model.safetensors"], False))

    assert len(loaded) == 1
    assert loaded[0][0] == "weight"
    assert loaded[0][1] is tensor
    assert observed == {
        "files": ["model.safetensors"],
        "framework": "pt",
        "device": 0,
        "process_group": None,
        "copy": expected_copy,
    }
    assert getattr(tensor, "_vllm_instanttensor_borrowed", False) is expected_borrowed


def test_instanttensor_copy_rejects_unknown_value(monkeypatch):
    def no_world_group():
        raise AssertionError

    monkeypatch.setenv("INSTANTTENSOR_COPY", "sometimes")
    monkeypatch.setattr(
        weight_utils,
        "current_platform",
        SimpleNamespace(is_cuda=lambda: True, current_device=lambda: 0),
    )
    monkeypatch.setattr(weight_utils, "get_world_group", no_world_group)
    monkeypatch.setitem(sys.modules, "instanttensor", SimpleNamespace())

    with pytest.raises(ValueError, match="INSTANTTENSOR_COPY must be 0 or 1"):
        next(instanttensor_weights_iterator(["model.safetensors"], False))


def test_instanttensor_restricts_io_to_indexed_shards(tmp_path):
    base_shard = tmp_path / "model-00001-of-00002.safetensors"
    overlay_shard = tmp_path / "model-00002-of-00002.safetensors"
    save_file(
        {
            "model.dense.weight": torch.tensor([2.0]),
            "model.expert.weight": torch.tensor([1.0, 1.0]),
        },
        base_shard,
    )
    save_file(
        {"model.expert.weight": torch.tensor([3.0, 3.0, 3.0])},
        overlay_shard,
    )

    buffer_sizes = []
    instant_open = SimpleNamespace(
        filename=[str(base_shard), str(overlay_shard)],
        ordered_tensor_metadatas=[
            ("model.dense.weight", {"data_offsets": [0, 4]}),
            ("model.expert.weight", {"data_offsets": [4, 12]}),
            ("model.expert.weight", {"data_offsets": [0, 12]}),
        ],
        tensor_offsets=[
            (0, 0),
            (0, 4),
            (0, 12),
            (1, 0),
            (1, 12),
        ],
        tensor_sizes=[4, 8, 12],
        total_tensor_size=24,
        tensor_name_to_index={},
        loader_handle=None,
        _determine_buffer_size=lambda requested: buffer_sizes.append(requested),
    )
    indexed_tensor_files = {
        "model.dense.weight": str(base_shard.resolve()),
        "model.expert.weight": str(overlay_shard.resolve()),
    }

    weight_utils._restrict_instanttensor_to_selected_ranges(
        instant_open,
        indexed_tensor_files=indexed_tensor_files,
        weight_name_prefixes=None,
    )

    assert instant_open.filename == [str(base_shard), str(overlay_shard)]
    assert [name for name, _ in instant_open.ordered_tensor_metadatas] == [
        "model.dense.weight",
        "model.expert.weight",
    ]
    assert instant_open.tensor_offsets == [
        (0, 0),
        (0, 4),
        (1, 0),
        (1, 12),
    ]
    assert instant_open.tensor_sizes == [4, 12]
    assert instant_open.total_tensor_size == 16
    assert instant_open.tensor_name_to_index == {
        "model.dense.weight": 0,
        "model.expert.weight": 1,
    }
    assert buffer_sizes == [None]


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="InstantTensor requires NVIDIA GPUs",
)
def test_instanttensor_model_loader():
    model_dir = download_weights_from_hf(
        "openai-community/gpt2", cache_dir=None, allow_patterns=["*.safetensors"]
    )
    safetensors = glob.glob(f"{model_dir}/*.safetensors")
    assert len(safetensors) > 0

    instanttensor_tensors = {}
    hf_safetensors_tensors = {}

    for name, tensor in instanttensor_weights_iterator(safetensors, True):
        instanttensor_tensors[name] = tensor.to("cpu")

    for name, tensor in safetensors_weights_iterator(safetensors, True):
        hf_safetensors_tensors[name] = tensor

    assert len(instanttensor_tensors) == len(hf_safetensors_tensors)

    for name, instanttensor_tensor in instanttensor_tensors.items():
        assert instanttensor_tensor.dtype == hf_safetensors_tensors[name].dtype
        assert instanttensor_tensor.shape == hf_safetensors_tensors[name].shape
        assert torch.all(instanttensor_tensor.eq(hf_safetensors_tensors[name]))


if __name__ == "__main__":
    test_instanttensor_model_loader()
