# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.model_loader.utils import process_weights_after_loading


class _RecordingQuantMethod(QuantizeMethodBase):
    def __init__(self, name: str, order: list[str], priority: int):
        self.name = name
        self.order = order
        self.process_weights_after_loading_priority = priority

    def create_weights(self, layer, *args, **kwargs):
        raise NotImplementedError

    def apply(self, layer, *args, **kwargs):
        raise NotImplementedError

    def process_weights_after_loading(self, layer):
        self.order.append(self.name)


def test_process_weights_after_loading_honors_method_priority():
    order: list[str] = []
    model = torch.nn.Module()
    model.high_priority = torch.nn.Module()
    model.high_priority.quant_method = _RecordingQuantMethod("high", order, 100)
    model.default_priority = torch.nn.Module()
    model.default_priority.quant_method = _RecordingQuantMethod("default", order, 0)
    model.early_priority = torch.nn.Module()
    model.early_priority.quant_method = _RecordingQuantMethod("early", order, -1)
    model_config = SimpleNamespace(dtype=torch.float32, quantization=None)

    process_weights_after_loading(model, model_config, torch.device("cpu"))

    assert order == ["early", "default", "high"]
