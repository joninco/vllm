# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from torch import nn

from vllm.model_executor.model_loader.default_loader import DefaultModelLoader


class _SerializedMethod:
    required_checkpoint_parameter_names = ("weight", "weight_scale")


class _SerializedLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(2, 2))
        self.weight_scale = nn.Parameter(
            torch.empty(2, 1, dtype=torch.uint8), requires_grad=False
        )
        self.quant_method = _SerializedMethod()


def test_required_checkpoint_parameters_accept_complete_layer() -> None:
    model = nn.Module()
    model.linear = _SerializedLayer()

    DefaultModelLoader.validate_required_checkpoint_parameters(
        model, {"linear.weight", "linear.weight_scale"}
    )


def test_required_checkpoint_parameters_reject_missing_scale() -> None:
    model = nn.Module()
    model.linear = _SerializedLayer()

    with pytest.raises(ValueError, match=r"linear\.weight_scale"):
        DefaultModelLoader.validate_required_checkpoint_parameters(
            model, {"linear.weight"}
        )
