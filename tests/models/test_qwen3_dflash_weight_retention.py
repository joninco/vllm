# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.models.qwen3_dflash import _retain_dflash_weight


def test_cpu_weight_retention_reuses_owned_storage() -> None:
    weight = torch.arange(8, dtype=torch.float32)

    retained = _retain_dflash_weight(weight)

    assert retained is weight


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_weight_retention_survives_source_storage_reuse() -> None:
    source = torch.arange(8, dtype=torch.float32, device="cuda")
    expected = source.cpu()

    retained = _retain_dflash_weight(source)
    source.fill_(-1)

    assert retained.device.type == "cpu"
    torch.testing.assert_close(retained, expected)
