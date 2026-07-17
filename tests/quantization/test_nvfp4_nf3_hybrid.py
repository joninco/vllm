# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.layers.quantization.nvfp4_nf3_hybrid import (
    NvFp4Nf3HybridConfig,
    _read_hybrid_keys,
    _unpack_nf3_codes,
)


@pytest.mark.parametrize(
    "config",
    [
        {
            "hybrid_bit_map": {"0": [4, 3]},
            "kept_format": "mxfp4_e8m0k32",
        },
        {
            "quantization": {
                "hybrid_bit_map": {"0": [4, 3]},
                "kept_format": "mxfp4_e8m0k32",
            }
        },
    ],
)
def test_reads_and_detects_hybrid_checkpoint(config):
    bit_map, kept_format = _read_hybrid_keys(config)

    assert bit_map == {"0": [4, 3]}
    assert kept_format == "mxfp4_e8m0k32"
    assert (
        NvFp4Nf3HybridConfig.override_quantization_method(config, None)
        == "nvfp4_nf3_hybrid"
    )
    assert NvFp4Nf3HybridConfig.override_quantization_method(config, "fp8") is None


def test_config_registration_and_parsing():
    assert get_quantization_config("nvfp4_nf3_hybrid") is NvFp4Nf3HybridConfig

    config = NvFp4Nf3HybridConfig.from_config(
        {
            "quant_method": "modelopt",
            "quant_algo": "NVFP4",
            "hybrid_bit_map": {"0": [4, 3]},
            "kept_format": "mxfp4_e8m0k32",
        }
    )

    assert config.hybrid_bit_map == {"0": [4, 3]}
    assert config.kept_format == "mxfp4_e8m0k32"


def test_config_rejects_missing_hybrid_bit_map():
    with pytest.raises(ValueError, match="hybrid_bit_map"):
        NvFp4Nf3HybridConfig.from_config(
            {
                "quant_method": "modelopt",
                "quant_algo": "NVFP4",
            }
        )


def test_unpack_nf3_codes():
    expected = torch.tensor([[[0, 1, 2, 3, 4, 5, 6, 7]]], dtype=torch.int32)
    word = sum(int(code) << (index * 3) for index, code in enumerate(expected[0, 0]))
    packed = torch.tensor(
        [[[word & 0xFF, (word >> 8) & 0xFF, (word >> 16) & 0xFF]]],
        dtype=torch.uint8,
    )

    torch.testing.assert_close(_unpack_nf3_codes(packed, size_k=8), expected)
