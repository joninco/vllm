# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.config.quantization import resolve_quantization_config
from vllm.model_executor.layers.linear import (
    ReplicatedLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.layers.quantization.fp8 import (
    Mxfp8SerializedLinearMethod,
)
from vllm.model_executor.layers.quantization.nvfp4_nf3_hybrid import (
    NvFp4Nf3HybridConfig,
    _b12x_tiles_for_geometry,
    _combined_tier_local_descriptors,
    _compose_exl3_intermediate_rotations,
    _decode_kquant_nf3_scale,
    _exl3_parameter_specs,
    _is_dense_layer_ignored,
    _read_hybrid_keys,
    _shard_exl3_weight,
    _unpack_nf3_codes,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead


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


def test_config_parses_serialized_dense_mxfp8():
    config = NvFp4Nf3HybridConfig.from_config(
        {
            "quant_method": "modelopt",
            "quant_algo": "NVFP4",
            "hybrid_bit_map": {"0": [4, 3]},
            "dense_format": "mxfp8",
            "ignored_layers": ["g_proj", "vision_tower"],
        }
    )

    assert config.dense_format == "mxfp8"
    assert config.dense_ignored_layers == ["g_proj", "vision_tower"]


def test_config_parses_checkpoint_nf3_codebook():
    levels = [-1.0, -0.6, -0.35, -0.1, 0.1, 0.35, 0.6, 1.0]
    config = NvFp4Nf3HybridConfig.from_config(
        {
            "quant_method": "modelopt",
            "quant_algo": "NVFP4",
            "hybrid_bit_map": {"0": [4, 3]},
            "nf3_levels": levels,
        }
    )

    assert config.nf3_levels == levels


def test_config_rejects_invalid_nf3_codebook():
    with pytest.raises(ValueError, match="exactly 8"):
        NvFp4Nf3HybridConfig.from_config(
            {
                "quant_method": "modelopt",
                "quant_algo": "NVFP4",
                "hybrid_bit_map": {"0": [4, 3]},
                "nf3_levels": [0.0],
            }
        )


def test_config_parses_native_exl3_trellis_tier():
    config = NvFp4Nf3HybridConfig.from_config(
        {
            "quantization": {
                "quant_algo": "NVFP4",
                "group_size": 16,
                "kv_cache_quant_algo": None,
                "exclude_modules": [],
                "hybrid_bit_map": {"0": [4, 3]},
                "demoted_format": "exl3_3",
                "trellis": {
                    "mcg_mult": 0xCBAC1FED,
                    "shared_su": True,
                },
            },
        }
    )

    assert config.demoted_format == "exl3_3"
    assert config.trellis_mcg == 0xCBAC1FED - 2**32
    assert config.trellis_shared_su is True


def test_config_rejects_exl3_without_trellis_marker():
    with pytest.raises(ValueError, match="trellis.mcg_mult"):
        NvFp4Nf3HybridConfig.from_config(
            {
                "quant_method": "modelopt",
                "quant_algo": "NVFP4",
                "hybrid_bit_map": {"0": [4, 3]},
                "demoted_format": "exl3_3",
            }
        )


def test_serialized_dense_mxfp8_selects_loader_after_exact_exclusions():
    config = NvFp4Nf3HybridConfig(
        is_checkpoint_nvfp4_serialized=True,
        hybrid_bit_map={"0": [4, 3]},
    )
    config.dense_format = "mxfp8"
    config.dense_ignored_layers = ["b_proj"]
    linear = ReplicatedLinear.__new__(ReplicatedLinear)

    assert isinstance(
        config.get_quant_method(linear, "model.layers.0.self_attn.q_b_proj"),
        Mxfp8SerializedLinearMethod,
    )
    assert isinstance(
        config.get_quant_method(linear, "model.layers.0.self_attn.b_proj"),
        UnquantizedLinearMethod,
    )


def test_hybrid_config_keeps_nonserialized_dense_and_lm_head_unquantized():
    config = NvFp4Nf3HybridConfig(
        is_checkpoint_nvfp4_serialized=True,
        hybrid_bit_map={"0": [4, 3]},
    )
    linear = ReplicatedLinear.__new__(ReplicatedLinear)
    lm_head = ParallelLMHead.__new__(ParallelLMHead)

    assert isinstance(
        config.get_quant_method(linear, "model.layers.0.self_attn.q_proj"),
        UnquantizedLinearMethod,
    )
    assert config.get_quant_method(lm_head, "lm_head") is None


def test_config_rejects_missing_hybrid_bit_map():
    with pytest.raises(ValueError, match="hybrid_bit_map"):
        NvFp4Nf3HybridConfig.from_config(
            {
                "quant_method": "modelopt",
                "quant_algo": "NVFP4",
            }
        )


def test_config_accepts_dense_mxfp8_online_overlay():
    resolved = resolve_quantization_config(
        "nvfp4_nf3_hybrid",
        {
            "linear": {"weight": "mxfp8"},
            "ignore": ["re:.*kv_b_proj"],
        },
    )

    assert resolved is not None
    assert resolved.linear is not None
    assert resolved.linear.weight == kMxfp8Dynamic
    assert resolved.ignore == ["re:.*kv_b_proj"]


@pytest.mark.parametrize(
    ("prefix", "expected"),
    [
        ("model.layers.0.self_attn.g_proj", True),
        ("model.layers.0.self_attn.b_proj", True),
        ("model.layers.0.self_attn.q_b_proj", False),
        ("model.layers.3.self_attn.kv_b_proj", True),
        ("model.vision_tower.encoder.layers.0.mlp.fc1", True),
        ("model.layers.0.self_attn.q_proj", False),
    ],
)
def test_dense_mxfp8_short_exclusions_match_path_components(prefix, expected):
    ignored = ["g_proj", "b_proj", "kv_b_proj", "vision_tower"]

    assert _is_dense_layer_ignored(prefix, ignored, {}) is expected


def test_dense_mxfp8_full_prefix_exclusion_still_matches():
    prefix = "model.layers.0.self_attn.q_proj"

    assert _is_dense_layer_ignored(prefix, [prefix], {})


def test_dense_mxfp8_rejects_partially_excluded_fused_linear():
    with pytest.raises(ValueError, match="some but not all shards"):
        _is_dense_layer_ignored(
            "model.layers.0.mlp.gate_up_proj",
            ["gate_proj"],
            {"gate_up_proj": ["gate_proj", "up_proj"]},
        )


def test_unpack_nf3_codes():
    expected = torch.tensor([[[0, 1, 2, 3, 4, 5, 6, 7]]], dtype=torch.int32)
    word = sum(int(code) << (index * 3) for index, code in enumerate(expected[0, 0]))
    packed = torch.tensor(
        [[[word & 0xFF, (word >> 8) & 0xFF, (word >> 16) & 0xFF]]],
        dtype=torch.uint8,
    )

    torch.testing.assert_close(_unpack_nf3_codes(packed, size_k=8), expected)


def test_kimi_tp16_uses_tuned_fc1_tile():
    assert _b12x_tiles_for_geometry(3584, 3072 // 16) == (128, 64, 64, 128)


def test_kquant_nf3_scale_reinterprets_raw_fp8_bits():
    biased = torch.tensor([0.5, 2.0, 8.0], dtype=torch.float8_e4m3fn)
    decoded = _decode_kquant_nf3_scale(biased.view(torch.uint8))

    assert decoded.dtype == torch.float8_e4m3fn
    torch.testing.assert_close(decoded.float() * (2.0**-4), biased.float() / 16)


def test_exl3_parameter_geometry_broadcasts_h_side_rotations():
    specs = {
        name: (shape, dtype)
        for name, shape, dtype in _exl3_parameter_specs(3, 128, 64, True)
    }

    assert specs["w13_exl3_trellis"] == ((3, 2, 8, 4, 48), torch.int16)
    assert specs["w13_exl3_suh"] == ((1, 2, 128), torch.float16)
    assert specs["w13_exl3_svh"] == ((3, 2, 64), torch.float16)
    assert specs["w2_exl3_trellis"] == ((3, 4, 8, 48), torch.int16)
    assert specs["w2_exl3_suh"] == ((3, 64), torch.float16)
    assert specs["w2_exl3_svh"] == ((1, 128), torch.float16)


@pytest.mark.parametrize(
    ("family", "part", "shape", "shard_axis"),
    [
        ("w13", "trellis", (4, 6, 2), 1),
        ("w13", "svh", (6,), 0),
        ("w2", "trellis", (6, 4, 2), 0),
        ("w2", "suh", (6,), 0),
    ],
)
def test_exl3_tp_shards_only_intermediate_axes(family, part, shape, shard_axis):
    weight = torch.arange(torch.Size(shape).numel()).reshape(shape)

    sharded = _shard_exl3_weight(weight, family, part, tp_size=2, tp_rank=1)

    torch.testing.assert_close(sharded, weight.chunk(2, shard_axis)[1])


def test_exl3_tp_replicates_hidden_side_rotations():
    weight = torch.arange(8)

    assert _shard_exl3_weight(weight, "w13", "suh", 2, 1) is weight
    assert _shard_exl3_weight(weight, "w2", "svh", 2, 1) is weight


def test_exl3_intermediate_rotation_order_matches_fused_runtime():
    w13_svh = torch.stack(
        (
            torch.full((2, 4), 1.0),
            torch.full((2, 4), 2.0),
        ),
        dim=1,
    )
    w2_suh = torch.full((2, 4), 3.0)

    rotations = _compose_exl3_intermediate_rotations(w13_svh, w2_suh)

    expected = torch.tensor([[1.0] * 4 + [2.0] * 4 + [3.0] * 4] * 2)
    torch.testing.assert_close(rotations, expected)


def test_grid188_tier_descriptors_encode_exact_partition():
    remap = {
        **{global_id: (0, global_id) for global_id in range(64)},
        **{global_id: (1, global_id - 64) for global_id in range(64, 256)},
    }

    descriptors = _combined_tier_local_descriptors(remap)

    assert descriptors[:64] == list(range(64))
    assert descriptors[64:] == [0x100 | local_id for local_id in range(192)]


def test_grid188_tier_descriptors_reject_incomplete_partition():
    with pytest.raises(ValueError, match="does not cover all 256"):
        _combined_tier_local_descriptors({0: (0, 0)})
