# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.layers.quantization.kquant_hybrid import (
    KQuantHybridConfig,
    _b12x_tiles_for_geometry,
    _is_dense_layer_ignored,
    _read_hybrid_keys,
    _require_rank_local_kept_kernel,
    _stack_exl3_intermediate_rotations,
)
from vllm.model_executor.layers.quantization.kquant_qsrt_atoms_v2 import (
    COUPLED_H308_ATOM_SLAB_BYTES,
    COUPLED_H308_PROFILE,
    PROFILE,
    PURE_K2_PROFILE,
    _atom_slot_stride_for_profile,
    _coupled_pair_extent,
)
from vllm.model_executor.layers.quantization.nvfp4_nf3_hybrid import (
    NvFp4Nf3HybridConfig,
)


def _base_config(**updates):
    config = {
        "quant_method": "modelopt",
        "quant_algo": "NVFP4",
        "hybrid_bit_map": {"1": [4, 3]},
        "kept_format": "mxfp4_e8m0k32",
    }
    config.update(updates)
    return config


def _qsrt_descriptor(**updates):
    descriptor = {
        "schema": "kquant_kimi_k3_qsrt_atoms_v1",
        "storage_format": "qsrt_atoms_v1",
        "encoding": "qsrt_sqg_e4m3",
        "codebook": "sqg_xor_cheb_t12",
        "artifact_manifest": "qsrt-manifest.json",
    }
    descriptor.update(updates)
    return descriptor


@pytest.mark.parametrize(
    "raw",
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
def test_reads_hybrid_checkpoint_without_claiming_legacy_nf3(raw) -> None:
    bit_map, kept_format = _read_hybrid_keys(raw)
    assert bit_map == {"0": [4, 3]}
    assert kept_format == "mxfp4_e8m0k32"
    assert KQuantHybridConfig.override_quantization_method(raw, None) is None
    assert NvFp4Nf3HybridConfig.override_quantization_method(raw, None) == (
        "nvfp4_nf3_hybrid"
    )
    assert KQuantHybridConfig.override_quantization_method(raw, "fp8") is None


def test_detects_qsrt_checkpoint() -> None:
    raw = _base_config(
        demoted_format="qsrt_sqg_e4m3",
        qsrt=_qsrt_descriptor(),
    )
    assert KQuantHybridConfig.override_quantization_method(raw, None) == (
        "kquant_hybrid"
    )
    assert KQuantHybridConfig.override_quantization_method(raw, "fp8") is None


def test_config_registration_and_generic_exl3_default() -> None:
    assert get_quantization_config("kquant_hybrid") is KQuantHybridConfig
    config = KQuantHybridConfig.from_config(_base_config())
    assert config.hybrid_bit_map == {"1": [4, 3]}
    assert config.kept_format == "mxfp4_e8m0k32"
    assert config.demoted_format == "exl3_3"
    assert config.kept_storage == "inline-mxfp4"


def test_config_accepts_tp_independent_qsrt() -> None:
    descriptor = _qsrt_descriptor()
    config = KQuantHybridConfig.from_config(
        _base_config(demoted_format="qsrt_sqg_e4m3", qsrt=descriptor)
    )
    assert config.demoted_format == "qsrt_sqg_e4m3"
    assert config.kept_storage == "x4t"
    assert config.qsrt == descriptor
    assert config.trellis_codebook == "sqg_xor_cheb_t12"


def test_config_accepts_atoms_v2_revision() -> None:
    descriptor = _qsrt_descriptor(
        schema="qsrt_kimi_k3_qsrt_atoms_v2",
        storage_format="qsrt_atoms_v2",
    )
    config = KQuantHybridConfig.from_config(
        _base_config(demoted_format="qsrt_sqg_e4m3", qsrt=descriptor)
    )
    assert config.qsrt == {**descriptor, "profile": "k3x22_k4x2"}


def test_config_accepts_coupled_pure_k2_atoms_v2_profile() -> None:
    descriptor = _qsrt_descriptor(
        schema="qsrt_kimi_k3_qsrt_atoms_v2",
        storage_format="qsrt_atoms_v2",
        profile="k2_coupled_h512_h128",
    )
    config = KQuantHybridConfig.from_config(
        _base_config(
            hybrid_bit_map={"1": [2, 2]},
            demoted_format="qsrt_sqg_e4m3",
            qsrt=descriptor,
        )
    )
    assert config.hybrid_bit_map == {"1": [2, 2]}
    assert config.qsrt == descriptor


def test_atoms_v2_profiles_have_canonical_row_strides() -> None:
    assert _atom_slot_stride_for_profile(PROFILE) == 119005184
    assert _atom_slot_stride_for_profile(PURE_K2_PROFILE) == 77242368
    with pytest.raises(ValueError, match="unsupported QSRT atoms-v2 profile"):
        _atom_slot_stride_for_profile("unknown")


def test_config_accepts_coupled_high_rate_atoms_v2_profile() -> None:
    descriptor = _qsrt_descriptor(
        schema="qsrt_kimi_k3_qsrt_atoms_v2",
        storage_format="qsrt_atoms_v2",
        profile=COUPLED_H308_PROFILE,
    )
    config = KQuantHybridConfig.from_config(
        _base_config(
            hybrid_bit_map={"1": [3] * 896},
            demoted_format="qsrt_sqg_e4m3",
            qsrt=descriptor,
        )
    )
    assert config.qsrt == descriptor


def test_coupled_high_rate_atom_extents_cover_flat_slab() -> None:
    cursor = 0
    for pair in range(12):
        begin, extent, row_bytes = _coupled_pair_extent(pair)
        assert begin == cursor
        assert extent == 8 * row_bytes
        assert row_bytes % 4096 == 0
        cursor += extent
    assert cursor == COUPLED_H308_ATOM_SLAB_BYTES


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"demoted_format": "obsolete_private"}, "unsupported demoted_format"),
        (
            {"demoted_format": "qsrt_sqg_e4m3"},
            "requires a qsrt format descriptor",
        ),
        (
            {
                "demoted_format": "qsrt_sqg_e4m3",
                "qsrt": _qsrt_descriptor(storage_format="legacy-v1"),
            },
            "storage_format",
        ),
    ],
)
def test_config_rejects_obsolete_or_noncanonical_secondary_formats(
    updates, message
) -> None:
    with pytest.raises(ValueError, match=message):
        KQuantHybridConfig.from_config(_base_config(**updates))


def test_exl3_rotation_bundle_follows_b12x_projection_order() -> None:
    w13 = torch.arange(2 * 2 * 4, dtype=torch.float16).reshape(2, 2, 4)
    w2 = (100 + torch.arange(2 * 4, dtype=torch.float16)).reshape(2, 4)
    result = _stack_exl3_intermediate_rotations(w13, w2)
    expected = torch.cat((w13[:, 0], w13[:, 1], w2), dim=1)
    torch.testing.assert_close(result, expected)


def test_hybrid_kept_kernel_must_return_rank_local_partial() -> None:
    _require_rank_local_kept_kernel(SimpleNamespace(output_is_reduced=lambda: False))
    with pytest.raises(RuntimeError, match="unreduced rank-local partial"):
        _require_rank_local_kept_kernel(SimpleNamespace(output_is_reduced=lambda: True))


def test_b12x_tile_selection_is_geometry_driven() -> None:
    assert _b12x_tiles_for_geometry(3584, 3072) == (64, 256, 64, 256)
    assert _b12x_tiles_for_geometry(4096, 1536) == (64, 256, 64, 256)
    with pytest.raises(ValueError, match="no fixed b12x tile"):
        _b12x_tiles_for_geometry(3585, 3072)


@pytest.mark.parametrize(
    ("prefix", "ignored", "expected"),
    [
        ("model.layers.1.self_attn.q_proj", ["q_proj"], True),
        ("model.layers.1.self_attn.q_b_proj", ["b_proj"], False),
        ("model.layers.1.self_attn.q_proj", ["model.layers.1.self_attn.q_proj"], True),
    ],
)
def test_dense_short_exclusions_match_path_components(
    prefix, ignored, expected
) -> None:
    assert _is_dense_layer_ignored(prefix, ignored, {}) is expected
