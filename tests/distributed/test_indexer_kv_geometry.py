# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check independent cache ownership, admission, and prefix geometry on CPU."""

from types import SimpleNamespace

import pytest
import torch

from vllm.distributed.indexer_kv_geometry import (
    dcp_local_to_indexer_position,
    effective_kv_shards,
    indexer_layer_shards,
)
from vllm.v1.kv_cache_interface import MLAAttentionSpec, UniformTypeKVCacheSpecs


@pytest.mark.parametrize("shards", [1, 2, 4])
@pytest.mark.parametrize(
    "layer,expected_target", [(0, True), (77, True), (78, False), (None, False)]
)
def test_target_replication_keeps_drafter_and_unknown_layers_sharded(
    shards, layer, expected_target
):
    assert indexer_layer_shards(
        dcp_size=4,
        pcp_size=1,
        requested_shards=shards,
        replicate_cache=False,
        layer_index=layer,
        target_layers=78,
        b12x_enabled=True,
    ) == (shards if expected_target else 4)


@pytest.mark.parametrize("shards", [1, 2, 4])
@pytest.mark.parametrize("interleave", [1, 4, 64])
def test_cache_mapping_preserves_every_global_token(shards, interleave):
    for rank in range(4):
        for local in range(259):
            global_position = (
                local // interleave * 4 + rank
            ) * interleave + local % interleave
            stored = dcp_local_to_indexer_position(
                local,
                dcp_size=4,
                dcp_rank=rank,
                indexer_shards=shards,
                interleave=interleave,
            )
            restored = (
                stored // interleave * shards + rank % shards
            ) * interleave + stored % interleave
            assert restored == global_position


@pytest.mark.parametrize("shards", [1, 2, 4])
def test_admission_capacity_counts_actual_indexer_shards(shards):
    spec = MLAAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=132,
        dtype=torch.uint8,
        dcp_kv_shard_count=shards,
    )
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=4),
        model_config=SimpleNamespace(max_model_len=4096),
    )
    expected_blocks = 4096 // (64 * shards)
    assert spec.max_num_blocks_per_req(config, 4096) == expected_blocks
    assert spec.max_memory_usage_bytes(config) == expected_blocks * 64 * 132
    group = UniformTypeKVCacheSpecs.from_specs({"a": spec, "b": spec})
    assert group is not None
    assert effective_kv_shards(group, 4) == shards
    assert group.max_num_blocks_per_req(config, 4096) == expected_blocks
    assert MLAAttentionSpec.merge([spec, spec]).dcp_kv_shard_count == shards


def test_distinct_shard_geometries_cannot_share_a_cache_group():
    def spec(shards):
        return MLAAttentionSpec(
            block_size=64,
            num_kv_heads=1,
            head_size=132,
            dtype=torch.uint8,
            dcp_kv_shard_count=shards,
        )

    a, b = spec(2), spec(4)
    assert UniformTypeKVCacheSpecs.from_specs({"a": a, "b": b}) is None
    with pytest.raises(AssertionError):
        MLAAttentionSpec.merge([a, b])


def test_dcp1_ignores_dcp4_replication_selection():
    assert (
        indexer_layer_shards(
            dcp_size=1,
            pcp_size=1,
            requested_shards=2,
            replicate_cache=False,
            layer_index=0,
            target_layers=78,
            b12x_enabled=True,
        )
        == 1
    )


def test_replication_rejects_an_indexer_without_native_b12x_dispatch():
    with pytest.raises(ValueError, match="b12x"):
        indexer_layer_shards(
            dcp_size=4,
            pcp_size=1,
            requested_shards=2,
            replicate_cache=False,
            layer_index=0,
            target_layers=78,
            b12x_enabled=False,
        )
