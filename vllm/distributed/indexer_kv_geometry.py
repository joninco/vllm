# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cache ownership rules shared by sparse-indexer admission and execution."""

from vllm.distributed.dcp_prefill import resolve_indexer_shards


def effective_kv_shards(spec: object, configured_dcp_size: int) -> int:
    """Resolve a cache group's shard count, including fully replicated caches."""
    requested = getattr(spec, "dcp_kv_shard_count", None)
    replicated = getattr(spec, "dcp_replicated", False)
    return resolve_indexer_shards(configured_dcp_size, requested or 0, replicated)


def indexer_layer_shards(
    *,
    dcp_size: int,
    pcp_size: int,
    requested_shards: int,
    replicate_cache: bool,
    layer_index: int | None,
    target_layers: int | None,
    b12x_enabled: bool,
) -> int:
    """Keep unidentified and speculative layers at the attention shard count."""
    configured = dcp_size * pcp_size
    if configured == 1 or layer_index is None or target_layers is None:
        return configured
    if layer_index >= target_layers:
        return configured
    shards = resolve_indexer_shards(dcp_size, requested_shards, replicate_cache)
    if shards == dcp_size:
        return configured
    if pcp_size != 1 or not 2 <= dcp_size <= 8:
        raise NotImplementedError("Indexer KV replication requires DCP2–8 and PCP1")
    if not b12x_enabled:
        raise ValueError("Indexer KV replication requires the b12x sparse indexer")
    return shards


def dcp_local_to_indexer_position(
    local_position: int,
    *,
    dcp_size: int,
    dcp_rank: int,
    indexer_shards: int,
    interleave: int,
) -> int:
    """Map one attention-local token position into replicated indexer storage."""
    resolve_indexer_shards(dcp_size, indexer_shards)
    if local_position < 0 or not 0 <= dcp_rank < dcp_size or interleave < 1:
        raise ValueError("Invalid position, DCP rank or interleave")
    return (
        local_position // interleave * (dcp_size // indexer_shards)
        + dcp_rank // indexer_shards
    ) * interleave + local_position % interleave
