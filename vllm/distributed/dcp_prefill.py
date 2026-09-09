# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure topology rules for sparse-indexer replicas under context parallelism."""


def resolve_indexer_shards(
    dcp_size: int, requested_shards: int, replicate_cache: bool = False
) -> int:
    """Resolve a launch's indexer shard count without creating process groups."""
    if dcp_size < 1:
        raise ValueError("DCP size must be positive")
    if dcp_size == 1:
        # DCP1 comparison launches retain the DCP4 mechanism environment.
        return 1
    if replicate_cache and requested_shards not in (0, 1):
        raise ValueError(
            "VLLM_DCP_REPLICATE_INDEXER_CACHE conflicts with "
            f"VLLM_DCP_INDEXER_SHARDS={requested_shards}"
        )
    shards = 1 if replicate_cache else requested_shards or dcp_size
    if shards < 1 or shards > dcp_size or dcp_size % shards:
        raise ValueError(
            "VLLM_DCP_INDEXER_SHARDS must be 0 or a positive divisor of DCP; "
            f"got shards={requested_shards}, DCP={dcp_size}"
        )
    return shards


def build_indexer_replica_group_ranks(
    tp_group_ranks: list[list[int]], indexer_shards: int
) -> tuple[list[list[int]], list[list[int]]]:
    """Return contiguous shard groups and corresponding cross-replica groups."""
    shard_groups: list[list[int]] = []
    query_groups: list[list[int]] = []
    for ranks in tp_group_ranks:
        if not ranks or indexer_shards < 1 or len(ranks) % indexer_shards:
            raise ValueError(
                f"Indexer shards={indexer_shards} must divide TP={len(ranks)}"
            )
        replicas = [
            ranks[start : start + indexer_shards]
            for start in range(0, len(ranks), indexer_shards)
        ]
        shard_groups.extend(replicas)
        query_groups.extend(
            [replica[shard] for replica in replicas] for shard in range(indexer_shards)
        )
    return shard_groups, query_groups
