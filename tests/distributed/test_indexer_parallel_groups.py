# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU topology and lifecycle contracts for opt-in DCP prefill collectives.

Inputs are TP rank lists, cache shard counts and feature flags. Outputs are
ordered collective groups or configuration errors. Mock group creation catches
rank-order, target/drafter selection and leaked-group failures without CUDA.
"""

from types import SimpleNamespace

import pytest

from vllm.distributed import parallel_state as state
from vllm.distributed.dcp_prefill import (
    build_indexer_replica_group_ranks,
    resolve_indexer_shards,
)


@pytest.mark.parametrize("shards", [1, 2, 4])
def test_tp8_partitions_every_rank_and_pairs_equal_shard_owners(shards):
    owners, replicas = build_indexer_replica_group_ranks([list(range(8))], shards)
    assert owners == [list(range(a, a + shards)) for a in range(0, 8, shards)]
    assert replicas == [list(range(s, 8, shards)) for s in range(shards)]


def test_replica_groups_preserve_separate_tp_domains():
    owners, replicas = build_indexer_replica_group_ranks(
        [list(range(8)), list(range(8, 16))], 4
    )
    assert owners[-1] == [12, 13, 14, 15]
    assert replicas == [[s, s + 4] for s in range(4)] + [
        [s, s + 4] for s in range(8, 12)
    ]


@pytest.mark.parametrize(
    "requested,replicate,expected",
    [(0, False, 4), (0, True, 1), (1, True, 1), (2, False, 2), (4, False, 4)],
)
def test_shard_configuration_alias(requested, replicate, expected):
    assert resolve_indexer_shards(4, requested, replicate) == expected


@pytest.mark.parametrize(
    "requested,replicate", [(-1, False), (3, False), (8, False), (2, True)]
)
def test_invalid_shards_fail_before_group_creation(requested, replicate):
    with pytest.raises(ValueError):
        resolve_indexer_shards(4, requested, replicate)


@pytest.fixture
def groups(monkeypatch):
    created, destroyed = [], []
    for name in (
        "_QUERY_SPLIT",
        "_INDEXER_DCP",
        "_INDEXER_QUERY_SPLIT",
        "_DCP_CKV_PREFETCH",
    ):
        monkeypatch.setattr(state, name, None)
    monkeypatch.setattr(state, "_DCP", SimpleNamespace(world_size=4))

    def create(ranks, local_rank, backend, *, group_name):
        group = SimpleNamespace(
            world_size=len(ranks[0]),
            destroy=lambda: destroyed.append(group_name),
        )
        created.append((group_name, ranks, group))
        return group

    monkeypatch.setattr(state, "init_model_parallel_group", create)
    yield created, destroyed
    state._destroy_dcp_prefill_groups()


def initialize(**overrides):
    args = dict(
        tp_ranks=[list(range(8))],
        dcp_ranks=[list(range(4)), list(range(4, 8))],
        dcp_size=4,
        pcp_size=1,
        elastic_ep=False,
        local_rank=0,
        backend="nccl",
        query_split=True,
        indexer_shards=2,
        replicate_indexer=False,
        ckv_gather=True,
        ckv_prefetch_depth=1,
    )
    args.update(overrides)
    state._initialize_dcp_prefill_groups(**args)


def test_target_and_drafter_choose_their_own_groups(groups):
    created, destroyed = groups
    initialize()
    assert [entry[0] for entry in created] == [
        "query_split",
        "indexer_dcp",
        "indexer_query_split",
        "dcp_ckv_prefetch",
    ]
    assert state.get_indexer_dcp_group(2) is created[1][2]
    assert state.get_indexer_dcp_group(4) is state._DCP
    assert state.get_indexer_query_split_group(2) is created[2][2]
    assert state.get_indexer_query_split_group(4) is created[0][2]
    assert state.get_dcp_ckv_prefetch_group() is created[3][2]
    with pytest.raises(RuntimeError):
        state.get_indexer_dcp_group(3)
    state._destroy_dcp_prefill_groups()
    state._destroy_dcp_prefill_groups()
    assert destroyed == [entry[0] for entry in reversed(created)]


@pytest.mark.parametrize(
    "dcp_size,query_split,ckv", [(1, True, True), (4, False, False)]
)
def test_dcp1_and_disabled_features_create_no_groups(
    groups, dcp_size, query_split, ckv
):
    initialize(
        dcp_size=dcp_size, indexer_shards=0, query_split=query_split, ckv_gather=ckv
    )
    assert not groups[0]


def test_single_shard_replication_has_tp_wide_query_group(groups):
    initialize(indexer_shards=1, ckv_gather=False)
    assert state.get_indexer_dcp_group(1).world_size == 1
    assert state.get_indexer_query_split_group(1).world_size == 8


@pytest.mark.parametrize("shards", [1, 2, 4])
@pytest.mark.parametrize("replicate", [False, True])
@pytest.mark.parametrize("depth", [-1, 0, 1])
def test_dcp1_initialization_ignores_dcp4_mechanism_options(
    groups, shards, replicate, depth
):
    assert resolve_indexer_shards(1, shards, replicate) == 1
    initialize(
        dcp_size=1,
        indexer_shards=shards,
        replicate_indexer=replicate,
        query_split=True,
        ckv_gather=True,
        ckv_prefetch_depth=depth,
    )
    assert not groups[0]
    assert state._QUERY_SPLIT is None
    assert state._INDEXER_DCP is None
    assert state._INDEXER_QUERY_SPLIT is None
    assert state._DCP_CKV_PREFETCH is None


@pytest.mark.parametrize(
    "overrides,expected",
    [
        (dict(query_split=False, indexer_shards=0), ["dcp_ckv_prefetch"]),
        (dict(ckv_gather=False, indexer_shards=0), ["query_split"]),
        (dict(ckv_gather=False, query_split=False), ["indexer_dcp"]),
        (dict(ckv_prefetch_depth=0, query_split=False, indexer_shards=0), []),
        (
            dict(
                ckv_gather=False,
                ckv_prefetch_depth=-1,
                query_split=False,
                indexer_shards=0,
            ),
            [],
        ),
    ],
)
def test_only_requested_collectives_are_created(groups, overrides, expected):
    initialize(**overrides)
    assert [entry[0] for entry in groups[0]] == expected


def test_enabled_prefetch_rejects_negative_depth_without_collectives(groups):
    with pytest.raises(ValueError, match="PREFETCH_DEPTH"):
        initialize(ckv_prefetch_depth=-1)
    assert not groups[0]


@pytest.mark.parametrize(
    "overrides",
    [
        dict(pcp_size=2),
        dict(elastic_ep=True),
        dict(dcp_ranks=[[0, 2, 4, 6], [1, 3, 5, 7]]),
    ],
)
def test_unsupported_topology_fails_without_collectives(groups, overrides):
    with pytest.raises(ValueError):
        initialize(**overrides)
    assert not groups[0]


def test_failed_creation_releases_previously_created_groups(groups, monkeypatch):
    created, destroyed = groups
    original = state.init_model_parallel_group

    def fail_second(*args, **kwargs):
        if created:
            raise RuntimeError("collective creation failed")
        return original(*args, **kwargs)

    monkeypatch.setattr(state, "init_model_parallel_group", fail_second)
    with pytest.raises(RuntimeError, match="creation failed"):
        initialize()
    assert destroyed == ["query_split"]
    assert state._QUERY_SPLIT is None
