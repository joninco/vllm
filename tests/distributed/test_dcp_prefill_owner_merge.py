# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eight-rank CUDA references for exact prefill candidate transport.

Run only with the serving process stopped, through the GPU coordinator:
  uv run --no-project .venv/bin/python -m torch.distributed.run \
    --standalone --nproc-per-node=8 -m pytest -q -s \
    tests/distributed/test_dcp_prefill_owner_merge.py

The tests use actual NCCL collectives, candidate packing, stable CuTe selection
and vLLM workspace/restore helpers. Independent CPU candidate unions determine
expected IDs, including signed-zero ordering. They do not validate indexer
scoring or cache ownership; those require separate model/cache references.
"""

import os
import struct
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from vllm.distributed import parallel_state
from vllm.distributed.dcp_prefill import build_indexer_replica_group_ranks
from vllm.v1.attention.backends.mla import b12x_indexer as indexer
from vllm.v1.worker import workspace


def _score_order(score: float) -> int:
    bits = struct.unpack("<I", struct.pack("<f", score))[0]
    return bits ^ (0xFFFFFFFF if bits & 0x80000000 else 0x80000000)


def _candidates(query: int, shard: int, shards: int, topk: int, iteration: int):
    ids, scores = [], []
    for col in range(topk):
        local_id = query * topk + col
        # Empty rows, empty shards and partially filled candidate arrays.
        absent = query % 7 == 0 or (query + shard + iteration) % 5 == 0 or col % 13 == 0
        ids.append(-1 if absent else local_id)
        selector = (col + query + iteration) % 7
        palette = (
            [0.0, -0.0, 2.0, 2.0, -3.0, 1.0, -float("inf")]
            if iteration == 0
            else [0.0, -0.0, 0.0, -0.0, -0.0, 0.0, -float("inf")]
        )
        scores.append(palette[selector])
    return ids, scores


def _expected(query: int, shards: int, topk: int, iteration: int, interleave: int):
    candidates = []
    for shard in range(shards):
        ids, scores = _candidates(query, shard, shards, topk, iteration)
        for local_id, score in zip(ids, scores):
            if local_id < 0:
                continue
            global_id = (
                (local_id // interleave) * shards + shard
            ) * interleave + local_id % interleave
            candidates.append((score, global_id))
    candidates.sort(key=lambda item: (-_score_order(item[0]), item[1]))
    selected = [token for _, token in candidates[:topk]]
    return sorted(selected + [-1] * (topk - len(selected)))


@pytest.fixture(scope="module")
def distributed_world():
    if int(os.environ.get("WORLD_SIZE", "0")) != 8:
        pytest.skip("requires a coordinator-authorized torchrun with eight workers")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    rank = dist.get_rank()
    device = torch.device("cuda", local_rank)
    workspace.init_workspace_manager(device, num_ubatches=1)
    group = SimpleNamespace(
        world_size=8,
        rank_in_group=rank,
        device_group=dist.group.WORLD,
        device_communicator=None,
    )
    yield rank, device, group
    workspace.reset_workspace_manager()
    dist.destroy_process_group()


@pytest.fixture(params=[1, 2, 4])
def topology(distributed_world, request):
    rank, device, tp = distributed_world
    shards = request.param
    shard_ranks, replica_ranks = build_indexer_replica_group_ranks(
        [list(range(8))], shards
    )
    handles = []
    selected = []
    for rank_lists in (shard_ranks, replica_ranks):
        local = None
        for ranks in rank_lists:
            handle = dist.new_group(
                ranks, backend="nccl", timeout=timedelta(seconds=180)
            )
            if rank in ranks:
                handles.append(handle)
                local = SimpleNamespace(
                    world_size=len(ranks),
                    rank_in_group=ranks.index(rank),
                    device_group=handle,
                    device_communicator=None,
                )
        selected.append(local)
    original = parallel_state._DCP
    parallel_state._DCP = selected[0]
    yield shards, rank, device, tp, selected[0], selected[1]
    parallel_state._DCP = original
    dist.barrier()
    for handle in reversed(handles):
        dist.destroy_process_group(handle)


@pytest.mark.parametrize("topk", [512, 1024, 2048])
@pytest.mark.parametrize("interleave", [1, 16])
@pytest.mark.parametrize("tail", [False, True])
def test_owner_and_replicated_merge_restore_exact_query_order(
    topology, topk, interleave, tail
):
    shards, rank, device, tp, shard_group, replicas = topology
    rows = shards + int(tail)
    total_rows = rows * replicas.world_size
    start = replicas.rank_in_group * rows
    manager = workspace.current_workspace_manager()
    manager.unlock()
    manager.reserve_all(*indexer._dcp_merge_shapes(rows, topk, shards))
    manager.reserve_all(*indexer._prefill_owner_shapes(rows, topk, shards))
    manager.reserve_all(((rows, topk), torch.int32))
    manager.lock()
    output = torch.empty((total_rows, topk), dtype=torch.int32, device=device)
    local = output[start : start + rows]
    local_pointer = local.data_ptr()
    for iteration in range(2):
        ids, values = zip(
            *[
                _candidates(q, rank % shards, shards, topk, iteration)
                for q in range(start, start + rows)
            ]
        )
        source_ids = torch.tensor(ids, dtype=torch.int32, device=device)
        source_scores = torch.tensor(values, dtype=torch.float32, device=device)
        expected = torch.tensor(
            [
                _expected(q, shards, topk, iteration, interleave)
                for q in range(total_rows)
            ],
            dtype=torch.int32,
        )

        def refill(source_ids=source_ids, source_scores=source_scores):
            local.copy_(source_ids)
            (scores,) = manager.get_simultaneous(((rows, topk), torch.float32))
            scores.copy_(source_scores)
            return scores

        scores = refill()
        indexer._merge_dcp_topk(local, scores, rank % shards, shards, interleave)
        indexer._restore_prefill_indices(replicas, local, output)
        reference = output.sort(dim=1).values.cpu()
        if shards == 1:
            # A one-shard indexer already emits its selected global IDs; this
            # fixture supplies unsorted candidates with absent slots instead.
            expected = torch.tensor(
                [
                    sorted(_candidates(q, 0, 1, topk, iteration)[0])
                    for q in range(total_rows)
                ],
                dtype=torch.int32,
            )
        assert torch.equal(reference, expected)
        scores = refill()
        used_owner = indexer._merge_prefill_topk_by_owner(
            local,
            scores,
            output,
            shard_group,
            tp,
            interleave,
        )
        assert used_owner is (shards > 1 and not tail)
        if not used_owner:
            indexer._merge_dcp_topk(local, scores, rank % shards, shards, interleave)
            indexer._restore_prefill_indices(replicas, local, output)
        assert torch.equal(output.sort(dim=1).values.cpu(), expected)
        assert local.data_ptr() == local_pointer
        assert manager.is_locked()
        # A nonaliasing view is rejected on every rank before communication.
        with pytest.raises(RuntimeError, match="alias"):
            indexer._restore_prefill_indices(replicas, local.clone(), output)
        dist.barrier()
