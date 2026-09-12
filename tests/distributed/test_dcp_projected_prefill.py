# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eight-rank references for projected DCP prefill with actual NCCL peers.

Run with the server stopped through the GPU coordinator:
  uv run --no-project .venv/bin/python -m torch.distributed.run \
    --standalone --nproc-per-node=8 -m pytest -q -s \
    tests/distributed/test_dcp_projected_prefill.py

The workspace helpers use real query, weight and LSE all-gathers and output
reduce-scatter in two DCP4 groups. A CPU float64 oracle computes each rank's
head partition independently. Small rows use dense weights; serving-sized
rows use two nonzero weights per output column to bound CPU reference cost.
The helpers are called directly; model-route eligibility is tested separately.
"""

import math
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from tests.distributed.test_dcp_prefill_owner_merge import (
    distributed_world,  # noqa: F401
)
from vllm.envs import disable_envs_cache
from vllm.v1.attention.ops import dcp
from vllm.v1.worker import workspace


def _override_envs(monkeypatch, name, value):
    """Override one lazily resolved ``vllm.envs`` value for the current test.

    The value goes through the environment so ``vllm.envs`` parses it the way
    a launch would; patching the module attribute instead would leave the
    resolved value behind as a permanent attribute after the test, hiding
    later environment changes in the same process.
    """
    disable_envs_cache()
    if isinstance(value, bool):
        value = "1" if value else "0"
    monkeypatch.setenv(name, str(value))


# Module fixtures own the process groups across parametrized references.
pytestmark = pytest.mark.skip_global_cleanup

_CAPACITY = 8192
_LOCAL_HEADS = 8
_HEADS = 32
_QUERY = 576
_LATENT = 512
_VALUE = 256


@pytest.fixture(scope="module")
def dcp_group(request):
    rank, device, _ = request.getfixturevalue("distributed_world")
    selected = None
    for ranks in (list(range(4)), list(range(4, 8))):
        handle = dist.new_group(ranks, backend="nccl", timeout=timedelta(seconds=180))
        if rank in ranks:
            selected = SimpleNamespace(
                world_size=4,
                rank_in_group=ranks.index(rank),
                ranks=ranks,
                device_group=handle,
            )
    assert selected is not None
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield rank, device, selected
    torch.set_num_threads(threads)
    dist.barrier()
    dist.destroy_process_group(selected.device_group)


def _query(rows, rank, iteration):
    row = torch.arange(rows)[:, None, None]
    head = torch.arange(_LOCAL_HEADS)[None, :, None] + rank * _LOCAL_HEADS
    dim = torch.arange(_QUERY)[None, None, :]
    return ((row + head * 7 + dim * 3 + iteration * 11) % 127).bfloat16() / 128


def _partials(rows, source, heads, iteration):
    row = torch.arange(rows)[:, None, None]
    head = torch.tensor(heads)[None, :, None]
    dim = torch.arange(_LATENT)[None, None, :]
    return (
        ((row * 3 + head * 7 + dim * 11 + source * 13 + iteration * 17) % 31 - 15)
        .to(torch.bfloat16)
        .div_(128)
    )


def _lse_and_lengths(rows, source, heads, iteration, base_e):
    row = torch.arange(rows)[:, None]
    head = torch.tensor(heads)[None, :]
    lse = (source % 4 - 1.5) / 2 + ((row + head + iteration) % 7).float() / 16
    if not base_e:
        lse /= math.log(2)
    lengths = ((torch.arange(rows) % 5) != (source + iteration) % 4).int()
    lengths[0] = 0
    return lse, lengths


def _weights(rank, iteration, dense):
    if dense:
        generator = torch.Generator().manual_seed(913 + 37 * rank + iteration)
        return (
            torch.randn(_LOCAL_HEADS, _LATENT, _VALUE, generator=generator)
            / math.sqrt(_LATENT)
        ).bfloat16()
    weights = torch.zeros(_LOCAL_HEADS, _LATENT, _VALUE, dtype=torch.bfloat16)
    col = torch.arange(_VALUE)
    for head in range(_LOCAL_HEADS):
        weights[head, 2 * col, col] = (rank + 1) / 8 + head / 64 + iteration / 32
        weights[head, 2 * col + 1, col] = -(head + 1) / 32 - iteration / 64
    return weights


def _expected(rows, rank, ranks, iteration, base_e, weights, dense):
    heads = list(range(rank * _LOCAL_HEADS, (rank + 1) * _LOCAL_HEADS))
    shard_lses = []
    valid = []
    for source in ranks:
        lse, lengths = _lse_and_lengths(rows, source, heads, iteration, base_e)
        shard_lses.append(lse.double().masked_fill(lengths[:, None] == 0, -math.inf))
        valid.append(lengths != 0)
    lses = torch.stack(shard_lses)
    if not base_e:
        lses *= math.log(2)
    factors = torch.softmax(lses, dim=0).nan_to_num(0)
    result = torch.zeros(rows, _LOCAL_HEADS, _VALUE, dtype=torch.float64)
    columns = torch.arange(_VALUE)
    for index, source in enumerate(ranks):
        partial = _partials(rows, source, heads, iteration).double()
        if dense:
            projected = torch.einsum("rhl,hlv->rhv", partial, weights.double())
        else:
            projected = (
                partial[..., 0::2] * weights[:, 2 * columns, columns].double()
                + partial[..., 1::2] * weights[:, 2 * columns + 1, columns].double()
            )
        projected.masked_fill_(~valid[index][:, None, None], 0)
        result.add_(projected * factors[index, :, :, None])
    return result


@pytest.mark.parametrize("borrowed", [False, True])
@pytest.mark.parametrize("rows", [33, 1025, _CAPACITY])
@pytest.mark.parametrize("base_e", [False, True])
def test_projected_workspace_matches_independent_peer_oracle(
    dcp_group, monkeypatch, borrowed, rows, base_e
):
    from b12x.comm.prefill import prepare_prefill_lse

    rank, device, group = dcp_group
    manager = workspace.current_workspace_manager()
    manager.unlock()
    specs = (
        ((_CAPACITY, _HEADS, _QUERY), torch.bfloat16),
        ((_CAPACITY * _HEADS * (_LATENT * 2 + 4),), torch.uint8),
    )
    transport = dcp.MLADCPManager.__new__(dcp.MLADCPManager)
    transport.group = group
    transport.use_a2a = True
    transport.is_lse_base_on_e = base_e
    for name, value in {
        "VLLM_DCP_PROJECT_BEFORE_MERGE": True,
        "VLLM_B12X_MLA_DCP_GATHER_IN_WORKSPACE": borrowed,
        "VLLM_DCP_PROJECT_BEFORE_MERGE_MIN_PREFILL_TOKENS": 1024,
        "VLLM_DCP_A2A_MAX_TOKENS": 16,
        "VLLM_DCP_A2A_LARGE_BACKEND": "ag_rs",
    }.items():
        _override_envs(monkeypatch, name, value)
    transport.configure_prefill(
        SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_batched_tokens=_CAPACITY),
            compilation_config=SimpleNamespace(cudagraph_capture_sizes=[16]),
        ),
        backend=SimpleNamespace(get_dcp_prefill_workspace_specs=lambda: specs),
        latent_dim=_LATENT,
        value_dim=_VALUE,
    )
    manager.lock()
    plan = transport.prefill_workspaces[borrowed]
    dense = rows == 33
    local_weights = _weights(rank, 0, dense).to(device)
    transport.prewarm_prefill(local_weights, backend_specs=specs)
    assert transport.prefill_warmup_record["completed"]
    pointers = None
    for iteration in range(2):
        buffers = plan.borrow(rows, backend_specs=specs)
        addresses = tuple(
            getattr(buffers, name).data_ptr()
            for name in (
                "query",
                "scratch",
                "weights",
                "local_lse",
                "all_lse",
                "projection_input",
                "projected",
                "result",
            )
        )
        if pointers is None:
            pointers = addresses
        assert addresses == pointers
        query = _query(rows, rank, iteration).to(device)
        gathered = buffers.gather_query((query[..., :512], query[..., 512:]), group)
        torch.testing.assert_close(
            gathered.cpu(),
            torch.cat(
                [_query(rows, source, iteration) for source in group.ranks], dim=1
            ),
            atol=0,
            rtol=0,
        )
        # Attention reborrows the same prefix after consuming the query.
        backend_query, scratch = manager.get_simultaneous(*specs)
        assert backend_query.data_ptr() == buffers.backend_query.data_ptr()
        assert scratch.data_ptr() == buffers.scratch.data_ptr()
        output_bytes = _CAPACITY * _HEADS * _LATENT * 2
        output = (
            scratch[:output_bytes]
            .view(torch.bfloat16)
            .as_strided((rows, _HEADS, _LATENT), (_LATENT, _CAPACITY * _LATENT, 1))
        )
        lse = (
            scratch[output_bytes:]
            .view(torch.float32)[: rows * _HEADS]
            .view(rows, _HEADS)
        )
        all_heads = list(range(group.ranks[0] * 8, (group.ranks[-1] + 1) * 8))
        output.copy_(_partials(rows, rank, all_heads, iteration))
        source_lse, lengths = _lse_and_lengths(rows, rank, all_heads, iteration, base_e)
        lse.copy_(source_lse)
        empty = (lengths == 0).to(device)
        output[empty] = float("nan")
        lse[empty] = float("nan")
        weights_cpu = _weights(rank, iteration, dense)
        local_weights.copy_(weights_cpu)
        projected = buffers.project(
            output, lse, lengths.to(device), local_weights, group, prepare_prefill_lse
        )
        result = buffers.combine(
            projected, group, dcp.correct_attn_out, is_lse_base_on_e=base_e
        )
        expected = _expected(
            rows, rank, group.ranks, iteration, base_e, weights_cpu, dense
        )
        torch.testing.assert_close(
            result.double().cpu(), expected, atol=0.008, rtol=0.012
        )
        assert result.data_ptr() == buffers.result.data_ptr()
        assert result.stride() == (_VALUE, rows * _VALUE, 1)
        assert torch.isfinite(result).all()
        assert manager.is_locked()
        dist.barrier()
