# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure worker and DCP communication storage without loading model weights.

Run with torchrun on an idle host, supplying normal EngineArgs plus
--output-dir and --runtime-manifest. The JSON files and worker log distinguish
distributed initialization, runner metadata, direct query/combine storage,
and the indexer merge workspace. These measurements do not size a serving KV
cache: model weights, activation peaks and CUDA graphs are absent.
"""

import gc
import hashlib
import json
import os
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import torch
import torch.distributed as dist

from vllm.config import set_current_vllm_config
from vllm.distributed import get_dcp_group
from vllm.engine.arg_utils import EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.mem_utils import MemorySnapshot
from vllm.v1.attention.backends.mla.b12x_indexer import _dcp_merge_shapes
from vllm.v1.attention.ops.dcp import MLADCPManager
from vllm.v1.worker.gpu_worker import Worker
from vllm.v1.worker.workspace import current_workspace_manager


def snapshot(device):
    torch.accelerator.synchronize(device)
    gc.collect()
    torch.accelerator.empty_cache()
    return MemorySnapshot(device=device)


def main():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--local-query-heads", type=int, default=8)
    parser.add_argument("--query-head-dim", type=int, default=576)
    parser.add_argument("--output-head-dim", type=int, default=512)
    parser.add_argument("--indexer-topk", type=int, default=2048)
    args = parser.parse_args()
    manifest = {
        "path": str(args.runtime_manifest),
        "sha256": hashlib.sha256(args.runtime_manifest.read_bytes()).hexdigest(),
    }
    config = EngineArgs.from_cli_args(args).create_engine_config()
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != config.parallel_config.world_size:
        raise ValueError("torchrun WORLD_SIZE must match the engine parallel size")
    if config.parallel_config.decode_context_parallel_size <= 1:
        raise ValueError("DCP startup memory profiling requires at least two ranks")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with set_current_vllm_config(config):
        worker = Worker(config, local_rank, rank, "env://")
        worker.init_device()
        after_worker = snapshot(worker.device)
        manager = MLADCPManager(
            config,
            worker.device,
            args.local_query_heads,
            args.query_head_dim,
            args.output_head_dim,
            torch.bfloat16,
            torch.bfloat16,
            None,
            True,
            False,
        )
        # Exercise each initialized collective so lazy communicator/module
        # storage is included in the communication footprint.
        rows = min(16, manager.max_num_tokens)
        query = torch.ones(
            (rows, args.local_query_heads, args.query_head_dim),
            dtype=torch.bfloat16,
            device=worker.device,
        )
        partial = torch.ones(
            (
                rows,
                args.local_query_heads * get_dcp_group().world_size,
                args.output_head_dim,
            ),
            dtype=torch.bfloat16,
            device=worker.device,
        )
        lse = torch.zeros(partial.shape[:2], device=worker.device)
        for _ in range(2):
            gathered = manager.query_gather(query)
            result = manager.combine(partial, lse)
            torch.testing.assert_close(gathered, torch.ones_like(gathered))
            torch.testing.assert_close(result, torch.ones_like(result))
        del query, partial, lse, gathered, result
        after_direct = snapshot(worker.device)
        specs = _dcp_merge_shapes(
            config.scheduler_config.max_num_batched_tokens,
            args.indexer_topk,
            config.parallel_config.decode_context_parallel_size,
        )
        workspace = current_workspace_manager()
        workspace.reserve_all(*specs)
        workspace.lock()
        after_workspace = snapshot(worker.device)
        stages = {
            "before_distributed_init": worker.init_snapshot,
            "after_worker_init": after_worker,
            "after_direct_communication": after_direct,
            "after_indexer_workspace": after_workspace,
        }
        record = {
            "semantic_role": "Worker and DCP startup storage without model weights",
            "status": "research-only",
            "created_at": datetime.now(UTC).isoformat(),
            "runtime_manifest": manifest,
            "rank": rank,
            "dcp_rank": get_dcp_group().rank_in_group,
            "requested_memory_bytes": worker.requested_memory,
            "distributed_init_bytes": worker.distributed_init_memory,
            "direct_communication_retained_bytes": after_worker.free_memory
            - after_direct.free_memory,
            "indexer_workspace_growth_bytes": after_direct.free_memory
            - after_workspace.free_memory,
            "indexer_merge_specs": [(shape, str(dtype)) for shape, dtype in specs],
            "snapshots": {name: asdict(value) for name, value in stages.items()},
        }
        path = args.output_dir / f"rank-{rank}.json"
        path.write_text(json.dumps(record, indent=2, default=str) + "\n")
        print(json.dumps({"rank": rank, "output": str(path)}), flush=True)
        dist.barrier()
        # Process exit releases IPC slabs after every peer has finished.


if __name__ == "__main__":
    main()
