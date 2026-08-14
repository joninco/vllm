# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""All-row Kimi-K3 routed-MoE calibration capture.

Rank zero records every prompt row at every routed-MoE layer.  Four writer
processes persist independent, checksummed safetensors chunks while the model
worker applies bounded backpressure.  Other tensor-parallel ranks write only a
completion receipt, so the large tensors are never duplicated by TP rank.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import regex as re
import torch
import zmq
from safetensors.torch import save_file

from vllm.logger import init_logger

logger = init_logger(__name__)

_KIND = "qsrt_all_routed_rows"
_SCHEMA_VERSION = 1
_NUM_MOE_LAYERS = 92
_FIRST_MOE_LAYER = 1
_INPUT_SIZE = 3584
_TOP_K = 16
_REQUEST_RE = re.compile(r"(?:^|-)qsrtcap-(\d+)-([0-9a-f]{32})(?:-|$)")
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_TENSOR_KEYS = (
    "input",
    "expert_indices",
    "route_weights",
    "routed_output",
    "request_index",
    "document_id",
    "token_offset",
    "role",
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return value


def _write_chunk(
    root: Path,
    *,
    layer: int,
    index: int,
    row_begin: int,
    tensors: dict[str, torch.Tensor],
) -> dict[str, Any]:
    rows = int(tensors["input"].shape[0])
    path_root = root / f"layer-{layer:05d}"
    path_root.mkdir(parents=True, exist_ok=True)
    path = path_root / f"chunk-{index:08d}.safetensors"
    receipt_path = path_root / f"chunk-{index:08d}.json"
    if path.exists() or receipt_path.exists():
        raise FileExistsError(
            f"all-row chunk already exists: layer={layer}, index={index}"
        )
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    save_file({key: tensors[key].contiguous() for key in _TENSOR_KEYS}, str(temporary))
    temporary.replace(path)
    receipt = {
        "kind": "qsrt_all_routed_rows_chunk",
        "schema_version": _SCHEMA_VERSION,
        "layer": layer,
        "index": index,
        "row_begin": row_begin,
        "row_end": row_begin + rows,
        "rows": rows,
        "file": path.name,
        "sha256": _sha256(path),
        "request_index_first": int(tensors["request_index"][0]),
        "request_index_last": int(tensors["request_index"][-1]),
        "token_offset_first": int(tensors["token_offset"][0]),
        "token_offset_last": int(tensors["token_offset"][-1]),
    }
    _atomic_json(receipt_path, receipt)
    return receipt


def _existing_layer_state(root: Path, layer: int) -> tuple[int, int, int]:
    receipts = sorted((root / f"layer-{layer:05d}").glob("chunk-*.json"))
    row_end = 0
    last_request = -1
    for expected_index, path in enumerate(receipts):
        receipt = _read_json(path)
        if (
            int(receipt.get("index", -1)) != expected_index
            or int(receipt.get("row_begin", -1)) != row_end
        ):
            raise ValueError(f"all-row layer {layer} has a noncontiguous chunk index")
        row_end = int(receipt["row_end"])
        last_request = int(receipt["request_index_last"])
    return len(receipts), row_end, last_request


def _writer_main(
    root_text: str,
    worker: int,
    endpoint: str,
    layers: tuple[int, ...],
    chunk_rows: int,
) -> None:
    root = Path(root_text)
    pending: dict[int, dict[str, list[torch.Tensor]]] = {
        layer: {key: [] for key in _TENSOR_KEYS} for layer in layers
    }
    pending_rows = {layer: 0 for layer in layers}
    next_index: dict[int, int] = {}
    next_row: dict[int, int] = {}
    last_request: dict[int, int] = {}
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.setsockopt(zmq.RCVHWM, 2)
    socket.bind(endpoint)
    ready_path = root / f"writer-{worker:02d}.ready"
    _atomic_json(ready_path, {"worker": worker, "pid": os.getpid()})
    try:
        for layer in layers:
            next_index[layer], next_row[layer], last_request[layer] = (
                _existing_layer_state(root, layer)
            )

        def flush_layer(layer: int) -> None:
            if pending_rows[layer] == 0:
                return
            tensors = {
                key: torch.cat(values, dim=0) for key, values in pending[layer].items()
            }
            receipt = _write_chunk(
                root,
                layer=layer,
                index=next_index[layer],
                row_begin=next_row[layer],
                tensors=tensors,
            )
            next_index[layer] += 1
            next_row[layer] = int(receipt["row_end"])
            last_request[layer] = int(receipt["request_index_last"])
            pending[layer] = {key: [] for key in _TENSOR_KEYS}
            pending_rows[layer] = 0
            progress_path = root / f"writer-{worker:02d}.json"
            _atomic_json(
                progress_path,
                {
                    "kind": "qsrt_all_routed_rows_writer",
                    "schema_version": _SCHEMA_VERSION,
                    "worker": worker,
                    "layers": list(layers),
                    "rows_by_layer": {str(value): next_row[value] for value in layers},
                    "last_request_by_layer": {
                        str(value): last_request[value] for value in layers
                    },
                    "complete": False,
                },
            )

        while True:
            item = socket.recv_pyobj()
            command = item[0]
            if command == "append":
                _, layer, tensors = item
                request_indices = tensors["request_index"]
                keep = request_indices > last_request[layer]
                if not bool(torch.all(keep)):
                    tensors = {key: value[keep] for key, value in tensors.items()}
                if not tensors["input"].numel():
                    continue
                for key in _TENSOR_KEYS:
                    pending[layer][key].append(tensors[key])
                pending_rows[layer] += int(tensors["input"].shape[0])
                # Chunks close only at batch boundaries.  Calibration requests
                # are single-document prefills, so a crash cannot persist half
                # of a document.
                if pending_rows[layer] >= chunk_rows:
                    flush_layer(layer)
            elif command == "finalize":
                for layer in layers:
                    flush_layer(layer)
                progress = {
                    "kind": "qsrt_all_routed_rows_writer",
                    "schema_version": _SCHEMA_VERSION,
                    "worker": worker,
                    "layers": list(layers),
                    "rows_by_layer": {str(layer): next_row[layer] for layer in layers},
                    "last_request_by_layer": {
                        str(layer): last_request[layer] for layer in layers
                    },
                    "complete": True,
                }
                progress_path = root / f"writer-{worker:02d}.json"
                _atomic_json(progress_path, progress)
                return
            else:
                raise ValueError(f"unknown all-row writer command {command!r}")
    finally:
        ready_path.unlink(missing_ok=True)
        socket.close(linger=0)
        context.term()


class AllRowCaptureState:
    """Runtime state for the canonical all-row calibration profile."""

    captures_mid = False

    def __init__(
        self,
        *,
        device: torch.device,
        rank: int,
        world_size: int,
        max_tokens: int,
        root: Path,
        model: str,
        revision: str,
    ) -> None:
        if device.type != "cuda":
            raise RuntimeError("all-row capture requires CUDA")
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.max_tokens = max_tokens
        self.root = root
        self.model = model
        self.revision = revision
        self.run_id = os.getenv("VLLM_KQUANT_CAPTURE_RUN_ID", root.name)
        self.chunk_rows = int(os.getenv("VLLM_KQUANT_CHUNK_ROWS", "16384"))
        self.writer_count = int(os.getenv("VLLM_KQUANT_WRITER_PROCESSES", "4"))
        self.writer_queue_depth = int(os.getenv("VLLM_KQUANT_WRITER_QUEUE_DEPTH", "2"))
        self.finalize_file = Path(
            os.getenv("VLLM_KQUANT_FINALIZE_FILE", str(root) + ".finalize")
        )
        if (
            self.chunk_rows <= 0
            or self.writer_count != 4
            or self.writer_queue_depth <= 0
        ):
            raise ValueError(
                "all-row capture requires positive chunk/queue sizes and exactly "
                "four writers"
            )
        self.prefixes: dict[int, str] = {}
        self.registered = torch.zeros(_NUM_MOE_LAYERS, dtype=torch.bool)
        self.input_values: torch.Tensor | None = None
        self.expert_indices: torch.Tensor | None = None
        self.route_weights: torch.Tensor | None = None
        self.routed_output: torch.Tensor | None = None
        self.capacity = 0
        self.route_ready = torch.zeros(_NUM_MOE_LAYERS, dtype=torch.bool)
        self.output_ready = torch.zeros(_NUM_MOE_LAYERS, dtype=torch.bool)
        self.batch_metadata: dict[str, torch.Tensor] | None = None
        self.batch_rows = 0
        self.steps = 0
        self.finalized = False
        self.processes: list[subprocess.Popen[bytes]] = []
        self.writer_logs: list[Any] = []
        self.writer_sockets: list[zmq.Socket] = []
        self.zmq_context: zmq.Context | None = None
        self.sealed_request_by_layer = {
            layer: -1
            for layer in range(_FIRST_MOE_LAYER, _FIRST_MOE_LAYER + _NUM_MOE_LAYERS)
        }
        if rank == 0:
            self._initialize_root()

    def _initialize_root(self) -> None:
        corpus_path = Path(os.environ["VLLM_KQUANT_CORPUS"]).resolve()
        if not corpus_path.is_file():
            raise FileNotFoundError(
                f"all-row corpus manifest is missing: {corpus_path}"
            )
        corpus = _read_json(corpus_path)
        plan_sha256 = str(corpus.get("plan_sha256", ""))
        if len(plan_sha256) != 64:
            raise ValueError("all-row corpus report lacks an immutable plan SHA-256")
        resident = os.getenv("VLLM_KQUANT_TEACHER_CHECKPOINT", "").strip()
        if not resident:
            raise ValueError("VLLM_KQUANT_TEACHER_CHECKPOINT is required")
        expected_rows = os.getenv("VLLM_KQUANT_EXPECTED_ROWS")
        if expected_rows is None:
            expected_rows = str(corpus.get("planned_tokens", "")) or None
        manifest: dict[str, Any] = {
            "kind": _KIND,
            "schema_version": _SCHEMA_VERSION,
            "complete": False,
            "run_id": self.run_id,
            "model": self.model,
            "revision": self.revision,
            "resident_checkpoint": str(Path(resident).resolve()),
            "teacher_checkpoint": str(Path(resident).resolve()),
            "source": os.getenv("VLLM_KQUANT_SOURCE", ""),
            "corpus": str(corpus_path),
            "corpus_file_sha256_at_start": _sha256(corpus_path),
            "corpus_manifest_sha256": plan_sha256,
            "geometry": {
                "layers": list(
                    range(_FIRST_MOE_LAYER, _FIRST_MOE_LAYER + _NUM_MOE_LAYERS)
                ),
                "input_size": _INPUT_SIZE,
                "top_k": _TOP_K,
            },
            "chunk_rows": self.chunk_rows,
            "expected_rows": int(expected_rows) if expected_rows else None,
            "tp_world_size": self.world_size,
            "canonical_tensor_rank": 0,
            "route_weight_convention": "applied_gate; squared_once_in_sse",
            "row_identity": "request_index,document_id,token_offset,role",
        }
        path = self.root / "manifest.json"
        self.root.mkdir(parents=True, exist_ok=True)
        if path.exists():
            existing = _read_json(path)
            immutable = (
                "kind",
                "schema_version",
                "run_id",
                "model",
                "revision",
                "resident_checkpoint",
                "teacher_checkpoint",
                "source",
                "corpus",
                "corpus_manifest_sha256",
                "geometry",
                "chunk_rows",
                "expected_rows",
                "tp_world_size",
                "canonical_tensor_rank",
                "route_weight_convention",
                "row_identity",
            )
            mismatched = [
                key for key in immutable if existing.get(key) != manifest[key]
            ]
            if mismatched or bool(existing.get("complete", False)):
                raise ValueError(
                    f"all-row capture identity differs from {path}: {mismatched}"
                )
        else:
            _atomic_json(path, manifest)
        for layer in manifest["geometry"]["layers"]:
            (self.root / f"layer-{layer:05d}").mkdir(exist_ok=True)

    def register(self, prefix: str) -> None:
        match = _LAYER_RE.search(prefix)
        if match is None:
            raise ValueError(
                f"cannot determine decoder layer from MoE prefix {prefix!r}"
            )
        row = int(match.group(1)) - _FIRST_MOE_LAYER
        if not 0 <= row < _NUM_MOE_LAYERS:
            raise ValueError(f"all-row capture received non-MoE layer {prefix!r}")
        old = self.prefixes.get(row)
        if old is not None and old != prefix:
            raise RuntimeError(f"all-row layer collision: {old!r} versus {prefix!r}")
        self.prefixes[row] = prefix
        self.registered[row] = True

    def _start(self, rows: int) -> None:
        if self.rank != 0:
            return
        if len(self.prefixes) != _NUM_MOE_LAYERS:
            missing = sorted(set(range(_NUM_MOE_LAYERS)) - self.prefixes.keys())
            raise RuntimeError(
                f"all-row capture is missing MoE layer rows {missing[:16]}"
            )
        if rows > self.capacity:
            capacity = min(self.max_tokens, 1 << (rows - 1).bit_length())
            self.input_values = torch.empty(
                (_NUM_MOE_LAYERS, capacity, _INPUT_SIZE),
                device="cpu",
                dtype=torch.bfloat16,
                pin_memory=True,
            )
            self.expert_indices = torch.empty(
                (_NUM_MOE_LAYERS, capacity, _TOP_K),
                device="cpu",
                dtype=torch.int32,
                pin_memory=True,
            )
            self.route_weights = torch.empty(
                (_NUM_MOE_LAYERS, capacity, _TOP_K),
                device="cpu",
                dtype=torch.float32,
                pin_memory=True,
            )
            self.routed_output = torch.empty(
                (_NUM_MOE_LAYERS, capacity, _INPUT_SIZE),
                device="cpu",
                dtype=torch.bfloat16,
                pin_memory=True,
            )
            self.capacity = capacity
        if self.processes:
            return
        self.zmq_context = zmq.Context()
        for worker in range(self.writer_count):
            layers = tuple(
                layer
                for layer in range(_FIRST_MOE_LAYER, _FIRST_MOE_LAYER + _NUM_MOE_LAYERS)
                if (layer - _FIRST_MOE_LAYER) % self.writer_count == worker
            )
            endpoint = f"ipc://{self.root}/writer-{worker:02d}.sock"
            log_path = self.root / f"writer-{worker:02d}.log"
            (self.root / f"writer-{worker:02d}.ready").unlink(missing_ok=True)
            log_stream = log_path.open("ab", buffering=0)
            self.writer_logs.append(log_stream)
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "vllm.model_executor.layers.fused_moe.kquant_all_row_capture",
                    "--writer",
                    str(self.root),
                    str(worker),
                    endpoint,
                    str(self.chunk_rows),
                    ",".join(str(layer) for layer in layers),
                ],
                stdin=subprocess.DEVNULL,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                close_fds=True,
            )
            socket = self.zmq_context.socket(zmq.PUSH)
            socket.setsockopt(zmq.SNDHWM, self.writer_queue_depth)
            socket.connect(endpoint)
            self.writer_sockets.append(socket)
            self.processes.append(process)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if all(
                (self.root / f"writer-{worker:02d}.ready").is_file()
                for worker in range(self.writer_count)
            ):
                break
            failed = [
                process.returncode for process in self.processes if process.poll()
            ]
            if failed:
                raise RuntimeError(f"all-row writers exited during startup: {failed}")
            time.sleep(0.05)
        else:
            raise TimeoutError("all-row writers did not bind their IPC endpoints")

    @staticmethod
    def _request_identity(request_id: str) -> tuple[int, int]:
        match = _REQUEST_RE.search(request_id)
        if match is None:
            raise ValueError(
                "all-row capture request IDs must contain "
                "qsrtcap-<decimal-index>-<32-hex-document-hash>"
            )
        request_index = int(match.group(1))
        document_id = int.from_bytes(
            bytes.fromhex(match.group(2))[:8], "little", signed=True
        )
        return request_index, document_id

    def prepare_batch(self, input_batch: Any) -> None:
        if self.finalized:
            return
        rows = int(input_batch.num_tokens_after_padding)
        if rows > self.max_tokens:
            raise RuntimeError(
                f"all-row batch has {rows} rows, capacity is {self.max_tokens}"
            )
        metadata = {
            "request_index": torch.full((rows,), -1, dtype=torch.int64),
            "document_id": torch.zeros(rows, dtype=torch.int64),
            "token_offset": torch.full((rows,), -1, dtype=torch.int32),
            "role": torch.ones(rows, dtype=torch.uint8),
            "keep": torch.zeros(rows, dtype=torch.bool),
        }
        for request_position, request_id in enumerate(input_batch.req_ids):
            begin = int(input_batch.query_start_loc_np[request_position])
            end = int(input_batch.query_start_loc_np[request_position + 1])
            if end <= begin:
                continue
            prefill_remaining = max(
                0,
                int(input_batch.prefill_len_np[request_position])
                - int(input_batch.num_computed_prefill_tokens_np[request_position]),
            )
            prompt_end = min(end, begin + prefill_remaining)
            if prompt_end <= begin:
                continue
            if _REQUEST_RE.search(request_id) is None:
                continue
            request_index, document_id = self._request_identity(request_id)
            offsets = torch.arange(
                int(input_batch.num_computed_tokens_np[request_position]),
                int(input_batch.num_computed_tokens_np[request_position])
                + prompt_end
                - begin,
                dtype=torch.int32,
            )
            metadata["request_index"][begin:prompt_end] = request_index
            metadata["document_id"][begin:prompt_end] = document_id
            metadata["token_offset"][begin:prompt_end] = offsets
            metadata["role"][begin:prompt_end] = 0
            metadata["keep"][begin:prompt_end] = True
        if not bool(torch.any(metadata["keep"])):
            # Profiling and kernel warmup use synthetic request IDs.  Deferring
            # allocation until an authenticated corpus request also keeps the
            # capture ring out of peak startup memory.
            self.batch_metadata = None
            self.batch_rows = 0
            return
        if not bool(torch.all(metadata["keep"])):
            raise RuntimeError(
                "all-row capture batches cannot mix authenticated corpus rows "
                "with scheduler padding, decode rows, or unrelated requests"
            )
        if self.rank == 0:
            self._start(rows)
        self.batch_rows = rows
        self.batch_metadata = metadata
        self.route_ready.zero_()
        self.output_ready.zero_()

    def collect_route_input(
        self,
        prefix: str,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> None:
        if self.rank != 0:
            return
        match = _LAYER_RE.search(prefix)
        if match is None:
            raise ValueError(
                f"cannot determine decoder layer from MoE prefix {prefix!r}"
            )
        row = int(match.group(1)) - _FIRST_MOE_LAYER
        if not 0 <= row < _NUM_MOE_LAYERS or row not in self.prefixes:
            raise RuntimeError(
                f"all-row route capture received unregistered layer {prefix!r}"
            )
        if self.batch_metadata is None:
            # vLLM runs synthetic forwards while profiling available memory.
            # Only scheduler batches carry authenticated request identities.
            return
        rows = int(x.shape[0])
        if rows != self.batch_rows:
            raise RuntimeError(
                f"all-row route rows differ: {rows} versus {self.batch_rows}"
            )
        if (
            tuple(x.shape) != (rows, _INPUT_SIZE)
            or tuple(topk_ids.shape) != (rows, _TOP_K)
            or tuple(topk_weights.shape) != (rows, _TOP_K)
            or topk_ids.dtype != torch.int32
            or topk_weights.dtype != torch.float32
        ):
            raise RuntimeError(
                "all-row route tensors do not match the Kimi-K3 contract"
            )
        assert self.input_values is not None
        assert self.expert_indices is not None
        assert self.route_weights is not None
        self.input_values[row, :rows].copy_(x, non_blocking=True)
        self.expert_indices[row, :rows].copy_(topk_ids, non_blocking=True)
        self.route_weights[row, :rows].copy_(topk_weights, non_blocking=True)
        self.route_ready[row] = True

    def collect_routed_latent(self, decoder_layer: int, values: torch.Tensor) -> None:
        if self.rank != 0:
            return
        if self.batch_metadata is None:
            return
        row = int(decoder_layer) - _FIRST_MOE_LAYER
        if self.batch_metadata is None or int(values.shape[0]) != self.batch_rows:
            raise RuntimeError(
                "all-row routed output does not align with the active batch"
            )
        if tuple(values.shape) != (self.batch_rows, _INPUT_SIZE):
            raise RuntimeError("all-row routed output has the wrong shape")
        assert self.routed_output is not None
        self.routed_output[row, : self.batch_rows].copy_(values, non_blocking=True)
        self.output_ready[row] = True

    def collect_mid(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def _check_writer_results(self) -> None:
        failed = [
            (worker, process.returncode)
            for worker, process in enumerate(self.processes)
            if process.poll() is not None and process.returncode != 0
        ]
        if failed:
            raise RuntimeError(f"all-row writer processes failed: {failed}")
        changed = False
        for worker in range(self.writer_count):
            path = self.root / f"writer-{worker:02d}.json"
            if not path.is_file():
                continue
            progress = _read_json(path)
            for layer_text, request_index in progress.get(
                "last_request_by_layer", {}
            ).items():
                layer = int(layer_text)
                value = int(request_index)
                if value > self.sealed_request_by_layer[layer]:
                    self.sealed_request_by_layer[layer] = value
                    changed = True
        if changed:
            manifest_path = self.root / "manifest.json"
            manifest = _read_json(manifest_path)
            manifest["sealed_request_index"] = min(
                self.sealed_request_by_layer.values()
            )
            _atomic_json(manifest_path, manifest)

    def _write_rank_receipt(self) -> str:
        payload = {
            "kind": "qsrt_all_routed_rows_rank",
            "schema_version": _SCHEMA_VERSION,
            "run_id": self.run_id,
            "rank": self.rank,
            "tp_world_size": self.world_size,
            "canonical_tensors": self.rank == 0,
            "steps": self.steps,
            "complete": True,
        }
        path = self.root / f"rank-{self.rank:05d}.json"
        _atomic_json(path, payload)
        return _sha256(path)

    def _seal_root(self) -> None:
        deadline = time.monotonic() + 300
        rank_receipts: dict[int, str] = {}
        while time.monotonic() < deadline:
            rank_receipts = {
                rank: _sha256(path)
                for rank in range(self.world_size)
                if (path := self.root / f"rank-{rank:05d}.json").is_file()
            }
            if len(rank_receipts) == self.world_size:
                break
            time.sleep(0.1)
        if len(rank_receipts) != self.world_size:
            raise TimeoutError("all-row TP rank receipts did not close")
        rows_by_layer: dict[str, int] = {}
        for layer in range(_FIRST_MOE_LAYER, _FIRST_MOE_LAYER + _NUM_MOE_LAYERS):
            _, rows, _ = _existing_layer_state(self.root, layer)
            if rows <= 0:
                raise RuntimeError(f"all-row layer {layer} contains no rows")
            rows_by_layer[str(layer)] = rows
        if len(set(rows_by_layer.values())) != 1:
            raise RuntimeError("all-row capture layers contain different row counts")
        rows = next(iter(rows_by_layer.values()))
        manifest_path = self.root / "manifest.json"
        manifest = _read_json(manifest_path)
        expected = manifest.get("expected_rows")
        if expected is not None and rows != int(expected):
            raise RuntimeError(
                f"all-row capture contains {rows} rows; expected {expected}"
            )
        manifest.update(
            {
                "complete": True,
                "rows": rows,
                "rows_by_layer": rows_by_layer,
                "rank_receipts": {
                    str(rank): digest for rank, digest in rank_receipts.items()
                },
            }
        )
        _atomic_json(manifest_path, manifest)

    def flush_and_arm(self) -> None:
        if self.finalized or self.batch_metadata is None:
            return
        finalize = self.finalize_file.exists()
        if self.rank == 0:
            if not bool(torch.all(self.route_ready)) or not bool(
                torch.all(self.output_ready)
            ):
                missing_route = torch.nonzero(~self.route_ready).flatten().tolist()
                missing_output = torch.nonzero(~self.output_ready).flatten().tolist()
                raise RuntimeError(
                    "all-row capture did not observe every MoE layer: "
                    f"route={missing_route[:16]}, output={missing_output[:16]}"
                )
            assert self.input_values is not None
            assert self.expert_indices is not None
            assert self.route_weights is not None
            assert self.routed_output is not None
            rows = self.batch_rows
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self.device))
            event.synchronize()
            keep = self.batch_metadata["keep"]
            if not bool(torch.all(keep)):
                raise RuntimeError(
                    "all-row calibration accepts prompt-only batches without "
                    "scheduler padding; use one pretokenized request at a time "
                    "and enforce eager execution"
                )
            metadata = {
                key: value.contiguous()
                for key, value in self.batch_metadata.items()
                if key != "keep"
            }
            for row in range(_NUM_MOE_LAYERS):
                # A view retains the complete [layer, row, feature] ring when
                # serialized by torch.  Clone each layer into exact-sized
                # storage before crossing the process boundary.
                tensors = {
                    "input": self.input_values[row, :rows].clone(),
                    "expert_indices": self.expert_indices[row, :rows].clone(),
                    "route_weights": self.route_weights[row, :rows].clone(),
                    "routed_output": self.routed_output[row, :rows].clone(),
                }
                tensors.update(metadata)
                worker = row % self.writer_count
                self._check_writer_results()
                self.writer_sockets[worker].send_pyobj(
                    ("append", row + _FIRST_MOE_LAYER, tensors)
                )
            self._check_writer_results()
        self.steps += 1
        self.batch_metadata = None
        self.batch_rows = 0
        if not finalize:
            return
        if self.rank == 0:
            for socket in self.writer_sockets:
                socket.send_pyobj(("finalize",))
            for process in self.processes:
                try:
                    process.wait(timeout=1800)
                except subprocess.TimeoutExpired as exc:
                    raise TimeoutError(
                        f"all-row writer {process.pid} did not exit"
                    ) from exc
                if process.returncode != 0:
                    raise RuntimeError(
                        f"all-row writer {process.pid} exited with {process.returncode}"
                    )
            for stream in self.writer_logs:
                stream.close()
            self._check_writer_results()
            if not all(
                bool(
                    _read_json(self.root / f"writer-{worker:02d}.json").get("complete")
                )
                for worker in range(self.writer_count)
            ):
                raise RuntimeError("all-row writer receipts did not close")
            for socket in self.writer_sockets:
                socket.close(linger=0)
            assert self.zmq_context is not None
            self.zmq_context.term()
        self._write_rank_receipt()
        if self.rank == 0:
            self._seal_root()
        self.finalized = True
        logger.info(
            "Finalized all-row QSRT capture on TP rank %d at %s after %d steps",
            self.rank,
            self.root,
            self.steps,
        )


__all__ = ["AllRowCaptureState"]


if __name__ == "__main__":
    if len(sys.argv) != 7 or sys.argv[1] != "--writer":
        raise SystemExit(
            "usage: python -m ...kquant_all_row_capture "
            "--writer ROOT WORKER ENDPOINT CHUNK_ROWS LAYERS"
        )
    _writer_main(
        sys.argv[2],
        int(sys.argv[3]),
        sys.argv[4],
        tuple(int(layer) for layer in sys.argv[6].split(",")),
        int(sys.argv[5]),
    )
