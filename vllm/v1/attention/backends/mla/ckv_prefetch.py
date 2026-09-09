# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persistent native-KV rings with explicit execution-lane and event ownership.

The caller supplies CUDA streams/events and issues the gathers and attention.
This component orders storage reuse; it never issues collectives or allocates
an overflow ring. Pool construction belongs before KV memory admission.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch


class CompletionEvent(Protocol):
    def synchronize(self) -> None: ...


class OrderedStream(Protocol):
    def wait_event(self, event: CompletionEvent) -> None: ...

    def wait_stream(self, stream: OrderedStream) -> None: ...


@dataclass(frozen=True)
class CKVPrefetchPlan:
    effective_depth: int
    dcp_world_size: int
    local_capacity: int
    record_bytes: int
    num_ubatches: int
    num_lanes: int

    def __post_init__(self) -> None:
        if (
            self.effective_depth < 0
            or min(
                self.dcp_world_size,
                self.local_capacity,
                self.record_bytes,
                self.num_ubatches,
                self.num_lanes,
            )
            < 1
        ):
            raise ValueError(
                "CKV reservation requires non-negative depth and positive geometry"
            )

    @classmethod
    def create(
        cls,
        *,
        requested_depth: int,
        budget_bytes: int,
        dcp_world_size: int,
        local_capacity: int,
        record_bytes: int,
        num_ubatches: int = 1,
        num_lanes: int = 1,
    ) -> CKVPrefetchPlan:
        """Choose lookahead within a per-lane budget; zero means uncapped."""
        if requested_depth < 0 or budget_bytes < 0:
            raise ValueError("CKV depth and budget must be non-negative")
        if (
            min(dcp_world_size, local_capacity, record_bytes, num_ubatches, num_lanes)
            < 1
        ):
            raise ValueError("CKV geometry and execution-lane counts must be positive")
        unit = local_capacity * record_bytes
        minimum = (1 + dcp_world_size) * unit
        if budget_bytes and budget_bytes < minimum:
            raise ValueError(
                f"CKV budget {budget_bytes} bytes cannot cover mandatory "
                f"depth-zero storage of {minimum} bytes per lane"
            )
        depth = requested_depth
        if budget_bytes:
            depth = min(depth, (budget_bytes // unit - 1) // dcp_world_size - 1)
        return cls(
            depth, dcp_world_size, local_capacity, record_bytes, num_ubatches, num_lanes
        )

    @property
    def ring_slots(self) -> int:
        return self.effective_depth + 1

    @property
    def lane_nbytes(self) -> int:
        return (
            (1 + self.dcp_world_size * self.ring_slots)
            * self.local_capacity
            * self.record_bytes
        )

    @property
    def total_nbytes(self) -> int:
        return self.lane_nbytes * self.num_ubatches * self.num_lanes


class CKVWorkspacePool:
    """Reserve disjoint persistent storage for every configured execution lane."""

    def __init__(self, plan: CKVPrefetchPlan, device: torch.device) -> None:
        self.plan = plan
        self.storage = torch.empty(plan.total_nbytes, dtype=torch.uint8, device=device)
        self._leased: set[tuple[int, int]] = set()

    def acquire(self, lane: tuple[int, int]) -> torch.Tensor:
        ubatch, model_lane = lane
        if not (
            0 <= ubatch < self.plan.num_ubatches
            and 0 <= model_lane < self.plan.num_lanes
        ):
            raise ValueError(f"CKV execution lane {lane} is outside the reservation")
        if lane in self._leased:
            raise RuntimeError(f"CKV execution lane {lane} already has a storage lease")
        self._leased.add(lane)
        offset = (ubatch * self.plan.num_lanes + model_lane) * self.plan.lane_nbytes
        return self.storage.narrow(0, offset, self.plan.lane_nbytes)

    def _release(self, lane: tuple[int, int]) -> None:
        self._leased.remove(lane)


@dataclass(frozen=True)
class _WorkspaceIdentity:
    device: torch.device
    storage_ptr: int
    storage_nbytes: int
    data_ptr: int
    offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype


def _identity(workspace: torch.Tensor) -> _WorkspaceIdentity:
    storage = workspace.untyped_storage()
    return _WorkspaceIdentity(
        workspace.device,
        storage.data_ptr(),
        storage.nbytes(),
        workspace.data_ptr(),
        workspace.storage_offset(),
        tuple(workspace.shape),
        tuple(workspace.stride()),
        workspace.dtype,
    )


@dataclass
class _Slot:
    layer: int
    writer: CompletionEvent | None = None
    consumer: CompletionEvent | None = None
    consuming: bool = False


class CKVPrefetchState:
    """Cache discovery and ordered ring reuse for one leased execution lane.

    Each gather must finish by registering an event, and each consumed slot must
    register a consumer event. Events must be recorded after the corresponding
    operation on its stream. A failed launch must be fenced before these hooks
    finish; an unfinished operation prevents lease release.
    """

    def __init__(
        self,
        pool: CKVWorkspacePool,
        lane: tuple[int, int],
        workspace: torch.Tensor,
        generation: int,
    ) -> None:
        self.pool = pool
        self.lane = lane
        self.generation = generation
        self.workspace_identity = _identity(workspace)
        self.workspace_storage_ref = weakref.ref(workspace.untyped_storage())
        self.storage = pool.acquire(lane)
        self.layer_caches: dict[int, torch.Tensor] = {}
        self.pending: dict[int, int] = {}
        self._slots: list[_Slot | None] = [None] * pool.plan.ring_slots
        self._staging_writer: CompletionEvent | None = None
        self._writing: int | None = None
        self._last_layer: int | None = None
        self._closed = False
        self._gather_stream: OrderedStream | None = None

    def get_gather_stream(
        self, factory: Callable[[], OrderedStream] | None = None
    ) -> OrderedStream:
        """Create one dedicated stream for this execution lane on first use."""
        self._check_open()
        if self._gather_stream is None:
            self._gather_stream = (
                factory()
                if factory is not None
                else torch.cuda.Stream(device=self.storage.device)
            )
        return self._gather_stream

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("CKV prefetch state is closed")

    def _check_finished(self) -> None:
        if self._writing is not None or any(
            slot is not None and slot.consuming and slot.consumer is None
            for slot in self._slots
        ):
            raise RuntimeError("CKV operation has no completion event")

    def views(self, slot: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return shared staging and one fixed-capacity gathered-cache view."""
        self._check_open()
        plan = self.pool.plan
        if not 0 <= slot < plan.ring_slots:
            raise ValueError(f"CKV ring slot {slot} is outside the reservation")
        unit = plan.local_capacity * plan.record_bytes
        staging = self.storage[:unit].view(plan.local_capacity, plan.record_bytes)
        gathered = self.storage.narrow(
            0, unit + slot * unit * plan.dcp_world_size, unit * plan.dcp_world_size
        )
        return staging, gathered.view(
            plan.dcp_world_size * plan.local_capacity, plan.record_bytes
        )

    def order_staging(self, stream: OrderedStream) -> None:
        """Order synchronous fallback after every preceding staging writer."""
        self._check_open()
        if self._writing is not None:
            raise RuntimeError("CKV gather has no completion event")
        if self._staging_writer is not None:
            stream.wait_event(self._staging_writer)

    def begin_step(self, stream: OrderedStream) -> None:
        self._check_open()
        self._check_finished()
        self.order_staging(stream)
        self.pending.clear()
        self._last_layer = None

    def enter_layer(self, layer: int, stream: OrderedStream) -> None:
        self._check_open()
        if layer < 0:
            raise ValueError("CKV layer index must be non-negative")
        if self._last_layer is not None and layer <= self._last_layer:
            self.begin_step(stream)
        self._last_layer = layer

    def reset_caches(self) -> None:
        """Finish all users before invalidating profiling or rebound caches."""
        self._check_open()
        self._synchronize()
        self.layer_caches.clear()
        self.pending.clear()
        self._slots = [None] * self.pool.plan.ring_slots
        self._staging_writer = None
        self._last_layer = None

    def register_cache(self, layer: int, cache: torch.Tensor) -> None:
        self._check_open()
        if layer < 0:
            raise ValueError("CKV layer index must be non-negative")
        previous = self.layer_caches.get(layer)
        if previous is not None and previous is not cache:
            self.reset_caches()
        self.layer_caches[layer] = cache

    def targets(self, layer: int) -> list[int]:
        """Stop lookahead at an undiscovered cache; skip queued layers."""
        self._check_open()
        targets = []
        for target in range(layer + 1, layer + self.pool.plan.effective_depth + 1):
            if target not in self.layer_caches:
                break
            if target not in self.pending:
                targets.append(target)
        return targets

    def prepare_gather(
        self, layer: int, stream: OrderedStream, *, producer_stream: OrderedStream
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fence cache producers, shared staging, and the slot's prior consumer."""
        self._check_open()
        if layer < 0 or layer in self.pending:
            raise ValueError(f"CKV gather layer {layer} is invalid or already pending")
        self.order_staging(stream)
        index = layer % self.pool.plan.ring_slots
        previous = self._slots[index]
        if previous is not None:
            if previous.layer in self.pending:
                raise RuntimeError("CKV ring would overwrite an unconsumed layer")
            if previous.consuming and previous.consumer is None:
                raise RuntimeError("CKV consumer has no completion event")
            if previous.consumer is not None:
                stream.wait_event(previous.consumer)
            elif previous.writer is not None:
                stream.wait_event(previous.writer)
        stream.wait_stream(producer_stream)
        self._slots[index] = _Slot(layer)
        self._writing = index
        return self.views(index)

    def finish_gather(self, layer: int, event: CompletionEvent) -> None:
        self._check_open()
        if event is None:
            raise ValueError("CKV gather requires a recorded completion event")
        index = self._writing
        slot = self._slots[index] if index is not None else None
        if index is None or slot is None or slot.layer != layer:
            raise RuntimeError("CKV gather completion has no matching producer")
        slot.writer = event
        self._staging_writer = event
        self.pending[layer] = index
        self._writing = None

    def consume(self, layer: int, stream: OrderedStream) -> torch.Tensor:
        """Join history gathering before chunk insertion and attention reads."""
        self._check_open()
        index = self.pending.pop(layer)
        slot = self._slots[index]
        assert slot is not None and slot.writer is not None
        stream.wait_event(slot.writer)
        slot.consuming = True
        return self.views(index)[1]

    def finish_consumer(self, layer: int, event: CompletionEvent) -> None:
        self._check_open()
        if event is None:
            raise ValueError("CKV consumer requires a recorded completion event")
        slot = self._slots[layer % self.pool.plan.ring_slots]
        if slot is None or slot.layer != layer or not slot.consuming:
            raise RuntimeError("CKV consumer completion has no matching reader")
        if slot.consumer is not None:
            raise RuntimeError("CKV consumer completion was already registered")
        slot.consumer = event

    def _synchronize(self) -> None:
        self._check_finished()
        for slot in self._slots:
            if slot is not None:
                if slot.writer is not None:
                    slot.writer.synchronize()
                if slot.consumer is not None:
                    slot.consumer.synchronize()

    def close(self) -> None:
        """Complete writers/readers before another lane generation can lease."""
        if self._closed:
            return
        self._synchronize()
        self.pool._release(self.lane)
        self.layer_caches.clear()
        self.pending.clear()
        self._closed = True


class CKVPrefetchRegistry:
    """Separate target/drafter and ubatch states using workspace lane keys."""

    def __init__(self, pool: CKVWorkspacePool) -> None:
        self.pool = pool
        self.states: dict[tuple[int, int], CKVPrefetchState] = {}

    def for_workspace(
        self, workspace: torch.Tensor, *, lane: tuple[int, int], generation: int = 0
    ) -> CKVPrefetchState:
        for key, previous_state in list(self.states.items()):
            if previous_state.workspace_storage_ref() is None:
                previous_state.close()
                del self.states[key]
        identity = _identity(workspace)
        state = self.states.get(lane)
        if state is not None and (
            state.workspace_identity != identity or state.generation != generation
        ):
            state.close()
            del self.states[lane]
            state = None
        if state is None:
            state = CKVPrefetchState(self.pool, lane, workspace, generation)
            self.states[lane] = state
        return state

    def reset_caches(self) -> None:
        for state in self.states.values():
            state.reset_caches()

    def clear(self) -> None:
        for lane, state in list(self.states.items()):
            state.close()
            del self.states[lane]
