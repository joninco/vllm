# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded host-only ownership annotations for diagnosis captures.

Enabled workers retain event-object identities through exact record/wait calls.
Scopes describe enqueue operations, not GPU completion. An opaque wait_stream
edge never claims the identity of PyTorch's internal event. No CUDA operation
is introduced by this module; delegates perform only the original operations.
"""

from __future__ import annotations

import functools
import json
import time
import weakref
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import torch

from vllm import envs

_OWNERSHIP_FIELDS = frozenset(
    {
        "lease",
        "generation",
        "ubatch",
        "lane",
        "slot",
        "use",
        "event",
        "record",
        "stream",
        "group",
        "collective",
        "source_layer",
    }
)


class ExecutionTicketSlot:
    """One exact runner-state handoff, released before sampling enters."""

    def __init__(self, trace):
        self.trace = trace
        self.pending = None

    def clear(self, reason="reset"):
        if self.pending is not None:
            _, _, record = self.pending
            record["disposition"] = reason
            self.pending = None

    def bind(self, state):
        self.clear("replaced")
        trace = self.trace
        if state is None or not trace.active:
            return
        if trace._ticket_count >= len(trace._ticket_records):
            trace.overflow = True
            return
        fields = {key: trace.current_fields[key] for key in ("capture", "batch")}
        fields.update(
            ticket=trace._ticket_count + 1,
            owner_role=0,
            ownership_enabled=0,
            parent_membership=fields["batch"],
        )
        record = {**fields, "disposition": "pending"}
        trace._ticket_records[trace._ticket_count] = record
        trace._ticket_count += 1
        self.pending = state, fields, record
        with trace.scope("execution_handoff", **fields):
            pass

    @contextmanager
    def resume(self, state):
        trace = self.trace
        pending, self.pending = self.pending, None
        if pending is None:
            with trace.unmatched_boundary("missing_ticket"):
                yield
            return
        expected, fields, record = pending
        matches = state is expected and state is not None
        del expected, pending, state
        if not matches or fields["capture"] != trace._capture_index:
            record["disposition"] = (
                "state_mismatch" if not matches else "capture_changed"
            )
            with trace.unmatched_boundary(record["disposition"]):
                yield
            return
        if not torch.autograd.profiler._is_profiler_enabled or trace.overflow:
            record["disposition"] = "capture_inactive"
            with trace.suspend():
                yield
            return
        token = trace._context.set(dict(fields))
        record["disposition"] = "sampling"
        try:
            with trace.scope("sample"):
                yield
            record["disposition"] = "complete"
        except BaseException:
            record["disposition"] = "sample_exception"
            raise
        finally:
            trace._context.reset(token)


class PrefillTrace:
    """A startup-allocated, bounded diagnosis ledger scoped to one worker."""

    def __init__(
        self,
        *,
        max_batches=64,
        max_objects=16384,
        max_members=4096,
        layer_range: tuple[int, int] | None = None,
    ):
        self.max_batches = max_batches
        self.max_objects = max_objects
        self.layer_range = layer_range
        self._context: ContextVar[dict[str, int] | None] = ContextVar(
            "dcp_prefill_trace", default=None
        )
        self._members: list[Any] = [None] * max_members
        self._member_count = 0
        self._requests: dict[str, int] = {}
        self._events: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        self._event_count = 0
        self._streams: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        self._stream_count = 0
        self._stream_handles: dict[int, int] = {}
        self._groups: dict[int, dict[str, Any]] = {}
        self._group_objects: list[Any] = []
        self._leases = 0
        self._batches = 0
        self._executions = 0
        self.overflow = False
        self._capture: dict[str, Any] = {}
        self._capture_index = 0
        self._layers: dict[tuple[str, int], dict[str, Any]] = {}
        self._ticket_slots: list[ExecutionTicketSlot] = []
        self._ticket_records: list[Any] = [None] * max_objects
        self._ticket_count = 0

    def register_layer(self, index: int, name: str, role: int) -> None:
        key = (name, role)
        if len(self._layers) >= self.max_objects and key not in self._layers:
            self.overflow = True
            return
        self._layers[key] = {"layer": index, "name": name, "role": role}

    def start_capture(self, *, rank: int, name: str, device: str) -> None:
        """Reset bounded capture records without changing execution resources."""
        for slot in self._ticket_slots:
            slot.clear("capture_changed")
        for index in range(self._ticket_count):
            self._ticket_records[index] = None
        self._ticket_count = 0
        for index in range(self._member_count):
            self._members[index] = None
        self._member_count = self._batches = self._executions = 0
        self._event_count = self._stream_count = 0
        self._requests.clear()
        self._events.clear()
        self._streams.clear()
        self._stream_handles.clear()
        self._group_objects.clear()
        self._groups.clear()
        self.overflow = False
        self._capture_index += 1
        self._capture = {
            "rank": rank,
            "name": name,
            "device": device,
            "capture": self._capture_index,
            "host_start_unix_ns": time.time_ns(),
        }

    def export_manifest(self, directory: str) -> str:
        """Write a unique rank/capture artifact; callers report export failures."""
        for slot in self._ticket_slots:
            slot.clear("capture_stopped")
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        rank = self._capture.get("rank", -1)
        output = path / (
            f"dcp-prefill-rank{rank}-capture{self._capture_index}-{time.time_ns()}.json"
        )
        payload = self.manifest()
        payload["host_export_unix_ns"] = time.time_ns()
        with output.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
        return str(output)

    def capture_active(self) -> bool:
        return (
            bool(torch.autograd.profiler._is_profiler_enabled)
            and self._batches < self.max_batches
            and not self.overflow
        )

    @property
    def active(self) -> bool:
        return self._context.get() is not None and not self.overflow

    @property
    def ownership_active(self) -> bool:
        return (
            self.active
            and self.current_fields.get("ownership_enabled", 1) == 1
            and self.current_fields.get("owner_role", 0) == 0
        )

    def new_ticket_slot(self):
        slot = ExecutionTicketSlot(self)
        self._ticket_slots.append(slot)
        return slot

    @contextmanager
    def unmatched_boundary(self, reason):
        if not torch.autograd.profiler._is_profiler_enabled or self.overflow:
            with self.suspend():
                yield
            return
        reasons = {"missing_ticket": 1, "state_mismatch": 2, "capture_changed": 3}
        token = self._context.set(
            {
                "capture": self._capture_index,
                "batch": -1,
                "ticket": -1,
                "ownership_enabled": 0,
                "owner_role": -1,
            }
        )
        try:
            with self.scope("ticket_rejected", reason=reasons[reason]):
                pass
            with self.suspend():
                yield
        finally:
            self._context.reset(token)

    @property
    def current_fields(self) -> dict[str, int]:
        return self._context.get() or {}

    def set_owner(self, **fields: int) -> None:
        if self.active:
            self._context.get().update(fields)  # type: ignore[union-attr]

    @contextmanager
    def batch(self, membership=None):
        if not self.capture_active():
            yield
            return
        self._executions += 1
        token = self._context.set(
            {
                "batch": self._executions,
                "capture": self._capture_index,
                "owner_role": 0,
                "ownership_enabled": 1,
            }
        )
        try:
            if membership is not None:
                self.set_membership(membership)
            with self.scope("batch"):
                yield
        finally:
            self._context.reset(token)

    @contextmanager
    def startup(self):
        if not torch.autograd.profiler._is_profiler_enabled or self.overflow:
            yield
            return
        token = self._context.set(
            {
                "batch": -1,
                "domain": 1,
                "capture": self._capture_index,
                "ownership_enabled": 1,
                "owner_role": 0,
            }
        )
        try:
            with self.scope("startup"):
                yield
        finally:
            self._context.reset(token)

    @contextmanager
    def suspend(self):
        """Exclude captured decode, mixed-batch kernels and speculative lanes."""
        token = self._context.set(None)
        try:
            yield
        finally:
            self._context.reset(token)

    def wrap(self, operation, function, **fields):
        @functools.wraps(function)
        def call(*args, **kwargs):
            if not self.ownership_active:
                return function(*args, **kwargs)
            with self.scope(operation, **fields):
                return function(*args, **kwargs)

        return call

    def set_membership(self, records) -> None:
        """Store ordered host records; request strings become opaque ordinals."""
        if not self.active:
            return
        records = list(records)
        if any(record.get("is_prefill", False) for record in records):
            self._batches += 1
        for row, record in enumerate(records):
            if self._member_count == len(self._members):
                self.overflow = True
                return
            item = dict(record)
            request = str(item.pop("request_id", item.pop("req_id", "unknown")))
            if request not in self._requests:
                self._requests[request] = len(self._requests)
            item.update(
                batch=self.current_fields["batch"],
                row=row,
                request=self._requests[request],
            )
            self._members[self._member_count] = item
            self._member_count += 1
            integers = {
                key: int(value)
                for key, value in item.items()
                if isinstance(value, (int, bool)) and key != "batch"
            }
            with self.scope("membership", **integers):
                pass

    @contextmanager
    def scope(self, operation: str, **fields: int):
        if not self.active:
            yield
            return
        merged = dict(self.current_fields)
        if fields.get("ownership_enabled", merged.get("ownership_enabled", 1)) == 0:
            merged = {
                key: value
                for key, value in merged.items()
                if key not in _OWNERSHIP_FIELDS
            }
            fields = {
                key: value
                for key, value in fields.items()
                if key not in _OWNERSHIP_FIELDS
            }
        merged.update(fields)
        layer = merged.get("layer", -1)
        if (
            self.layer_range is not None
            and layer >= 0
            and not self.layer_range[0] <= layer <= self.layer_range[1]
        ):
            token = self._context.set(merged)
            try:
                yield
            finally:
                self._context.reset(token)
            return
        name = (
            "dcp_prefill.v1/"
            + operation
            + " "
            + " ".join(f"{key}={value}" for key, value in sorted(merged.items()))
        )
        token = self._context.set(merged)
        try:
            with torch.profiler.record_function(name):
                try:
                    yield
                except BaseException:
                    with torch.profiler.record_function(name + " exceptional=1"):
                        pass
                    raise
        finally:
            self._context.reset(token)

    def event_identity(self, event, *, recorded=False) -> tuple[int, int]:
        identity = self._events.get(event)
        if identity is None:
            if not recorded or self._event_count >= self.max_objects:
                if recorded:
                    self.overflow = True
                return -1, -1
            self._event_count += 1
            identity = [self._event_count, 0]
            self._events[event] = identity
        if recorded:
            identity[1] += 1
        return identity[0], identity[1]

    def stream_identity(self, stream) -> int:
        # current_stream() can return distinct Python wrappers for one native
        # stream. Reading cuda_stream is a host property, not a device query.
        handle = getattr(stream, "cuda_stream", None)
        if handle is not None:
            if handle not in self._stream_handles:
                if self._stream_count >= self.max_objects:
                    self.overflow = True
                    return -1
                self._stream_handles[handle] = self._stream_count
                self._stream_count += 1
            return self._stream_handles[handle]
        if stream not in self._streams:
            if self._stream_count >= self.max_objects:
                self.overflow = True
                return -1
            self._streams[stream] = self._stream_count
            self._stream_count += 1
        return self._streams[stream]

    def group_fields(self, group) -> dict[str, int]:
        key = id(group)
        if key not in self._groups:
            if len(self._groups) >= self.max_objects:
                self.overflow = True
                return {"group": -1, "collective": -1}
            self._group_objects.append(group)
            self._groups[key] = {
                "id": len(self._groups),
                "ranks": list(getattr(group, "ranks", [])),
                "sequence": 0,
            }
        entry = self._groups[key]
        entry["sequence"] += 1
        return {"group": entry["id"], "collective": entry["sequence"]}

    def record(self, event, stream, **fields: int) -> None:
        if not self.ownership_active:
            event.record(stream)
            return
        event_id, sequence = self.event_identity(event, recorded=True)
        with self.scope(
            "event_record",
            event=event_id,
            record=sequence,
            stream=self.stream_identity(stream),
            **fields,
        ):
            event.record(stream)

    def wait(self, stream, event, *, reason: int) -> None:
        if not self.ownership_active:
            stream.wait_event(event)
            return
        event_id, sequence = self.event_identity(event)
        with self.scope(
            "event_wait",
            event=event_id,
            record=sequence,
            stream=self.stream_identity(stream),
            reason=reason,
        ):
            stream.wait_event(event)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": "dcp_prefill.v1",
            "execution_context_schema": "dcp_prefill.execution.v1",
            "capabilities": ["execution_tickets", "ownership_enabled", "owner_role"],
            "tickets": self._ticket_records[: self._ticket_count],
            "incomplete_tickets": sum(
                record["disposition"] != "complete"
                for record in self._ticket_records[: self._ticket_count]
            ),
            "codes": {
                "attention_plan": {0: "decode", 1: "extend", 2: "full CKV extend"},
                "membership_relation": {
                    0: "target batch rows",
                    1: "parent candidate requests; draft rows unproven",
                },
                "ticket_rejected_reason": {
                    1: "missing ticket",
                    2: "state identity mismatch",
                    3: "capture changed",
                },
                "wait_reason": {
                    1: "staging writer",
                    2: "previous slot consumer",
                    3: "gather consumer",
                    4: "previous slot writer",
                    5: "startup stream join",
                },
                "role": {-1: "unknown", 0: "target", 1: "speculative"},
                "route": {
                    0: "local",
                    1: "configured",
                    2: "full CKV",
                    3: "all-to-all",
                    4: "all-gather/reduce-scatter",
                    5: "projected all-gather/reduce-scatter",
                },
                "domain": {1: "startup; batch=-1"},
                "stream_role": {0: "main", 1: "history prefetch"},
            },
            "capture": dict(self._capture),
            "layers": list(self._layers.values()),
            "ambiguous_layer_roles": [
                list(identity)
                for identity in {
                    (entry["layer"], entry["role"]) for entry in self._layers.values()
                }
                if sum(
                    (entry["layer"], entry["role"]) == identity
                    for entry in self._layers.values()
                )
                > 1
            ],
            "overflow": self.overflow,
            "batch_limit": self.max_batches,
            "observed_prefill_batches": self._batches,
            "observed_scheduler_executions": self._executions,
            "batch_limit_reached": self._batches >= self.max_batches,
            "annotation_complete": False,
            "completeness_reason": "requires exported trace dependency validation",
            "object_limit": self.max_objects,
            "layer_range": self.layer_range,
            "membership": self._members[: self._member_count],
            "groups": list(self._groups.values()),
            "wait_stream_event_identity": "unsupported",
        }


@functools.lru_cache(maxsize=1)
def get_prefill_trace() -> PrefillTrace | None:
    """Resolve the diagnosis flag once; callers install wrappers at startup."""
    return PrefillTrace() if envs.VLLM_DCP_PREFILL_TRACE else None


class _ObservedStream:
    def __init__(self, trace, stream, reason):
        self.trace, self.stream, self.reason = trace, stream, reason

    def wait_event(self, event):
        self.trace.wait(
            self.stream,
            event,
            reason=self.reason(event) if callable(self.reason) else self.reason,
        )

    def wait_stream(self, producer):
        source = producer.stream if isinstance(producer, _ObservedStream) else producer
        if not self.trace.ownership_active:
            self.stream.wait_stream(source)
            return
        with self.trace.scope(
            "stream_wait",
            stream=self.trace.stream_identity(self.stream),
            source_stream=self.trace.stream_identity(source),
            event=-1,
        ):
            self.stream.wait_stream(source)


def install_state_trace(state, trace: PrefillTrace) -> None:
    """Decorate only diagnosed leases; original state classes remain unchanged."""
    if getattr(state, "_prefill_trace_installed", False):
        return
    state._prefill_trace_installed = True
    trace._leases += 1
    lease = trace._leases
    uses = [0] * state.pool.plan.ring_slots
    base = dict(
        lease=lease,
        generation=state.generation,
        ubatch=state.lane[0],
        lane=state.lane[1],
    )
    state._prefill_trace_identity = base
    state._prefill_trace_uses = uses

    def wrap(name, factory):
        setattr(state, name, factory(getattr(state, name)))

    def preparation(original):
        def call(layer, stream, *, producer_stream):
            if not trace.ownership_active:
                return original(layer, stream, producer_stream=producer_stream)
            slot = layer % len(uses)
            uses[slot] += 1

            def wait_reason(event):
                previous = state._slots[slot]
                return 2 if previous is not None and previous.consumer is event else 4

            with trace.scope(
                "slot_acquire",
                **base,
                layer=layer if trace.current_fields.get("layer", -1) >= 0 else -1,
                slot=slot,
                use=uses[slot],
            ):
                return original(
                    layer,
                    _ObservedStream(trace, stream, wait_reason),
                    producer_stream=producer_stream,
                )

        return call

    wrap("prepare_gather", preparation)

    def staging(original):
        def call(stream):
            if not trace.ownership_active:
                return original(stream)
            real = stream.stream if isinstance(stream, _ObservedStream) else stream
            return original(_ObservedStream(trace, real, 1))

        return call

    wrap("order_staging", staging)

    def consumption(original):
        def call(layer, stream):
            if not trace.ownership_active:
                return original(layer, stream)
            slot = layer % len(uses)
            trace.set_owner(
                **base,
                layer=layer if trace.current_fields.get("layer", -1) >= 0 else -1,
                slot=slot,
                use=uses[slot],
            )
            with trace.scope("slot_consume"):
                return original(layer, _ObservedStream(trace, stream, 3))

        return call

    wrap("consume", consumption)

    def association(original, operation):
        def call(layer, event):
            slot = layer % len(uses)
            if not trace.ownership_active:
                return original(layer, event)
            event_id, record = trace.event_identity(event)
            with trace.scope(
                operation,
                **base,
                layer=layer if trace.current_fields.get("layer", -1) >= 0 else -1,
                slot=slot,
                use=uses[slot],
                event=event_id,
                record=record,
            ):
                return original(layer, event)

        return call

    for name in ("finish_gather", "finish_consumer"):
        wrap(name, lambda original, name=name: association(original, name))
    original_synchronize = state._synchronize

    def synchronize():
        if not trace.ownership_active:
            return original_synchronize()
        state._check_finished()
        for slot in state._slots:
            if slot is not None:
                for event in (slot.writer, slot.consumer):
                    if event is not None:
                        event_id, record = trace.event_identity(event)
                        with trace.scope(
                            "event_synchronize", **base, event=event_id, record=record
                        ):
                            event.synchronize()

    state._synchronize = synchronize

    for name in ("begin_step", "reset_caches", "register_cache", "close"):

        def lifecycle(original, operation=name):
            def call(*args, **kwargs):
                if not trace.ownership_active:
                    return original(*args, **kwargs)
                with trace.scope(operation, **base):
                    result = original(*args, **kwargs)
                    with trace.scope(operation + "_complete", **base):
                        pass
                    return result

            return call

        wrap(name, lifecycle)


def install_backend_trace(impl, trace: PrefillTrace) -> None:
    """Install eager-prefill wrappers only in an explicitly diagnosed worker."""
    impl._prefill_trace = trace
    original_consume = impl._consume_ckv

    def consume(cache, metadata, layer, original_cache):
        if not trace.ownership_active:
            if trace.active:
                with trace.scope("dispatch", route=2):
                    return original_consume(cache, metadata, layer, original_cache)
            return original_consume(cache, metadata, layer, original_cache)
        registry = impl._ckv_reservation.registry
        if not getattr(registry, "_prefill_trace_installed", False):
            registry._prefill_trace_installed = True
            original_workspace = registry.for_workspace

            def workspace(*args, **kwargs):
                state = original_workspace(*args, **kwargs)
                install_state_trace(state, trace)
                return state

            registry.for_workspace = workspace
        with trace.scope("dispatch", route=2):
            result = original_consume(cache, metadata, layer, original_cache)
        _, state, layer_idx = result
        slot = layer_idx % state.pool.plan.ring_slots
        trace.set_owner(
            **state._prefill_trace_identity,
            slot=slot,
            use=state._prefill_trace_uses[slot],
        )
        return result

    impl._consume_ckv = consume
    original_queue = impl._queue_ckv_gather

    def queue(state, layer_idx, cache, metadata, stream, producer, *, asynchronous):
        if not trace.ownership_active:
            return original_queue(
                state,
                layer_idx,
                cache,
                metadata,
                stream,
                producer,
                asynchronous=asynchronous,
            )
        source_layer = trace.current_fields.get("layer", -1)
        slot = layer_idx % state.pool.plan.ring_slots
        use = state._prefill_trace_uses[slot] + 1
        with trace.scope(
            "history_gather" if asynchronous else "full_gather",
            **state._prefill_trace_identity,
            layer=layer_idx if source_layer >= 0 else -1,
            source_layer=source_layer,
            slot=slot,
            use=use,
            stream=trace.stream_identity(stream),
            prefetched=int(asynchronous),
        ):
            return original_queue(
                state,
                layer_idx,
                cache,
                metadata,
                stream,
                producer,
                asynchronous=asynchronous,
            )

    impl._queue_ckv_gather = queue
    original_bind = getattr(impl, "_bind", None)
    if callable(original_bind):

        def bind(plan, *args, **kwargs):
            if not trace.active or trace.ownership_active:
                return original_bind(plan, *args, **kwargs)
            kind = next(
                (
                    index
                    for index, name in enumerate(
                        ("_decode_plan", "_extend_plan", "_ckv_extend_plan")
                    )
                    if plan is getattr(impl, name, None)
                ),
                -1,
            )
            with trace.scope("attention_dispatch", plan=kind):
                return original_bind(plan, *args, **kwargs)

        impl._bind = bind

    original_run = impl._run

    def run(*args, **kwargs):
        if not trace.ownership_active:
            return original_run(*args, **kwargs)
        with trace.scope("attention_read"):
            return original_run(*args, **kwargs)

    impl._run = run

    original_gather = impl._gather_full_ckv

    def gather(*args, **kwargs):
        if not trace.ownership_active:
            return original_gather(*args, **kwargs)
        from vllm.distributed.parallel_state import get_dcp_group

        group = kwargs.get("group") or get_dcp_group()
        with trace.scope(
            "history_exchange" if kwargs.get("history_only") else "full_exchange",
            **trace.group_fields(group),
        ):
            return original_gather(*args, **kwargs)

    impl._gather_full_ckv = gather

    original_update = impl.do_kv_cache_update
    impl.do_kv_cache_update = trace.wrap("cache_produce", original_update, producer=1)

    original_warmup = impl._warmup_ckv_collectives

    def warmup():
        with trace.startup():
            return original_warmup()

    impl._warmup_ckv_collectives = warmup
