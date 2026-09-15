# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Graph-owned four-rank DCP exchange for GLM sparse attention.

Channels and output buffers are allocated at model load, one channel per
captured full decode graph that exchanges rows through the transport. A
channel's row capacity equals the token count of its graph, and the largest
capacity is the launch's transport capacity: the sequence limit times the
tokens each sequence contributes per step, clipped to the CUDA graph capture
size. Target verification, first draft, and later draft graphs have independent
channels, including separate owners for temporary memory-profiling graphs.
Per-layer calls only use the active capture binding; eager and unsupported
paths retain the generic DCP manager.
"""

from collections.abc import Iterable, Mapping
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from vllm import envs
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed import get_dcp_group
from vllm.logger import init_logger
from vllm.utils.math_utils import round_up

logger = init_logger(__name__)
ROLES = ("target", "draft_prefill", "draft_decode")
_active: ContextVar[Any] = ContextVar("b12x_dcp_capture_binding", default=None)


def active_dcp_transport():
    """Return the graph's preallocated transport, or None for generic dispatch."""
    return _active.get()


def uniform_decode_row_counts(
    capture_sizes: Iterable[int],
    *,
    tokens_per_request: int,
    max_num_reqs: int,
    max_capture_size: int,
    request_sizes: frozenset[int] | None = None,
) -> tuple[int, ...]:
    """Return the token counts of the uniform full decode graphs a manager captures.

    This mirrors the separate decode routine of ``CudaGraphManager``: every
    capture size rounds up to a multiple of ``tokens_per_request`` and is kept
    when the rounded count fits the request limit and the capture size limit.
    ``request_sizes`` restricts the graphs to the given request counts, as the
    first draft position does with its sparse full-capture request sizes.
    """
    rows: set[int] = set()
    for size in capture_sizes:
        tokens = round_up(int(size), tokens_per_request)
        requests = tokens // tokens_per_request
        if tokens > max_num_reqs * tokens_per_request or tokens > max_capture_size:
            continue
        if request_sizes is not None and requests not in request_sizes:
            continue
        rows.add(tokens)
    return tuple(sorted(rows))


@dataclass(frozen=True)
class TransportPlan:
    """Row counts of the captured decode graphs that exchange through the transport.

    ``capacity`` is the largest row count any channel accepts: the sequence
    limit times the tokens each sequence contributes per step, clipped to the
    CUDA graph capture size. ``rows_by_role`` lists, per graph role, the
    exchanged row counts of its full decode graphs in ascending order; every
    value is at most ``capacity``. Roles without full decode graphs are absent.
    """

    capacity: int
    rows_by_role: Mapping[str, tuple[int, ...]]

    @property
    def channel_count(self) -> int:
        return 2 * sum(len(rows) for rows in self.rows_by_role.values())


def plan_transport(vllm_config) -> TransportPlan:
    """Derive the transport plan from the launch configuration.

    The target role exchanges one row per scheduled token of a uniform decode
    graph: ``num_speculative_tokens + 1`` per request. The first draft position
    runs the same padded token count for the request counts of its sparse
    full-capture set, and later draft positions run one token per request.
    """
    scheduler = vllm_config.scheduler_config
    compilation = vllm_config.compilation_config
    max_num_reqs = int(scheduler.max_num_seqs)
    capture_sizes = tuple(sorted(compilation.cudagraph_capture_sizes or ()))
    max_capture_size = int(compilation.max_cudagraph_capture_size or 0)
    tokens_per_request = int(vllm_config.num_speculative_tokens) + 1
    capacity = min(max_num_reqs * tokens_per_request, max_capture_size)
    rows_by_role: dict[str, tuple[int, ...]] = {}
    target_rows = uniform_decode_row_counts(
        capture_sizes,
        tokens_per_request=tokens_per_request,
        max_num_reqs=max_num_reqs,
        max_capture_size=max_capture_size,
    )
    if target_rows:
        rows_by_role["target"] = target_rows
    if vllm_config.speculative_config is not None:
        from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
            _sparse_full_capture_request_sizes,
        )

        draft_prefill_rows = uniform_decode_row_counts(
            capture_sizes,
            tokens_per_request=tokens_per_request,
            max_num_reqs=max_num_reqs,
            max_capture_size=max_capture_size,
            request_sizes=_sparse_full_capture_request_sizes(max_num_reqs),
        )
        if draft_prefill_rows:
            rows_by_role["draft_prefill"] = draft_prefill_rows
        draft_decode_rows = uniform_decode_row_counts(
            capture_sizes,
            tokens_per_request=1,
            max_num_reqs=max_num_reqs,
            max_capture_size=max_capture_size,
        )
        if draft_decode_rows:
            rows_by_role["draft_decode"] = draft_decode_rows
    for role, rows in rows_by_role.items():
        if rows[-1] > capacity:
            raise ValueError(
                f"DCP graph role {role} exchanges {rows[-1]} rows above the "
                f"transport capacity {capacity}"
            )
    return TransportPlan(capacity=capacity, rows_by_role=rows_by_role)


@dataclass
class _Buffers:
    query: torch.Tensor
    output: torch.Tensor
    masked_lse: torch.Tensor


@dataclass
class _GraphTransport:
    attention: Any
    candidates: Any
    buffers: _Buffers
    max_rows: int

    def accepts(self, query: torch.Tensor, local_seq_lens: torch.Tensor | None) -> bool:
        return (
            query.dtype == torch.bfloat16
            and query.shape[1:] == (8, 576)
            and 1 <= query.shape[0] <= self.max_rows
            and query.device == self.buffers.query.device
            and query.is_contiguous()
            and local_seq_lens is not None
            and local_seq_lens.shape == query.shape[:1]
            and local_seq_lens.dtype == torch.int32
            and local_seq_lens.device == query.device
            and local_seq_lens.is_contiguous()
        )

    def query(self, query: torch.Tensor) -> torch.Tensor:
        out = self.buffers.query[: query.shape[0]]
        self.attention.query(query, out)
        return out

    def combine(
        self, partial: torch.Tensor, lse: torch.Tensor, local_seq_lens: torch.Tensor
    ) -> torch.Tensor:
        rows = partial.shape[0]
        out = self.buffers.output[:rows]
        self.attention.combine_masked(
            partial, lse, local_seq_lens, self.buffers.masked_lse[:rows], out
        )
        return out

    def merge_candidates(self, packed: torch.Tensor, indices: torch.Tensor) -> bool:
        if packed.shape[1:] != (2048, 2) or not 1 <= packed.shape[0] <= self.max_rows:
            return False
        self.candidates.merge(packed, indices)
        return True


class B12XDCPTransport:
    """Own one graph channel per captured full decode graph and role workspaces.

    A graph key is its execution role, profiling/serving purpose and exchanged
    row count, which is also the channel's row capacity. Live row counts never
    select a compiled specialization. Each role's output buffers may be shared
    because the v2 runner replays on one compute stream. Graphs from this
    catalog must not replay concurrently.
    """

    def __init__(
        self, group, device, attention_factory, candidate_factory, plan: TransportPlan
    ):
        self.plan = plan
        self._channels: dict[tuple[str, bool, int], _GraphTransport] = {}
        self._used: set[tuple[str, bool, int]] = set()
        self._logged = False
        self.allocated_bytes = 0
        self.role_bytes: dict[str, int] = {}
        self._role_buffers: dict[str, _Buffers] = {}
        try:
            for role, rows_list in plan.rows_by_role.items():
                role_rows = rows_list[-1]
                buffers = _Buffers(
                    torch.empty(
                        (role_rows, 32, 576), dtype=torch.bfloat16, device=device
                    ),
                    torch.empty(
                        (role_rows, 8, 512), dtype=torch.bfloat16, device=device
                    ),
                    torch.empty((role_rows, 32), dtype=torch.float32, device=device),
                )
                self._role_buffers[role] = buffers
                role_bytes = sum(
                    t.numel() * t.element_size() for t in vars(buffers).values()
                )
                for profiling in (True, False):
                    purpose = "profile" if profiling else "serving"
                    for rows in rows_list:
                        key = (role, profiling, rows)
                        attention = attention_factory(
                            process_group=group.cpu_group,
                            device=device,
                            channel_id=f"{role}:{purpose}:rows-{rows}",
                            max_rows=rows,
                        )
                        try:
                            candidates = candidate_factory(
                                process_group=group.cpu_group,
                                device=device,
                                max_rows=rows,
                                topk=2048,
                            )
                        except Exception:
                            attention.close()
                            raise
                        self._channels[key] = _GraphTransport(
                            attention, candidates, buffers, rows
                        )
                        role_bytes += attention.allocated_bytes + candidates.slab_bytes
                self.role_bytes[role] = role_bytes
                self.allocated_bytes += role_bytes
        except Exception:
            self.close()
            raise
        logger.info(
            "B12X DCP memory budget component: capacity=%d rows, channels=%d, "
            "persistent_bytes=%d (included in persistent_total)",
            plan.capacity,
            len(self._channels),
            self.allocated_bytes,
        )
        for role, rows_list in plan.rows_by_role.items():
            logger.info(
                "B12X DCP transport role %s: graph rows %s, persistent_bytes=%d",
                role,
                list(rows_list),
                self.role_bytes[role],
            )

    def bind_graph_manager(self, manager, role: str, *, profiling: bool):
        """Attach the capture scope of a graph manager and check its graph set.

        Every uniform full decode graph the manager captures must own a channel
        of matching row count; a mismatch means the launch configuration was
        planned differently from the manager and is reported before capture.
        """
        if role not in ROLES:
            raise ValueError(f"Unknown DCP graph role: {role}")
        planned = self.plan.rows_by_role.get(role, ())
        captured = manager.uniform_full_decode_token_counts()
        if tuple(captured) != tuple(planned):
            raise RuntimeError(
                f"B12X DCP transport planned rows {list(planned)} for role {role}, "
                f"but its manager captures uniform full decode graphs at "
                f"{list(captured)} tokens"
            )
        manager.b12x_dcp_capture_scope = lambda desc: self.capture_scope(
            role, profiling, desc
        )

    @contextmanager
    def capture_scope(self, role: str, profiling: bool, desc):
        if (
            desc.cg_mode != CUDAGraphMode.FULL
            or desc.uniform_token_count is None
            or desc.num_active_loras
        ):
            yield
            return
        key = (role, profiling, desc.num_tokens)
        binding = self._channels.get(key)
        if binding is None:
            yield
            return
        if key in self._used:
            raise RuntimeError(
                f"DCP channel already belongs to a captured graph: {key}"
            )
        self._used.add(key)
        if not self._logged:
            logger.info(
                "Using B12X peer-pull DCP transport: DCP size=4; "
                "query gather, LSE reduce-scatter, candidate publication; "
                "row capacity %d",
                self.plan.capacity,
            )
            self._logged = True
        with ExitStack() as stack:
            stack.enter_context(binding.attention.capture())
            stack.enter_context(binding.candidates.capture())
            token = _active.set(binding)
            try:
                yield
            finally:
                _active.reset(token)

    def close(self):
        """Close collectively after all graphs using this catalog are destroyed."""
        for binding in reversed(list(self._channels.values())):
            binding.candidates.close()
            binding.attention.close()
        self._channels.clear()
        self._role_buffers.clear()
        self.role_bytes.clear()
        self.allocated_bytes = 0


def initialize_b12x_dcp_transport(vllm_config, device):
    """Register eligible v2 GLM transport storage before worker memory profiling."""
    parallel = vllm_config.parallel_config
    if parallel.decode_context_parallel_size != 4:
        return None
    if (
        parallel.dcp_comm_backend != "a2a"
        or envs.VLLM_USE_DIRECT_DCP_A2A is False
        or parallel.tensor_parallel_size != 8
        or parallel.pipeline_parallel_size != 1
        or parallel.prefill_context_parallel_size != 1
        or parallel.data_parallel_size != 1
        or parallel.enable_dbo
        or vllm_config.lora_config is not None
        or vllm_config.model_config.dtype != torch.bfloat16
        or vllm_config.model_config.architecture != "GlmMoeDsaForCausalLM"
        or not vllm_config.compilation_config.cudagraph_mode.separate_routine()
    ):
        return None
    speculative = vllm_config.speculative_config
    if speculative is not None and (
        speculative.method != "mtp"
        or speculative.uses_acceptance_length_adaptation()
        or speculative.uses_batch_size_dynamic_speculative_decoding()
    ):
        return None
    from vllm.v1.attention.backends.mla.b12x_mla_sparse import B12xMLASparseImpl

    implementations: list[B12xMLASparseImpl] = []
    for layer in vllm_config.compilation_config.static_forward_context.values():
        impl = getattr(layer, "impl", None)
        if isinstance(impl, B12xMLASparseImpl):
            implementations.append(impl)
    if not implementations or any(
        (impl.num_heads, impl._q_head_dim, impl.kv_lora_rank, impl._topk_tokens)
        != (8, 576, 512, 2048)
        for impl in implementations
    ):
        return None
    plan = plan_transport(vllm_config)
    if "target" not in plan.rows_by_role:
        logger.info(
            "B12X DCP decode transport unused: no uniform full decode graphs "
            "within the capture sizes; retaining generic DCP dispatch"
        )
        return None
    group = get_dcp_group()
    try:
        from b12x.comm.pcie.pcie_dcp_attention import PCIeDCPAttention
        from b12x.comm.pcie.pcie_dcp_topk_pull import PCIeDCPTopKPull

        available = True
    except ImportError:
        available = False
    availability = [None] * group.world_size
    dist.all_gather_object(availability, available, group=group.cpu_group)
    if not all(availability):
        logger.info(
            "B12X DCP decode transport unavailable; retaining generic DCP dispatch"
        )
        return None
    return B12XDCPTransport(group, device, PCIeDCPAttention, PCIeDCPTopKPull, plan)
