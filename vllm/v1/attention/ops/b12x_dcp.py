# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Graph-owned four-rank DCP exchange for GLM sparse attention.

Channels and output buffers are allocated at model load. Target verification,
first draft, and later draft graphs have independent channels, including separate
owners for temporary memory-profiling graphs. Per-layer calls only use the active
capture binding; eager and unsupported paths retain the generic DCP manager.
"""

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

logger = init_logger(__name__)
MAX_ROWS = 16
ROLES = ("target", "draft_prefill", "draft_decode")
_active: ContextVar[Any] = ContextVar("b12x_dcp_capture_binding", default=None)


def active_dcp_transport():
    """Return the graph's preallocated transport, or None for generic dispatch."""
    return _active.get()


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

    def accepts(self, query: torch.Tensor, local_seq_lens: torch.Tensor | None) -> bool:
        return (
            query.dtype == torch.bfloat16
            and query.shape[1:] == (8, 576)
            and 1 <= query.shape[0] <= MAX_ROWS
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
        if packed.shape[1:] != (2048, 2) or not 1 <= packed.shape[0] <= MAX_ROWS:
            return False
        self.candidates.merge(packed, indices)
        return True


class B12XDCPTransport:
    """Own a fixed catalog of graph channels and serialized role workspaces.

    A graph key is its execution role, profiling/serving purpose and planned
    row capacity. Live row counts never select a compiled specialization. Each
    role's output buffers may be shared because the v2 runner replays on one
    compute stream. Graphs from this catalog must not replay concurrently.
    """

    def __init__(self, group, device, attention_factory, candidate_factory):
        self._channels = {}
        self._used = set()
        self._logged = False
        self.allocated_bytes = 0
        self._role_buffers = {}
        try:
            for role in ROLES:
                buffers = _Buffers(
                    torch.empty(
                        (MAX_ROWS, 32, 576), dtype=torch.bfloat16, device=device
                    ),
                    torch.empty(
                        (MAX_ROWS, 8, 512), dtype=torch.bfloat16, device=device
                    ),
                    torch.empty((MAX_ROWS, 32), dtype=torch.float32, device=device),
                )
                self._role_buffers[role] = buffers
                self.allocated_bytes += sum(
                    t.numel() * t.element_size() for t in vars(buffers).values()
                )
                for profiling in (True, False):
                    purpose = "profile" if profiling else "serving"
                    for capacity in range(1, MAX_ROWS + 1):
                        key = (role, profiling, capacity)
                        attention = attention_factory(
                            process_group=group.cpu_group,
                            device=device,
                            channel_id=f"{role}:{purpose}:capacity-{capacity}",
                            max_rows=MAX_ROWS,
                        )
                        try:
                            candidates = candidate_factory(
                                process_group=group.cpu_group,
                                device=device,
                                max_rows=MAX_ROWS,
                                topk=2048,
                            )
                        except Exception:
                            attention.close()
                            raise
                        self._channels[key] = _GraphTransport(
                            attention, candidates, buffers
                        )
                        self.allocated_bytes += (
                            attention.allocated_bytes + candidates.slab_bytes
                        )
        except Exception:
            self.close()
            raise
        logger.info(
            "B12X DCP memory budget component: channels=%d, persistent_bytes=%d "
            "(included in persistent_total)",
            len(self._channels),
            self.allocated_bytes,
        )

    def bind_graph_manager(self, manager, role: str, *, profiling: bool):
        if role not in ROLES:
            raise ValueError(f"Unknown DCP graph role: {role}")
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
        capacity = desc.num_reqs if role == "draft_prefill" else desc.num_tokens
        key = (role, profiling, capacity)
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
                "query gather, LSE reduce-scatter, candidate publication"
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
    return B12XDCPTransport(group, device, PCIeDCPAttention, PCIeDCPTopKPull)
