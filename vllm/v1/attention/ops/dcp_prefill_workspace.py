# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded intra-layer storage for projected eager DCP prefill.

Query and attention scratch occupy the same prefix used by the backend.
All extra regions are reserved before KV admission. Borrowed mode reuses dead
query and attention regions; separate mode retains dedicated projection buffers.
"""

from collections.abc import Callable
from math import prod

import torch
import torch.distributed as dist

from vllm.v1.worker.workspace import current_workspace_manager

WorkspaceSpec = tuple[tuple[int, ...], torch.dtype]


def _region(tensor: torch.Tensor) -> tuple[int, int]:
    size = 1 + sum((n - 1) * s for n, s in zip(tensor.shape, tensor.stride()))
    return tensor.data_ptr(), tensor.data_ptr() + size * tensor.element_size()


class DCPPrefillWorkspace:
    """Plan and borrow query, weight, LSE, projection and RS storage."""

    def __init__(
        self,
        backend_specs: tuple[WorkspaceSpec, WorkspaceSpec],
        *,
        world_size: int,
        latent_dim: int,
        value_dim: int,
        borrowed: bool,
        max_rows: int | None = None,
    ) -> None:
        query_spec, scratch_spec = backend_specs
        shape, dtype = query_spec
        if len(shape) != 3 or world_size <= 1 or shape[1] % world_size:
            raise ValueError("Projected DCP workspace requires divisible query heads")
        if dtype != torch.bfloat16 or scratch_spec[1] != torch.uint8:
            raise ValueError(
                "Projected DCP workspace requires BF16 query and byte scratch"
            )
        if min(*shape, latent_dim, value_dim) <= 0 or latent_dim > shape[2]:
            raise ValueError(
                "Projected DCP workspace dimensions must be positive and fit query"
            )
        self.capacity, self.heads, self.query_dim = shape
        if max_rows is not None:
            if not 0 < max_rows <= self.capacity:
                raise ValueError(
                    "Projected row capacity must fit backend query storage"
                )
            self.capacity = max_rows
        query_spec = ((self.capacity, self.heads, self.query_dim), dtype)
        self.world_size = world_size
        self.local_heads = self.heads // world_size
        self.latent_dim = latent_dim
        self.value_dim = value_dim
        self.borrowed = borrowed
        self.dtype = dtype
        self.backend_specs = backend_specs
        if prod(query_spec[0]) < self.local_heads * latent_dim * value_dim:
            raise ValueError(
                "Query workspace cannot stage local value projection weights"
            )
        if borrowed and prod(scratch_spec[0]) < (
            self.capacity * self.heads * value_dim * dtype.itemsize
        ):
            raise ValueError("Attention scratch cannot hold projected partial output")
        if prod(scratch_spec[0]) < self.heads * self.query_dim * dtype.itemsize:
            raise ValueError("Attention scratch cannot gather one query row")
        extra = (
            ((self.heads, latent_dim, value_dim), dtype),
            ((self.capacity, self.heads), torch.float32),
            ((world_size, self.capacity, self.heads), torch.float32),
            ((self.capacity, self.heads), torch.float32),
        )
        separate = (
            query_spec,
            ((self.heads, self.capacity, latent_dim), dtype),
            ((self.heads, self.capacity, value_dim), dtype),
            ((self.local_heads, self.capacity, value_dim), dtype),
        )
        self.specs = (*backend_specs, *extra, *(separate if not borrowed else ()))

    @property
    def reserved_bytes(self) -> int:
        return sum((prod(s) * d.itemsize + 255) // 256 * 256 for s, d in self.specs)

    def reserve(self) -> None:
        """Reserve every execution slot while the admission profiler owns memory."""
        current_workspace_manager().reserve_all(*self.specs)

    def borrow(
        self,
        rows: int,
        *,
        backend_specs: tuple[WorkspaceSpec, WorkspaceSpec] | None = None,
    ) -> "DCPPrefillBuffers":
        if backend_specs is not None and backend_specs != self.backend_specs:
            raise RuntimeError(
                "Backend workspace geometry changed after prefill reservation"
            )
        if not 0 < rows <= self.capacity:
            raise ValueError("Projected DCP rows exceed reserved capacity")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Projected DCP workspace is eager-only")
        return DCPPrefillBuffers(
            self, rows, current_workspace_manager().get_simultaneous(*self.specs)
        )


class DCPPrefillBuffers:
    """Views valid until another operation borrows the shared workspace."""

    def __init__(
        self, plan: DCPPrefillWorkspace, rows: int, tensors: list[torch.Tensor]
    ):
        self.plan = plan
        self.rows = rows
        (
            self.backend_query,
            self.scratch,
            self.weights,
            local_lse,
            all_lse,
            final_lse,
        ) = tensors[:6]
        self.local_lse = local_lse[:rows]
        self.all_lse = all_lse.view(-1)[: plan.world_size * rows * plan.heads].view(
            plan.world_size, rows, plan.heads
        )
        self.final_lse = final_lse[:rows]
        if plan.borrowed:
            self.query = self.backend_query
            self.projection_input = self.query.view(-1)[
                : plan.heads * rows * plan.latent_dim
            ].view(plan.heads, rows, plan.latent_dim)
            size = plan.heads * rows * plan.value_dim * plan.dtype.itemsize
            self.projected = (
                self.scratch[:size]
                .view(plan.dtype)
                .view(plan.heads, rows, plan.value_dim)
            )
            self.result = self.query.view(-1)[
                : plan.local_heads * rows * plan.value_dim
            ].view(plan.local_heads, rows, plan.value_dim)
        else:
            self.query = tensors[6]
            self.projection_input = (
                tensors[7]
                .view(-1)[: plan.heads * rows * plan.latent_dim]
                .view(plan.heads, rows, plan.latent_dim)
            )
            self.projected = (
                tensors[8]
                .view(-1)[: plan.heads * rows * plan.value_dim]
                .view(plan.heads, rows, plan.value_dim)
            )
            self.result = (
                tensors[9]
                .view(-1)[: plan.local_heads * rows * plan.value_dim]
                .view(plan.local_heads, rows, plan.value_dim)
            )

    def gather_query(
        self, query: torch.Tensor | tuple[torch.Tensor, torch.Tensor], group
    ) -> torch.Tensor:
        """Gather heads through scratch in bounded chunks, with explicit copies."""
        p = self.plan
        parts = query if isinstance(query, tuple) else (query,)
        if (
            sum(part.shape[-1] for part in parts) != p.query_dim
            or any(part.shape[:2] != (self.rows, p.local_heads) for part in parts)
            or any(
                part.dtype != p.dtype or part.device != self.query.device
                for part in parts
            )
        ):
            raise ValueError("Projected DCP query does not match reserved geometry")
        for part in parts:
            if any(stride <= 0 for stride in part.stride()):
                raise ValueError("Projected DCP query requires positive strides")
            begin, end = _region(part)
            for temporary in (self.query, self.scratch):
                other_begin, other_end = _region(temporary)
                if begin < other_end and other_begin < end:
                    raise ValueError("DCP query source overlaps gather workspace")
        if group.world_size != p.world_size:
            raise ValueError("Projected DCP process group has the wrong size")
        chunk_capacity = self.scratch.numel() // (
            p.heads * p.query_dim * p.dtype.itemsize
        )
        for start in range(0, self.rows, chunk_capacity):
            count = min(chunk_capacity, self.rows - start)
            local = (
                self.query.view(-1)
                .narrow(
                    0,
                    start * p.heads * p.query_dim,
                    count * p.local_heads * p.query_dim,
                )
                .view(count, p.local_heads, p.query_dim)
            )
            offset = 0
            for part in parts:
                local[..., offset : offset + part.shape[-1]].copy_(
                    part[start : start + count]
                )
                offset += part.shape[-1]
            size = count * p.heads * p.query_dim * p.dtype.itemsize
            gathered = (
                self.scratch[:size]
                .view(p.dtype)
                .view(p.world_size * count, p.local_heads, p.query_dim)
            )
            dist.all_gather_into_tensor(gathered, local, group=group.device_group)
            ranked = gathered.view(p.world_size, count, p.local_heads, p.query_dim)
            for rank in range(p.world_size):
                self.query[
                    start : start + count,
                    rank * p.local_heads : (rank + 1) * p.local_heads,
                ].copy_(ranked[rank])
        return self.query[: self.rows]

    def project(
        self, output, lse, local_lengths, local_weights, group, prepare_lse: Callable
    ) -> torch.Tensor:
        """Save LSE, gather weights, compact pitched output, then project once."""
        p = self.plan
        if (
            output.shape != (self.rows, p.heads, p.latent_dim)
            or output.dtype != p.dtype
            or output.stride(-1) != 1
            or not (
                (
                    output.stride(0) == p.latent_dim
                    and output.stride(1) >= self.rows * p.latent_dim
                )
                or (
                    output.stride(1) == p.latent_dim
                    and output.stride(0) >= p.heads * p.latent_dim
                )
            )
            or lse.shape != (self.rows, p.heads)
            or lse.dtype != torch.float32
            or local_weights.shape != (p.local_heads, p.latent_dim, p.value_dim)
            or local_weights.dtype != p.dtype
            or any(
                t.device != self.query.device
                for t in (output, lse, local_weights, local_lengths)
            )
        ):
            raise ValueError(
                "Projected DCP attention or weights have incompatible geometry"
            )
        start, end = _region(output)
        scratch_start, scratch_end = _region(self.scratch)
        if not scratch_start <= start < end <= scratch_end:
            raise ValueError("Attention partial output must belong to backend scratch")
        # LSE can share backend scratch with attention output. Save it before
        # projected output overwrites any part of that allocation.
        prepare_lse(lse, local_lengths, self.local_lse)
        staging = self.query.view(-1)[: local_weights.numel()].view(local_weights.shape)
        staging.copy_(local_weights)
        dist.all_gather_into_tensor(self.weights, staging, group=group.device_group)
        self.projection_input.copy_(output.transpose(0, 1))
        torch.bmm(self.projection_input, self.weights, out=self.projected)
        return self.projected.transpose(0, 1)

    def combine(
        self, output, group, correct_output: Callable, *, is_lse_base_on_e: bool
    ) -> torch.Tensor:
        """Correct projected partials and reduce-scatter into reserved output."""
        p = self.plan
        if output.data_ptr() != self.projected.data_ptr() or output.stride() != (
            p.value_dim,
            self.rows * p.value_dim,
            1,
        ):
            raise ValueError(
                "Projected DCP merge requires the reserved head-major view"
            )
        dist.all_gather_into_tensor(
            self.all_lse.flatten(0, 1), self.local_lse, group=group.device_group
        )
        corrected, _ = correct_output(
            output,
            self.all_lse,
            group.rank_in_group,
            None,
            is_lse_base_on_e=is_lse_base_on_e,
            lse_output=self.final_lse,
        )
        if (
            corrected.data_ptr() != self.projected.data_ptr()
            or corrected.stride() != output.stride()
        ):
            raise RuntimeError("DCP correction replaced reserved projected storage")
        dist.reduce_scatter_tensor(
            self.result, corrected.transpose(0, 1), group=group.device_group
        )
        return self.result.transpose(0, 1)
