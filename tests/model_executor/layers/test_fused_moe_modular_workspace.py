# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation


class _FakeWorkspaceManager:
    def __init__(self) -> None:
        self.specs: tuple[tuple[tuple[int, ...], torch.dtype], ...] | None = None

    def get_simultaneous(
        self, *shapes_and_dtypes: tuple[tuple[int, ...], torch.dtype]
    ) -> list[torch.Tensor]:
        self.specs = shapes_and_dtypes
        return [
            torch.empty(shape, dtype=dtype) for shape, dtype in shapes_and_dtypes
        ]


class _FakeExperts:
    def __init__(
        self,
        *,
        workspace13_shape: tuple[int, ...] = (0,),
        workspace2_shape: tuple[int, ...] = (17,),
        output_shape: tuple[int, ...] = (4, 8),
    ) -> None:
        self.workspace13_shape = workspace13_shape
        self.workspace2_shape = workspace2_shape
        self.output_shape = output_shape

    def workspace_dtype(self, out_dtype: torch.dtype) -> torch.dtype:
        return out_dtype

    def workspace_shapes(self, *_args, **_kwargs):
        return self.workspace13_shape, self.workspace2_shape, self.output_shape


def _make_impl(experts: _FakeExperts) -> mk.FusedMoEKernelModularImpl:
    impl = object.__new__(mk.FusedMoEKernelModularImpl)
    impl.fused_experts = experts
    return impl


def test_fused_moe_output_alias_is_not_used_as_intermediate(monkeypatch):
    workspace_manager = _FakeWorkspaceManager()
    monkeypatch.setattr(
        mk, "current_workspace_manager", lambda: workspace_manager
    )
    output_alias = torch.empty((4, 8), dtype=torch.float16)

    workspace13, workspace2, fused_out = _make_impl(_FakeExperts())._allocate_buffers(
        torch.float16,
        output_alias.device,
        M_chunk=4,
        M_full=4,
        N=16,
        K=8,
        top_k=1,
        global_num_experts=1,
        local_num_experts=1,
        expert_tokens_meta=None,
        activation=MoEActivation.SILU,
    )

    assert workspace13.shape == (0,)
    assert workspace2.shape == (17,)
    assert fused_out is not output_alias
    assert fused_out.shape == output_alias.shape
    assert workspace_manager.specs == (
        ((32,), torch.float16),
        ((17,), torch.float16),
    )


def test_non_alias_fused_moe_output_is_reserved_as_intermediate(monkeypatch):
    workspace_manager = _FakeWorkspaceManager()
    monkeypatch.setattr(
        mk, "current_workspace_manager", lambda: workspace_manager
    )
    output_alias = torch.empty((4, 8), dtype=torch.float16)

    _, _, fused_out = _make_impl(
        _FakeExperts(workspace13_shape=(5,))
    )._allocate_buffers(
        torch.float16,
        output_alias.device,
        M_chunk=4,
        M_full=4,
        N=16,
        K=8,
        top_k=1,
        global_num_experts=1,
        local_num_experts=1,
        expert_tokens_meta=None,
        activation=MoEActivation.SILU,
    )

    assert fused_out is not output_alias
    assert fused_out.shape == output_alias.shape
    assert workspace_manager.specs == (
        ((32,), torch.float16),
        ((17,), torch.float16),
    )
