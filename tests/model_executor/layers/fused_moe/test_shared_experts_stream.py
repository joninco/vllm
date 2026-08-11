# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import torch

import vllm.model_executor.layers.fused_moe.runner.moe_runner as moe_runner_module
import vllm.model_executor.layers.fused_moe.runner.shared_experts as shared_module
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExperts


class _PartialOutputTransform(torch.nn.Module):
    output_is_tp_partial = True

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states


def test_aux_stream_output_lifetime_extends_to_consumer(monkeypatch) -> None:
    shared_experts = object.__new__(SharedExperts)
    aux_stream = Mock()
    consumer_stream = Mock()
    output = Mock()
    shared_experts_input = Mock()
    shared_experts._stream = aux_stream
    shared_experts._layer = Mock(return_value=output)

    @contextmanager
    def use_stream(stream):
        assert stream is aux_stream
        yield

    monkeypatch.setattr(torch.cuda, "stream", use_stream)
    monkeypatch.setattr(shared_module, "current_stream", lambda: consumer_stream)

    result = shared_experts._run_in_aux_stream(shared_experts_input)

    assert result is output
    shared_experts._layer.assert_called_once_with(shared_experts_input)
    consumer_stream.wait_stream.assert_called_once_with(aux_stream)
    output.record_stream.assert_called_once_with(consumer_stream)


def test_tp_partial_output_transform_defers_shared_reduce(monkeypatch) -> None:
    runner = object.__new__(MoERunner)
    torch.nn.Module.__init__(runner)
    runner.routed_output_transform = _PartialOutputTransform()
    runner.routed_input_transform = None
    runner.routed_scaling_factor = 1.0
    runner.router = None
    runner.layer_name = "test"
    runner.moe_config = SimpleNamespace(
        hidden_dim_unpadded=4,
        is_sequence_parallel=False,
        skip_final_all_reduce=False,
        tp_size=2,
        ep_size=1,
    )
    runner.routed_experts = SimpleNamespace(
        quant_method=SimpleNamespace(
            has_unpadded_output=False,
            moe_kernel=SimpleNamespace(output_is_reduced=lambda: True),
        )
    )
    runner._maybe_pad_hidden_states = MethodType(
        lambda self, shared, routed: (routed, None, None),
        runner,
    )

    shared_output = torch.full((2, 4), 2.0)
    fused_output = torch.full((2, 4), 3.0)
    runner._forward_entry = Mock(return_value=(shared_output, fused_output))
    reduced_inputs = []

    def all_reduce(hidden_states: torch.Tensor) -> torch.Tensor:
        reduced_inputs.append(hidden_states.clone())
        return hidden_states * 2

    monkeypatch.setattr(
        moe_runner_module,
        "tensor_model_parallel_all_reduce",
        all_reduce,
    )

    hidden_states = torch.zeros_like(shared_output)
    actual = runner.forward(
        hidden_states,
        router_logits=torch.empty(2, 1),
        shared_experts_input=hidden_states,
    )

    assert len(reduced_inputs) == 1
    torch.testing.assert_close(reduced_inputs[0], shared_output + fused_output)
    torch.testing.assert_close(actual, (shared_output + fused_output) * 2)
