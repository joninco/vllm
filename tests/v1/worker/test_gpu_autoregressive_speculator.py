# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.models import supports_multimodal_embeddings
from vllm.model_executor.models.exaone4_5_mtp import Exaone4_5_MTP
from vllm.model_executor.models.llama4_eagle import EagleLlama4ForCausalLM
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.model_executor.models.mistral_eagle import EagleMistralForCausalLM
from vllm.model_executor.models.mistral_large_3_eagle import (
    EagleMistralLarge3ForCausalLM,
)
from vllm.v1.worker.gpu.spec_decode import speculator as base_spec_module
from vllm.v1.worker.gpu.spec_decode.autoregressive import speculator as spec_module
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    AutoRegressiveSpeculator,
)
from vllm.v1.worker.gpu.spec_decode.multi_module_mtp.speculator import (
    MultiModuleMTPSpeculator,
)
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator


class _TestSpeculator(AutoRegressiveSpeculator):
    def load_draft_model(self, target_model, target_attn_layer_names):
        return self.test_draft_model


class _DraftModel(torch.nn.Module):
    def __init__(self, output: torch.Tensor | tuple[torch.Tensor, torch.Tensor]):
        super().__init__()
        self.output = output

    def forward(self, **kwargs):
        return self.output


class _MultimodalDraftModel(torch.nn.Module):
    supports_multimodal_embeddings = True

    def embed_input_ids(
        self,
        input_ids,
        multimodal_embeddings=None,
        *,
        is_multimodal=None,
    ):
        raise AssertionError("embed_input_ids should not be called during loading")


class _TextOnlyDraftModel(torch.nn.Module):
    def embed_input_ids(
        self,
        input_ids,
        multimodal_embeddings=None,
        *,
        is_multimodal=None,
    ):
        raise AssertionError("embed_input_ids should not be called during loading")


def _mock_base_model_load(monkeypatch):
    monkeypatch.setattr(
        base_spec_module,
        "get_layers_from_vllm_config",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        DraftModelSpeculator,
        "_validate_local_argmax_reduction",
        lambda self: None,
    )


def _make_speculator(
    monkeypatch,
    output: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
) -> _TestSpeculator:
    monkeypatch.setattr(
        spec_module,
        "set_forward_context",
        lambda *args, **kwargs: nullcontext(),
    )

    speculator = object.__new__(_TestSpeculator)
    speculator.supports_mm_inputs = False
    speculator.vllm_config = None
    speculator.input_buffers = SimpleNamespace(
        input_ids=torch.arange(4),
        positions=torch.arange(4),
    )
    speculator.hidden_states = torch.zeros(4, 3)
    speculator.model = _DraftModel(output)
    return speculator


def test_mm_support_configured_after_model_load(monkeypatch):
    target_model_config = object()
    draft_model_config = object()
    vllm_config = SimpleNamespace(model_config=target_model_config)
    draft_model = _MultimodalDraftModel()

    def init_base(speculator, vllm_config, device):
        speculator.vllm_config = vllm_config
        speculator.device = device
        speculator.max_num_tokens = 4
        speculator.max_num_reqs = 2
        speculator.hidden_size = 3
        speculator.dtype = torch.float32
        speculator.draft_model_config = draft_model_config
        speculator.supports_mm_inputs = False

    checked_configs = []

    def supports_multimodal_inputs(model_config):
        checked_configs.append(model_config)
        return True

    monkeypatch.setattr(DraftModelSpeculator, "__init__", init_base)
    _mock_base_model_load(monkeypatch)
    monkeypatch.setattr(
        base_spec_module.MULTIMODAL_REGISTRY,
        "supports_multimodal_inputs",
        supports_multimodal_inputs,
    )

    speculator = _TestSpeculator(vllm_config, torch.device("cpu"))

    assert checked_configs == []
    assert not speculator.supports_mm_inputs
    assert speculator.inputs_embeds is None

    speculator.test_draft_model = draft_model
    speculator.load_model(torch.nn.Module())

    assert checked_configs == [target_model_config]
    assert speculator.supports_mm_inputs
    assert speculator.inputs_embeds is not None
    assert speculator.inputs_embeds.shape == (4, 3)


def test_load_model_keeps_mm_support_for_capable_drafter(monkeypatch):
    speculator = object.__new__(_TestSpeculator)
    speculator.supports_mm_inputs = False
    speculator.inputs_embeds = None
    speculator.vllm_config = SimpleNamespace(model_config=object())
    speculator.max_num_tokens = 4
    speculator.hidden_size = 3
    speculator.dtype = torch.float32
    speculator.device = torch.device("cpu")
    draft_model = _MultimodalDraftModel()
    speculator.test_draft_model = draft_model
    _mock_base_model_load(monkeypatch)
    monkeypatch.setattr(
        base_spec_module.MULTIMODAL_REGISTRY,
        "supports_multimodal_inputs",
        lambda model_config: True,
    )

    speculator.load_model(torch.nn.Module())

    assert speculator.supports_mm_inputs
    assert speculator.inputs_embeds is not None


def test_load_model_disables_mm_support_for_text_only_drafter(monkeypatch):
    speculator = object.__new__(_TestSpeculator)
    speculator.supports_mm_inputs = False
    speculator.inputs_embeds = None
    speculator.vllm_config = SimpleNamespace(model_config=object())
    draft_model = _TextOnlyDraftModel()
    speculator.test_draft_model = draft_model
    warning_messages = []
    _mock_base_model_load(monkeypatch)
    monkeypatch.setattr(
        base_spec_module.MULTIMODAL_REGISTRY,
        "supports_multimodal_inputs",
        lambda model_config: True,
    )
    monkeypatch.setattr(
        base_spec_module.logger,
        "warning_once",
        lambda message, *args: warning_messages.append(message % args),
    )

    speculator.load_model(torch.nn.Module())

    assert not speculator.supports_mm_inputs
    assert warning_messages == [
        "Draft model _TextOnlyDraftModel does not support external multimodal "
        "embeddings. Embeddings from the target model will not be passed to the "
        "drafter; using text-only draft inputs instead."
    ]


def test_multi_module_mm_support_configured_after_model_load(monkeypatch):
    speculator = object.__new__(MultiModuleMTPSpeculator)
    speculator.supports_mm_inputs = False
    speculator.inputs_embeds = None
    speculator.cached_draft_input_embeds = None
    speculator.vllm_config = SimpleNamespace(model_config=object())
    speculator.max_num_tokens = 4
    speculator.max_num_reqs = 2
    speculator.num_speculative_steps = 3
    speculator.hidden_size = 3
    speculator.dtype = torch.float32
    speculator.device = torch.device("cpu")
    draft_model = _MultimodalDraftModel()
    _mock_base_model_load(monkeypatch)
    monkeypatch.setattr(
        MultiModuleMTPSpeculator,
        "load_draft_model",
        lambda self, target_model, target_attn_layer_names: draft_model,
    )
    monkeypatch.setattr(
        base_spec_module.MULTIMODAL_REGISTRY,
        "supports_multimodal_inputs",
        lambda model_config: True,
    )

    speculator.load_model(torch.nn.Module())

    assert speculator.supports_mm_inputs
    assert speculator.inputs_embeds is not None
    assert speculator.inputs_embeds.shape == (4, 3)
    assert speculator.cached_draft_input_embeds is not None
    assert speculator.cached_draft_input_embeds.shape == (2, 2, 3)


@pytest.mark.parametrize(
    ("model_cls", "expected"),
    [
        (EagleLlama4ForCausalLM, True),
        (EagleMistralForCausalLM, True),
        (EagleMistralLarge3ForCausalLM, True),
        (Exaone4_5_MTP, True),
        (Eagle3LlamaForCausalLM, False),
    ],
)
def test_draft_model_multimodal_embedding_capability(model_cls, expected):
    assert supports_multimodal_embeddings(model_cls) is expected


def test_run_model_unpacks_tuple_return_for_mtp(monkeypatch):
    logits_hidden = torch.full((4, 3), 1.0)
    feedback_hidden = torch.full((4, 3), 2.0)
    speculator = _make_speculator(monkeypatch, (logits_hidden, feedback_hidden))

    actual_logits_hidden, actual_feedback_hidden = speculator._run_model(
        4,
        attn_metadata=None,
        slot_mappings=None,
        num_tokens_across_dp=None,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )

    assert actual_logits_hidden is logits_hidden
    assert actual_feedback_hidden is feedback_hidden


def test_run_model_reuses_tensor_return_for_mtp(monkeypatch):
    hidden = torch.full((4, 3), 1.0)
    speculator = _make_speculator(monkeypatch, hidden)

    actual_logits_hidden, actual_feedback_hidden = speculator._run_model(
        4,
        attn_metadata=None,
        slot_mappings=None,
        num_tokens_across_dp=None,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )

    assert actual_logits_hidden is hidden
    assert actual_feedback_hidden is hidden


def test_probabilistic_draft_sampler_owns_disjoint_philox_offset(monkeypatch):
    captured = {}

    def fake_gumbel_sample(
        logits,
        idx_mapping,
        temperature,
        seeds,
        positions,
        **kwargs,
    ):
        captured["positions"] = positions
        captured.update(kwargs)
        return torch.zeros(logits.shape[0], dtype=torch.int64)

    monkeypatch.setattr(base_spec_module, "gumbel_sample", fake_gumbel_sample)
    positions = torch.tensor([12, 99], dtype=torch.int64)
    active_rows = torch.tensor(2, dtype=torch.int32)
    speculator = object.__new__(_TestSpeculator)
    speculator.use_fp64_gumbel = False

    speculator._sample_probabilistic_draft(
        logits=torch.zeros(2, 5),
        positions=positions,
        idx_mapping=torch.arange(2),
        temperature=torch.ones(2),
        seeds=torch.tensor([7, 11], dtype=torch.int64),
        draft_step=torch.tensor(0, dtype=torch.int64),
        draft_logits=torch.empty(2, 5),
        active_rows=active_rows,
    )

    torch.testing.assert_close(
        captured["positions"], positions + 1 + (1 << 30), rtol=0, atol=0
    )
    assert captured["apply_temperature"] is True
    assert captured["logits_cache_active_rows"] is active_rows


def test_ar_probabilistic_draft_uses_shared_sampler(monkeypatch):
    captured = {}

    def fake_sample_probabilistic_draft(
        self,
        logits,
        positions,
        idx_mapping,
        temperature,
        seeds,
        draft_step,
        draft_logits,
        active_rows=None,
    ):
        captured["positions"] = positions
        captured["active_rows"] = active_rows
        return torch.zeros(logits.shape[0], dtype=torch.int64)

    monkeypatch.setattr(
        DraftModelSpeculator,
        "_sample_probabilistic_draft",
        fake_sample_probabilistic_draft,
    )
    speculator = object.__new__(_TestSpeculator)
    speculator.model = SimpleNamespace(
        compute_logits=lambda hidden_states: torch.zeros(hidden_states.shape[0], 5)
    )
    speculator.active_num_reqs = torch.tensor(2, dtype=torch.int32)
    speculator.use_fp64_gumbel = False
    positions = torch.tensor([12, 99], dtype=torch.int64)

    speculator.sample_draft(
        hidden_states=torch.zeros(2, 3),
        positions=positions,
        idx_mapping=torch.arange(2),
        temperature=torch.ones(2),
        seeds=torch.tensor([7, 11], dtype=torch.int64),
        draft_step=torch.tensor(0, dtype=torch.int64),
        draft_logits=torch.empty(2, 5),
    )

    assert captured["positions"] is positions
    assert captured["active_rows"] is speculator.active_num_reqs


def test_autoregressive_capture_ids_are_deterministic_and_phase_separated():
    def capture_channel_ids() -> list[str]:
        events = []

        class FakeManager:
            use_breakable_cg = False

            def capture(self, *args, **kwargs):
                events.append(kwargs["channel_id"])

        speculator = object.__new__(_TestSpeculator)
        speculator.last_token_indices = torch.zeros(1)
        speculator.prefill_cudagraph_manager = FakeManager()
        speculator.decode_cudagraph_manager = FakeManager()
        speculator.model = object()
        speculator._prefill = object()
        speculator._generate_draft = object()
        speculator.model_state = object()
        speculator.target_input_buffers = object()
        speculator.input_buffers = object()
        speculator.block_tables = object()
        speculator.target_attn_groups = object()
        speculator.attn_groups = object()
        speculator.kv_cache_config = object()
        speculator.num_speculative_steps = 3
        speculator.max_num_reqs = 1

        speculator.capture(capture_phase="profile")
        speculator.capture(capture_phase="production")
        return events

    expected = [
        "vllm:draft:prefill:profile",
        "vllm:draft:decode:profile",
        "vllm:draft:prefill:production",
        "vllm:draft:decode:production",
    ]
    assert capture_channel_ids() == capture_channel_ids() == expected


def test_multi_module_mtp_capture_uses_phase_specific_channel():
    events = []

    class FakeManager:
        use_breakable_cg = False

        def capture(self, *args, **kwargs):
            events.append(kwargs["channel_id"])

    speculator = object.__new__(MultiModuleMTPSpeculator)
    speculator.last_token_indices = torch.zeros(1)
    speculator.cudagraph_manager = FakeManager()
    speculator.model = object()
    speculator._generate_drafts = object()
    speculator.model_state = object()
    speculator.input_buffers = object()
    speculator.block_tables = object()
    speculator.attn_groups = object()
    speculator.kv_cache_config = object()

    speculator.capture(capture_phase="profile")
    speculator.capture(capture_phase="production")

    assert events == [
        "vllm:draft:multi-module-mtp:profile",
        "vllm:draft:multi-module-mtp:production",
    ]


def test_autoregressive_graph_channel_is_bound_at_capture_not_construction(
    monkeypatch,
):
    created = []

    class FakeManager:
        def __init__(self, vllm_config, device, cudagraph_mode, decode_query_len):
            created.append((vllm_config, device, cudagraph_mode, decode_query_len))

    monkeypatch.setattr(spec_module, "SpeculatorCudaGraphManager", FakeManager)
    speculator = object.__new__(_TestSpeculator)
    speculator.vllm_config = object()
    speculator.device = torch.device("cpu")
    speculator.num_speculative_steps = 3

    AutoRegressiveSpeculator.init_cudagraph_manager(
        speculator,
        CUDAGraphMode.FULL,
    )

    assert created == [
        (
            speculator.vllm_config,
            speculator.device,
            CUDAGraphMode.FULL,
            4,
        ),
        (
            speculator.vllm_config,
            speculator.device,
            CUDAGraphMode.FULL_DECODE_ONLY,
            1,
        ),
    ]
