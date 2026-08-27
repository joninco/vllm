# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.models.glm5next.model_state import (
    Glm5NextAttnMetadata,
    Glm5NextModelState,
)
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState


def _bare_model_state() -> Glm5NextModelState:
    state = Glm5NextModelState.__new__(Glm5NextModelState)
    state.max_num_reqs = 8
    state.uses_pooled_selector = True
    state.selector_pool_size = 4
    state.selector_state_slot_ids = torch.full((8,), -1, dtype=torch.int32)
    state.selector_state_is_fresh = torch.ones(8, dtype=torch.bool)
    state.selector_num_accepted_tokens = torch.ones(8, dtype=torch.int32)
    state.mamba_num_accepted_tokens = torch.ones(8, dtype=torch.int32)
    state.selector_state_is_fresh_gpu = torch.tensor(
        [True, False, True, True, True, False, True, True]
    )
    state.selector_committed_num_accepted_tokens_gpu = torch.tensor(
        [1, 2, 1, 1, 1, 4, 1, 1], dtype=torch.int32
    )
    state.num_accepted_tokens_gpu = torch.ones(8, dtype=torch.int32)
    state._selector_draft_is_prefilling = torch.zeros(8, dtype=torch.bool)
    state._selector_draft_is_prefilling_gpu = torch.zeros(8, dtype=torch.bool)
    return state


def test_glm5next_metadata_targets_only_selector_builder() -> None:
    metadata = Glm5NextAttnMetadata(
        is_prefilling=torch.zeros(4, dtype=torch.bool),
        selector_state_slot_ids=torch.tensor([5, 1, -1, -1], dtype=torch.int32),
        selector_state_is_fresh=torch.tensor([False, False, True, True]),
        selector_num_accepted_tokens=torch.tensor([4, 2, 1, 1], dtype=torch.int32),
        selector_is_prefilling=torch.tensor([False, True, False, False]),
    )
    selector_builder = SimpleNamespace(requires_glm_next_selector_metadata=True)

    kwargs = metadata.get_extra_attn_kwargs(selector_builder, 2)

    assert set(kwargs) == {
        "selector_state_slot_ids",
        "selector_state_is_fresh",
        "selector_num_accepted_tokens",
        "selector_is_prefilling",
    }
    assert torch.equal(
        kwargs["selector_state_slot_ids"], torch.tensor([5, 1], dtype=torch.int32)
    )
    assert metadata.get_extra_attn_kwargs(SimpleNamespace(), 2) == {}


def test_glm5next_selector_state_tracks_reordering_and_invalidates_padding() -> None:
    state = _bare_model_state()
    first_batch = SimpleNamespace(
        num_reqs=2,
        idx_mapping=torch.tensor([5, 1], dtype=torch.int64),
    )

    slots, fresh, accepted = state._prepare_selector_state(first_batch, 4)
    pointers = slots.data_ptr(), fresh.data_ptr(), accepted.data_ptr()
    assert torch.equal(slots, torch.tensor([5, 1, -1, -1], dtype=torch.int32))
    assert torch.equal(fresh, torch.tensor([False, False, True, True]))
    assert torch.equal(accepted, torch.tensor([4, 2, 1, 1], dtype=torch.int32))

    reordered = SimpleNamespace(
        num_reqs=2,
        idx_mapping=torch.tensor([1, 5], dtype=torch.int64),
    )
    slots, fresh, accepted = state._prepare_selector_state(reordered, 4)
    assert (slots.data_ptr(), fresh.data_ptr(), accepted.data_ptr()) == pointers
    assert torch.equal(slots, torch.tensor([1, 5, -1, -1], dtype=torch.int32))
    assert torch.equal(accepted, torch.tensor([2, 4, 1, 1], dtype=torch.int32))


def test_glm5next_selector_and_mamba_use_independent_acceptance_after_alignment() -> (
    None
):
    state = _bare_model_state()
    state.selector_committed_num_accepted_tokens_gpu[5] = 4
    state.num_accepted_tokens_gpu[5] = 1
    input_batch = SimpleNamespace(
        num_reqs=1,
        idx_mapping=torch.tensor([5], dtype=torch.int32),
    )

    _, _, selector_accepted = state._prepare_selector_state(input_batch, num_reqs=1)
    mamba_accepted = state._prepare_mamba_acceptance(input_batch, num_reqs=1)

    assert selector_accepted.data_ptr() != mamba_accepted.data_ptr()
    assert torch.equal(selector_accepted, torch.tensor([4], dtype=torch.int32))
    assert torch.equal(mamba_accepted, torch.tensor([1], dtype=torch.int32))


def test_glm5next_draft_metadata_preserves_first_step_acceptance() -> None:
    state = _bare_model_state()
    idx_mapping = torch.tensor([5, 1], dtype=torch.int32)

    first = state.prepare_draft_attn_metadata(
        idx_mapping=idx_mapping,
        num_reqs=2,
        num_reqs_padded=4,
        draft_index=1,
    )
    assert first is not None
    assert torch.equal(
        first.selector_state_slot_ids,
        torch.tensor([5, 1, -1, -1], dtype=torch.int32),
    )
    assert torch.equal(
        first.selector_state_is_fresh,
        torch.tensor([False, False, True, True]),
    )
    assert torch.equal(
        first.selector_num_accepted_tokens,
        torch.tensor([4, 2, 1, 1], dtype=torch.int32),
    )

    later = state.prepare_draft_attn_metadata(
        idx_mapping=idx_mapping,
        num_reqs=2,
        num_reqs_padded=4,
        draft_index=2,
    )
    assert later is not None
    assert torch.equal(
        later.selector_num_accepted_tokens,
        torch.ones(4, dtype=torch.int32),
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_glm5next_postprocess_commits_selector_before_mamba_alignment_reset() -> None:
    class ResetAcceptedTokens:
        def run_fused_postprocess_align(
            self,
            num_reqs: int,
            num_accepted_tokens_gpu: torch.Tensor,
            state_idx_gpu: torch.Tensor,
            num_computed_tokens: torch.Tensor,
            idx_mapping: torch.Tensor,
        ) -> None:
            num_accepted_tokens_gpu.fill_(1)

    state = Glm5NextModelState.__new__(Glm5NextModelState)
    state.uses_pooled_selector = True
    state.selector_committed_num_accepted_tokens_gpu = torch.full(
        (5,), 9, dtype=torch.int32, device="cuda"
    )
    state.selector_state_is_fresh_gpu = torch.ones(5, dtype=torch.bool, device="cuda")
    state.num_accepted_tokens_gpu = torch.full(
        (5,), 9, dtype=torch.int32, device="cuda"
    )
    state._align_mode = True
    state._mamba_ctx = ResetAcceptedTokens()
    state._mamba_state_idx_gpu = torch.zeros(5, dtype=torch.int32, device="cuda")
    state.recoverssm = None

    idx_mapping = torch.tensor([3, -1, 1], dtype=torch.int32, device="cuda")
    num_sampled = torch.tensor([4, 2, 3], dtype=torch.int32, device="cuda")
    num_computed_tokens = torch.zeros(5, dtype=torch.int32, device="cuda")

    state.postprocess_state(idx_mapping, num_sampled, num_computed_tokens)

    assert state.num_accepted_tokens_gpu.tolist() == [1, 1, 1, 1, 1]
    assert state.selector_committed_num_accepted_tokens_gpu.tolist() == [
        9,
        3,
        9,
        4,
        9,
    ]
    assert state.selector_state_is_fresh_gpu.tolist() == [
        True,
        False,
        True,
        False,
        True,
    ]


def test_glm5next_recycled_and_rebound_state_is_fresh(monkeypatch) -> None:
    state = _bare_model_state()
    state.selector_state_is_fresh_gpu.fill_(False)
    state.selector_committed_num_accepted_tokens_gpu.fill_(7)
    monkeypatch.setattr(
        MambaHybridModelState,
        "add_request",
        lambda self, req_index, new_req_data: None,
    )
    monkeypatch.setattr(
        MambaHybridModelState,
        "reset_kv_cache_state",
        lambda self: None,
    )

    state.add_request(5, SimpleNamespace(num_computed_tokens=0))

    assert state.selector_state_is_fresh_gpu[5]
    assert state.selector_committed_num_accepted_tokens_gpu[5] == 1
    assert not torch.any(state.selector_state_is_fresh_gpu[:5])
    state.reset_kv_cache_state()
    assert torch.all(state.selector_state_is_fresh_gpu)
    assert torch.all(state.selector_committed_num_accepted_tokens_gpu == 1)


@pytest.mark.parametrize(
    "prefix_length",
    [
        pytest.param(1, id="prefix-match-unit-1"),
        pytest.param(2, id="prefix-match-unit-2"),
        pytest.param(5, id="odd-connector-hit"),
    ],
)
def test_glm5next_rejects_unaligned_fresh_prefix(
    monkeypatch,
    prefix_length: int,
) -> None:
    state = _bare_model_state()
    calls = []
    monkeypatch.setattr(
        MambaHybridModelState,
        "add_request",
        lambda self, req_index, new_req_data: calls.append(req_index),
    )

    with pytest.raises(
        ValueError,
        match=rf"num_computed_tokens={prefix_length}.*divisible by index_kpool=4",
    ):
        state.add_request(
            3,
            SimpleNamespace(num_computed_tokens=prefix_length),
        )

    assert calls == []


def test_glm5next_accepts_pool_aligned_fresh_prefix(monkeypatch) -> None:
    state = _bare_model_state()
    calls = []
    monkeypatch.setattr(
        MambaHybridModelState,
        "add_request",
        lambda self, req_index, new_req_data: calls.append(req_index),
    )

    state.add_request(3, SimpleNamespace(num_computed_tokens=4))

    assert calls == [3]
    assert state.selector_state_is_fresh_gpu[3]
