# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.model_executor.layers.fused_moe.kquant_all_row_capture import (
    AllRowCaptureState,
)


def _unarmed_state() -> AllRowCaptureState:
    state = AllRowCaptureState.__new__(AllRowCaptureState)
    state.rank = 0
    state.batch_metadata = None
    state.prefixes = {0: "model.layers.1.mlp.experts"}
    return state


def test_all_row_request_identity() -> None:
    request_index, document_id = AllRowCaptureState._request_identity(
        "cmpl-qsrtcap-713-0123456789abcdef0123456789abcdef"
    )
    assert request_index == 713
    assert document_id == int.from_bytes(
        bytes.fromhex("0123456789abcdef")[:8], "little", signed=True
    )


def test_all_row_request_identity_rejects_opaque_ids() -> None:
    with pytest.raises(ValueError, match="all-row capture request IDs"):
        AllRowCaptureState._request_identity("cmpl-random")


def test_prepare_batch_binds_prompt_rows_and_excludes_decode() -> None:
    state = AllRowCaptureState.__new__(AllRowCaptureState)
    state.finalized = False
    state.rank = 1
    state.max_tokens = 8
    state.batch_metadata = None
    state.batch_rows = 0
    state.route_ready = torch.ones(92, dtype=torch.bool)
    state.output_ready = torch.ones(92, dtype=torch.bool)
    request = "cmpl-qsrtcap-4-00112233445566778899aabbccddeeff"
    batch = SimpleNamespace(
        num_tokens_after_padding=4,
        req_ids=[request, request],
        query_start_loc_np=np.array([0, 3, 4], dtype=np.int32),
        prefill_len_np=np.array([5, 5], dtype=np.int32),
        num_computed_prefill_tokens_np=np.array([2, 5], dtype=np.int32),
        num_computed_tokens_np=np.array([2, 5], dtype=np.int32),
    )

    with pytest.raises(RuntimeError, match="cannot mix authenticated corpus rows"):
        state.prepare_batch(batch)


def test_prepare_batch_ignores_warmup_requests_without_allocating() -> None:
    state = AllRowCaptureState.__new__(AllRowCaptureState)
    state.finalized = False
    state.rank = 0
    state.max_tokens = 8
    state.batch_metadata = None
    state.batch_rows = 0
    state.route_ready = torch.ones(92, dtype=torch.bool)
    state.output_ready = torch.ones(92, dtype=torch.bool)
    state._start = lambda rows: pytest.fail(
        f"warmup must not start capture writers for {rows} rows"
    )
    batch = SimpleNamespace(
        num_tokens_after_padding=2,
        req_ids=["warmup-request"],
        query_start_loc_np=np.array([0, 2], dtype=np.int32),
        prefill_len_np=np.array([2], dtype=np.int32),
        num_computed_prefill_tokens_np=np.array([0], dtype=np.int32),
        num_computed_tokens_np=np.array([0], dtype=np.int32),
    )

    state.prepare_batch(batch)

    assert state.batch_metadata is None
    assert state.batch_rows == 0


def test_unarmed_capture_ignores_startup_forwards() -> None:
    state = _unarmed_state()
    state.collect_route_input(
        "model.layers.1.mlp.experts",
        torch.empty(1, 3584, dtype=torch.bfloat16),
        torch.empty(1, 16, dtype=torch.float32),
        torch.empty(1, 16, dtype=torch.int32),
    )
    state.collect_routed_latent(
        1,
        torch.empty(1, 3584, dtype=torch.bfloat16),
    )
