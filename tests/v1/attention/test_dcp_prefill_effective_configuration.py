# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Default environment values select the DCP sparse-MLA prefill dispatch.

With no prefill switch set, a DCP launch gathers native KV records for pure
prefill batches, scores fresh chunks against a step-local index copy, splits
indexer queries across replicas, and routes large mixed batches through
all-gather/reduce-scatter. Decode, capture and MTP verification keep the
configured transport; DCP world size 1 keeps every exchange local. These
checks pin those defaults and the per-class routes they select.
"""

from types import SimpleNamespace

import pytest

import vllm.envs as envs
from vllm.envs import disable_envs_cache
from vllm.v1.attention.backends.mla.b12x_mla_sparse import (
    _round_up_ckv_rank_tokens,
    _use_b12x_full_ckv_gather,
)
from vllm.v1.attention.ops import dcp
from vllm.v1.attention.ops.dcp_prefill_policy import DCPPrefillBatch

PREFILL_SWITCHES = {
    "VLLM_B12X_MLA_CKV_GATHER": True,
    "VLLM_B12X_MLA_CKV_GATHER_MIN_TOKENS": 16,
    "VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS": 524288,
    "VLLM_B12X_MLA_CKV_PREFETCH_DEPTH": 0,
    "VLLM_B12X_MLA_CKV_PREFETCH_WORKSPACE_MIB": 1024,
    "VLLM_B12X_MLA_PREFILL_QUERY_BMM": False,
    "VLLM_B12X_MLA_DCP_GATHER_IN_WORKSPACE": False,
    "VLLM_DCP_QUERY_SPLIT": True,
    "VLLM_DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS": 0,
    "VLLM_DCP_TOPK_OWNER_MERGE": False,
    "VLLM_DCP_INDEXER_LOCAL_CONTEXT": True,
    "VLLM_DCP_INDEXER_SHARDS": 0,
    "VLLM_DCP_REPLICATE_INDEXER_CACHE": False,
    "VLLM_DCP_A2A_MAX_TOKENS": 16,
    "VLLM_DCP_A2A_LARGE_BACKEND": "ag_rs",
    "VLLM_DCP_PROJECT_BEFORE_MERGE": False,
    "VLLM_DCP_PROJECT_BEFORE_MERGE_MIN_PREFILL_TOKENS": 1024,
    "VLLM_DCP_PREFILL_TRACE": False,
}

# Serving geometry of the GLM-5.3 DCP launch: 8,192 batched tokens, 64
# sequences, 64-token pages, cudagraph capture sizes up to 256 tokens.
MAX_TOKENS = 8192
MAX_SEQS = 64
PAGE = 64
CAPTURE_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256]


@pytest.fixture
def default_environment(monkeypatch):
    disable_envs_cache()
    for name in PREFILL_SWITCHES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("B12X_MLA_DCP_GATHER_IN_WORKSPACE", raising=False)


def _policy(world_size):
    manager = dcp.MLADCPManager.__new__(dcp.MLADCPManager)
    manager.group = SimpleNamespace(world_size=world_size)
    manager.use_a2a = True
    manager.is_lse_base_on_e = True
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=MAX_TOKENS),
        compilation_config=SimpleNamespace(cudagraph_capture_sizes=CAPTURE_SIZES),
    )
    return manager.configure_prefill(config)


def test_default_switch_values(default_environment):
    for name, expected in PREFILL_SWITCHES.items():
        assert getattr(envs, name) == expected, name


def test_default_policy_fields(default_environment):
    policy = _policy(4)
    assert policy.enabled
    assert policy.base_backend == "a2a"
    assert policy.a2a_max_tokens == 16
    assert policy.large_backend == "ag_rs"
    assert policy.min_prefill_tokens == 16
    assert policy.max_capture_tokens == 256
    assert policy.max_num_tokens == MAX_TOKENS
    assert not policy.project_before_merge
    assert not policy.borrow_workspace


def _batch(tokens, prefills=1, decodes=0, **flags):
    return DCPPrefillBatch(
        num_tokens=tokens, num_prefills=prefills, num_decodes=decodes, **flags
    )


@pytest.mark.parametrize(
    ("batch", "route"),
    [
        # Decode-only batches use the configured DCP transport.
        (_batch(8, prefills=0, decodes=8), "configured"),
        # Captured graphs and MTP verification never enter the eager routes.
        (_batch(4096, is_capturing=True, full_ckv_eligible=True), "configured"),
        (_batch(4096, is_mtp=True, full_ckv_eligible=True), "configured"),
        # Pure prefill at or below the exclusive 16-token minimum.
        (_batch(16, full_ckv_eligible=True), "configured"),
        # Pure prefill within the CKV storage contract gathers native records.
        (_batch(17, full_ckv_eligible=True), "full_ckv"),
        (_batch(MAX_TOKENS, full_ckv_eligible=True), "full_ckv"),
        # Pure prefill beyond the CKV contract exceeds the 16-token A2A cap.
        (_batch(17), "ag_rs"),
        (_batch(MAX_TOKENS), "ag_rs"),
        # Mixed batches up to the largest captured size keep the transport.
        (_batch(256, prefills=1, decodes=8), "configured"),
        (_batch(256, prefills=1, decodes=8, full_ckv_eligible=True), "configured"),
        # Larger mixed batches share the latent-output AG/RS contract.
        (_batch(257, prefills=1, decodes=8), "ag_rs"),
        (_batch(MAX_TOKENS, prefills=3, decodes=60), "ag_rs"),
    ],
)
def test_default_policy_routes_each_workload_class(default_environment, batch, route):
    decision = _policy(4).select(batch)
    assert decision.route == route
    assert not decision.borrow_workspace


def test_dcp1_keeps_every_batch_local(default_environment):
    policy = _policy(1)
    for batch in (
        _batch(8, prefills=0, decodes=8),
        _batch(MAX_TOKENS, full_ckv_eligible=True),
        _batch(257, prefills=1, decodes=8),
    ):
        assert policy.select(batch).route == "local"


def test_policy_is_fixed_at_configuration(default_environment, monkeypatch):
    policy = _policy(4)
    monkeypatch.setenv("VLLM_DCP_A2A_MAX_TOKENS", "0")
    monkeypatch.setenv("VLLM_DCP_A2A_LARGE_BACKEND", "a2a")
    assert policy.select(_batch(MAX_TOKENS)).route == "ag_rs"


@pytest.mark.parametrize(
    ("tokens", "decode_tokens", "spec", "world", "expected"),
    [
        (17, 0, False, 4, True),
        (MAX_TOKENS, 0, False, 4, True),
        (16, 0, False, 4, False),
        (MAX_TOKENS, 8, False, 4, False),
        (MAX_TOKENS, 0, True, 4, False),
        (MAX_TOKENS, 0, False, 1, False),
    ],
)
def test_default_full_ckv_gather_predicate(
    default_environment, tokens, decode_tokens, spec, world, expected
):
    assert (
        _use_b12x_full_ckv_gather(
            enabled=envs.VLLM_B12X_MLA_CKV_GATHER,
            is_glm_next=False,
            is_glm_dsa=True,
            is_spec_decode=spec,
            dcp_world_size=world,
            max_query_len=tokens,
            num_tokens=tokens,
            num_decode_tokens=decode_tokens,
            min_tokens=envs.VLLM_B12X_MLA_CKV_GATHER_MIN_TOKENS,
            max_tokens=envs.VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS,
        )
        is expected
    )


def test_default_ckv_rank_capacity_covers_the_model_length(default_environment):
    """Each rank stages 131,136 records: a quarter of 524,288 tokens plus one
    interleave unit per sequence, rounded to the 16-token rank alignment."""
    world = 4
    capacity = _round_up_ckv_rank_tokens(
        (envs.VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS + world - 1) // world + MAX_SEQS,
        page_size=PAGE,
        dcp_world_size=world,
    )
    assert capacity == 131136
    # A 524,288-token context pads to exactly the reserved rank capacity, so
    # every prompt the launch admits (max model length 524,288) is eligible.
    padded = _round_up_ckv_rank_tokens(
        -(-524288 // world), page_size=PAGE, dcp_world_size=world
    )
    assert padded <= capacity
