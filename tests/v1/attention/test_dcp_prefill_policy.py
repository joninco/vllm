# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for rank-uniform prefill route selection.

Shared batch facts and immutable configuration select one mutually exclusive
exchange. These tests do not establish serving integration or GPU correctness.
"""

from dataclasses import replace

import pytest

from vllm.v1.attention.ops.dcp_prefill_policy import (
    DCPPrefillBatch,
    DCPPrefillDecision,
    DCPPrefillPolicy,
)


@pytest.fixture
def policy():
    return DCPPrefillPolicy(
        dcp_world_size=4,
        max_num_tokens=8192,
        enabled=True,
        a2a_max_tokens=256,
        project_before_merge=True,
        borrow_workspace=True,
    )


@pytest.mark.parametrize(
    ("rows", "route", "borrow"),
    [
        (0, "configured", False),
        (16, "configured", False),
        (17, "a2a", False),
        (256, "a2a", False),
        (257, "ag_rs", False),
        (1024, "ag_rs", False),
        (1025, "projected_ag_rs", True),
        (8192, "projected_ag_rs", True),
        (8193, "configured", False),
    ],
)
def test_prefill_thresholds_are_exclusive_and_capacity_is_inclusive(
    policy, rows, route, borrow
):
    batch = DCPPrefillBatch(rows, num_prefills=1, num_decodes=0)
    assert policy.select(batch) == DCPPrefillDecision(route, borrow)


@pytest.mark.parametrize(
    "changes",
    [
        {"is_capturing": True},
        {"is_mtp": True},
        {"num_decodes": 1},
        {"num_prefills": 0},
    ],
)
def test_decode_mixed_mtp_and_capture_use_configured_route(policy, changes):
    batch = DCPPrefillBatch(2048, 1, 0, full_ckv_eligible=True)
    assert policy.select(replace(batch, **changes)).route == "configured"


def test_full_ckv_excludes_query_and_projected_partial_exchange(policy):
    batch = DCPPrefillBatch(2048, 1, 0, full_ckv_eligible=True)
    assert policy.select(batch) == DCPPrefillDecision("full_ckv")
    assert replace(policy, dcp_world_size=1).select(batch) == DCPPrefillDecision(
        "local"
    )
    assert replace(policy, enabled=False).select(batch).route == "configured"


@pytest.mark.parametrize("rows", [1, 4, 8, 16])
def test_decode_rows_preserve_transport_with_all_prefill_options_enabled(policy, rows):
    assert policy.select(DCPPrefillBatch(rows, 0, rows)).route == "configured"


@pytest.mark.parametrize(
    ("changes", "route", "borrow"),
    [
        ({"a2a_max_tokens": 0}, "a2a", False),
        ({"a2a_max_tokens": -1}, "a2a", False),
        ({"large_backend": "a2a"}, "a2a", False),
        ({"project_before_merge": False, "borrow_workspace": False}, "ag_rs", False),
        ({"non_dbo_workspace": False}, "projected_ag_rs", False),
        ({"borrow_workspace": False}, "projected_ag_rs", False),
        ({"base_backend": "ag_rs", "a2a_max_tokens": 0}, "projected_ag_rs", True),
    ],
)
def test_explicit_transport_and_workspace_options(policy, changes, route, borrow):
    assert replace(policy, **changes).select(DCPPrefillBatch(2048, 1, 0)) == (
        DCPPrefillDecision(route, borrow)
    )


def test_shared_batch_facts_produce_the_same_decision_on_every_rank(policy):
    batch = DCPPrefillBatch(3073, 3, 0)
    decisions = [replace(policy).select(replace(batch)) for _ in range(4)]
    assert decisions == [DCPPrefillDecision("projected_ag_rs", True)] * 4


@pytest.mark.parametrize(
    "changes",
    [
        {"dcp_world_size": 0},
        {"max_num_tokens": 0},
        {"project_min_tokens": -1},
        {"base_backend": "invalid"},
        {"large_backend": "invalid"},
        {"project_min_tokens": 15},
        {"project_before_merge": False},
    ],
)
def test_invalid_configuration_fails_before_collectives(policy, changes):
    with pytest.raises(ValueError):
        replace(policy, **changes)


def test_negative_batch_counts_are_rejected():
    with pytest.raises(ValueError, match="counts"):
        DCPPrefillBatch(-1, 1, 0)
