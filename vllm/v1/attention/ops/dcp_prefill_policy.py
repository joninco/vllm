# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rank-uniform selection of optional eager sparse-MLA prefill exchanges.

This module owns no collectives or environment reads. Callers must construct
batch facts from shared scheduler metadata, never rank-local KV lengths.
"""

from dataclasses import dataclass
from typing import Literal

DCPPrefillRoute = Literal[
    "local", "configured", "full_ckv", "a2a", "ag_rs", "projected_ag_rs"
]


@dataclass(frozen=True)
class DCPPrefillBatch:
    """Shared execution facts; CKV eligibility includes its storage contract."""

    num_tokens: int
    num_prefills: int
    num_decodes: int
    is_capturing: bool = False
    is_mtp: bool = False
    full_ckv_eligible: bool = False

    def __post_init__(self) -> None:
        if min(self.num_tokens, self.num_prefills, self.num_decodes) < 0:
            raise ValueError("DCP batch counts must be non-negative")


@dataclass(frozen=True)
class DCPPrefillDecision:
    """Exchange route and permission to borrow intra-layer scratch storage."""

    route: DCPPrefillRoute
    borrow_workspace: bool = False


@dataclass(frozen=True)
class DCPPrefillPolicy:
    """Validated configuration for optional prefill routing.

    ``configured`` delegates to the configured manager transport and
    projection. A non-positive A2A cap means uncapped A2A. Token minima are exclusive.
    The attention layer must consume the decision before issuing collectives.
    """

    dcp_world_size: int
    max_num_tokens: int
    enabled: bool = False
    base_backend: Literal["a2a", "ag_rs"] = "a2a"
    a2a_max_tokens: int = 0
    large_backend: Literal["a2a", "ag_rs"] = "ag_rs"
    min_prefill_tokens: int = 16
    project_before_merge: bool = False
    project_min_tokens: int = 1024
    max_capture_tokens: int = 16
    borrow_workspace: bool = False
    non_dbo_workspace: bool = True

    def __post_init__(self) -> None:
        if self.dcp_world_size < 1 or self.max_num_tokens < 1:
            raise ValueError("DCP world size and token capacity must be positive")
        if (
            min(
                self.min_prefill_tokens,
                self.project_min_tokens,
                self.max_capture_tokens,
            )
            < 0
        ):
            raise ValueError("DCP token thresholds must be non-negative")
        if self.base_backend not in ("a2a", "ag_rs") or self.large_backend not in (
            "a2a",
            "ag_rs",
        ):
            raise ValueError("DCP prefill backends must be 'a2a' or 'ag_rs'")
        if self.project_before_merge and (
            self.project_min_tokens < self.max_capture_tokens
        ):
            raise ValueError("DCP projection threshold must cover captured token sizes")
        if self.borrow_workspace and not self.project_before_merge:
            raise ValueError("Borrowed DCP workspace requires projected merge")

    def select(self, batch: DCPPrefillBatch) -> DCPPrefillDecision:
        """Select one exchange before any rank starts a collective."""
        if self.dcp_world_size == 1:
            return DCPPrefillDecision("local")
        if (
            not self.enabled
            or batch.is_capturing
            or batch.is_mtp
            or batch.num_prefills == 0
            or batch.num_decodes != 0
            or not self.min_prefill_tokens < batch.num_tokens <= self.max_num_tokens
        ):
            return DCPPrefillDecision("configured")
        if batch.full_ckv_eligible:
            return DCPPrefillDecision("full_ckv")
        backend = self.base_backend
        if (
            backend == "a2a"
            and self.a2a_max_tokens > 0
            and batch.num_tokens > self.a2a_max_tokens
        ):
            backend = self.large_backend
        if (
            backend == "ag_rs"
            and self.project_before_merge
            and batch.num_tokens > self.project_min_tokens
        ):
            return DCPPrefillDecision(
                "projected_ag_rs",
                borrow_workspace=(
                    self.borrow_workspace
                    and self.non_dbo_workspace
                    and batch.num_tokens >= 1025
                ),
            )
        return DCPPrefillDecision(backend)
