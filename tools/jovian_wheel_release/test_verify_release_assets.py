# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for the vLLM wheel release verifier."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.jovian_wheel_release.verify_release_assets import verify_release

COMMIT = "1" * 40
TAG = f"vllm-jovian-cu134-beta-{COMMIT}"
WHEEL = "vllm-1-py3-none-any.whl"


def write_release(directory: Path) -> None:
    """Write the smallest complete release accepted by the verifier."""
    wheel = directory / WHEEL
    wheel.write_bytes(b"wheel")
    manifest = {
        "schema": "local-inference-vllm-wheel-release/v2",
        "source": {"commit": COMMIT},
        "release_tag": TAG,
        "packages": [{"file": WHEEL, "sha256": hashlib.sha256(b"wheel").hexdigest()}],
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))


def test_accepts_matching_release(tmp_path: Path) -> None:
    """Matching source identity and wheel bytes satisfy the contract."""
    write_release(tmp_path)
    verify_release(tmp_path, COMMIT, TAG)


def test_rejects_changed_wheel(tmp_path: Path) -> None:
    """A post-publication wheel modification is rejected."""
    write_release(tmp_path)
    (tmp_path / WHEEL).write_bytes(b"changed")
    with pytest.raises(ValueError, match="wheel digest mismatch"):
        verify_release(tmp_path, COMMIT, TAG)
