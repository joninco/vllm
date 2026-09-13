# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for the application-wheel release verifier."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.jovian_wheel_release.verify_release_assets import verify_release

VLLM_COMMIT = "1" * 40
B12X_COMMIT = "2" * 40
LMCACHE_COMMIT = "3" * 40
ARCHIVE = (
    f"jovian-judgement-wheels-{VLLM_COMMIT}-b12x-{B12X_COMMIT}"
    f"-lmcache-{LMCACHE_COMMIT}.tar.zst"
)


def write_release(directory: Path) -> None:
    """Write the smallest complete release accepted by the verifier."""
    manifest = {
        "source": {
            "vllm": {"commit": VLLM_COMMIT},
            "b12x": {"commit": B12X_COMMIT},
            "lmcache": {"commit": LMCACHE_COMMIT},
        }
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    payload = b"archive"
    (directory / ARCHIVE).write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    (directory / f"{ARCHIVE}.sha256").write_text(f"{digest}  {ARCHIVE}\n")


def test_accepts_complete_release(tmp_path: Path) -> None:
    """A release with matching identities and archive hash is accepted."""
    write_release(tmp_path)
    verify_release(tmp_path, VLLM_COMMIT, B12X_COMMIT, LMCACHE_COMMIT)


def test_rejects_changed_archive(tmp_path: Path) -> None:
    """A post-publication archive modification is rejected."""
    write_release(tmp_path)
    (tmp_path / ARCHIVE).write_bytes(b"changed")
    with pytest.raises(ValueError, match="archive digest mismatch"):
        verify_release(tmp_path, VLLM_COMMIT, B12X_COMMIT, LMCACHE_COMMIT)


def test_rejects_extra_asset(tmp_path: Path) -> None:
    """Unexpected release members cannot hide under a valid manifest."""
    write_release(tmp_path)
    (tmp_path / "extra").write_text("unexpected")
    with pytest.raises(ValueError, match="asset set mismatch"):
        verify_release(tmp_path, VLLM_COMMIT, B12X_COMMIT, LMCACHE_COMMIT)
