#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Verify immutable vLLM wheel release assets before publication or promotion."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    """Return a file's lowercase SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_release(
    directory: Path,
    source_commit: str,
    beta_tag: str,
    *,
    promotion: bool = False,
) -> None:
    """Verify source identity and every wheel digest declared by a release."""
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["schema"] != "local-inference-vllm-wheel-release/v2":
        raise ValueError("release schema mismatch")
    if manifest["source"]["commit"] != source_commit:
        raise ValueError("source commit mismatch")
    if manifest["release_tag"] != beta_tag:
        raise ValueError("beta tag mismatch")
    for package in manifest["packages"]:
        wheel = directory / package["file"]
        if not wheel.is_file():
            wheel = directory / "wheels" / package["file"]
        if not wheel.is_file() or sha256(wheel) != package["sha256"]:
            raise ValueError(f"wheel digest mismatch: {package['file']}")
    if promotion and not (directory / "stable-promotion.json").is_file():
        raise ValueError("stable promotion record missing")


def main() -> None:
    """Parse command-line arguments and enforce the release contract."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--beta-tag", required=True)
    parser.add_argument("--promotion", action="store_true")
    args = parser.parse_args()
    verify_release(
        args.directory,
        args.source_commit,
        args.beta_tag,
        promotion=args.promotion,
    )


if __name__ == "__main__":
    main()
