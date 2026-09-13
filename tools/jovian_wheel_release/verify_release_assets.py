#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Verify an immutable Jovian application-wheel release asset set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    """Return the hexadecimal SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def checksum_entries(path: Path) -> dict[str, str]:
    """Read sha256sum output and reject non-basename paths."""
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, declared_path = line.split(maxsplit=1)
        name = declared_path.lstrip("* ")
        if Path(name).name != name or name in entries:
            raise ValueError(f"invalid checksum member: {name}")
        entries[name] = digest
    return entries


def verify_release(
    directory: Path,
    vllm_commit: str,
    b12x_commit: str,
    lmcache_commit: str,
) -> None:
    """Verify source identity, exact membership, and the archive digest."""
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    expected_source = {
        "vllm": vllm_commit,
        "b12x": b12x_commit,
        "lmcache": lmcache_commit,
    }
    for component, commit in expected_source.items():
        if manifest["source"][component]["commit"] != commit:
            raise ValueError(f"{component} source commit mismatch")

    archive = (
        f"jovian-judgement-wheels-{vllm_commit}-b12x-{b12x_commit}"
        f"-lmcache-{lmcache_commit}.tar.zst"
    )
    expected = {"manifest.json", archive, f"{archive}.sha256"}
    actual = {path.name for path in directory.iterdir() if path.is_file()}
    if actual != expected:
        raise ValueError(
            f"release asset set mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    checksums = checksum_entries(directory / f"{archive}.sha256")
    if checksums != {archive: sha256(directory / archive)}:
        raise ValueError("application archive digest mismatch")


def main() -> None:
    """Parse command-line arguments and enforce the release contract."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--vllm-commit", required=True)
    parser.add_argument("--b12x-commit", required=True)
    parser.add_argument("--lmcache-commit", required=True)
    args = parser.parse_args()
    verify_release(
        args.directory,
        args.vllm_commit,
        args.b12x_commit,
        args.lmcache_commit,
    )


if __name__ == "__main__":
    main()
