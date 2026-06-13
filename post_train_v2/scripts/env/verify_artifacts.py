#!/usr/bin/env python3
"""Verify exact repository-local wheel filenames and SHA-256 digests."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path, PureWindowsPath
from typing import Any


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"manifest must be an object: {path}")
    artifacts = data.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError(f"manifest has no artifacts: {path}")
    return data


def validate_artifact_metadata(
    artifact_name: str,
    metadata: object,
) -> tuple[str, str]:
    if not isinstance(metadata, dict):
        raise ValueError(f"artifact {artifact_name} metadata must be an object")

    filename = metadata.get("filename")
    url = metadata.get("url")
    expected_hash = metadata.get("sha256")

    if not isinstance(filename, str) or not filename:
        raise ValueError(
            f"invalid exact filename for artifact {artifact_name}: {filename!r}"
        )
    if (
        "/" in filename
        or "\\" in filename
        or Path(filename).is_absolute()
        or PureWindowsPath(filename).is_absolute()
        or PureWindowsPath(filename).drive
        or Path(filename).name != filename
    ):
        raise ValueError(
            f"invalid exact filename for artifact {artifact_name}: {filename!r}"
        )
    if not isinstance(url, str) or not url:
        raise ValueError(f"invalid URL for artifact {artifact_name}: {url!r}")
    if not isinstance(expected_hash, str) or re.fullmatch(
        r"[0-9a-f]{64}", expected_hash
    ) is None:
        raise ValueError(
            f"invalid SHA-256 for artifact {artifact_name}: {expected_hash!r}"
        )

    return filename, expected_hash


def verify_artifacts(
    manifest_path: Path,
    wheels_dir: Path,
) -> list[tuple[str, str]]:
    manifest = load_manifest(manifest_path)
    results: list[tuple[str, str]] = []

    for artifact_name, metadata in manifest["artifacts"].items():
        filename, expected_hash = validate_artifact_metadata(
            artifact_name, metadata
        )

        artifact_path = wheels_dir / filename
        if artifact_path.is_symlink():
            raise ValueError(f"artifact path must not be a symlink: {artifact_path}")
        if not artifact_path.is_file():
            raise FileNotFoundError(f"missing artifact: {artifact_path}")

        actual_hash = sha256_file(artifact_path)
        if actual_hash != expected_hash:
            raise ValueError(
                "SHA-256 mismatch for "
                f"{filename}: expected {expected_hash}, got {actual_hash}"
            )

        print(f"OK {filename} {actual_hash}")
        results.append((filename, actual_hash))

    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify exact filenames and SHA-256 hashes for runtime wheels."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to runtime-cu128.json.",
    )
    parser.add_argument(
        "--wheels-dir",
        type=Path,
        required=True,
        help="Directory containing the exact wheel files.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        verify_artifacts(args.manifest, args.wheels_dir)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
