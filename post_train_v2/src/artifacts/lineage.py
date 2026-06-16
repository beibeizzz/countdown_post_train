"""Manifest-driven artifact freshness checks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from post_train_v2.src.artifacts.hashing import sha256_config, sha256_file
from post_train_v2.src.artifacts.manifest import ManifestV2, load_manifest

ArtifactState = Literal["missing", "complete", "stale", "failed"]


@dataclass(frozen=True)
class ArtifactStatus:
    state: ArtifactState
    reason: str
    manifest_path: Path | None


def check_artifact_status(
    manifest_path: str | Path,
    *,
    expected_config: Mapping | None = None,
    parent_manifest_paths: Iterable[str | Path] = (),
) -> ArtifactStatus:
    path = Path(manifest_path)
    if not path.is_file():
        return ArtifactStatus("missing", "manifest missing", path)

    try:
        manifest = load_manifest(path)
    except Exception as error:
        return ArtifactStatus("failed", str(error), path)

    if expected_config is not None and manifest.config_sha256 != sha256_config(
        dict(expected_config)
    ):
        return ArtifactStatus("stale", "config hash mismatch", path)

    if manifest.stage_metadata.get("completed") is not True:
        return ArtifactStatus("stale", "manifest not marked complete", path)

    parent_status = _validate_parent_manifests(
        manifest,
        [Path(parent_path) for parent_path in parent_manifest_paths],
    )
    if parent_status is not None:
        return ArtifactStatus("stale", parent_status, path)

    output_status = _validate_output_files(path.parent, manifest)
    if output_status is not None:
        return ArtifactStatus("stale", output_status, path)

    return ArtifactStatus("complete", "complete", path)


def _validate_parent_manifests(
    manifest: ManifestV2,
    parent_paths: list[Path],
) -> str | None:
    if not parent_paths and not manifest.parents:
        return None
    by_artifact_id = {}
    for parent_path in parent_paths:
        if not parent_path.is_file():
            return f"parent manifest missing: {parent_path}"
        try:
            parent = load_manifest(parent_path)
        except Exception as error:
            return f"parent manifest invalid: {error}"
        by_artifact_id[parent.artifact_id] = (parent, sha256_file(parent_path))

    for declared in manifest.parents:
        parent = by_artifact_id.get(declared.artifact_id)
        if parent is None:
            return f"parent artifact not provided: {declared.artifact_id}"
        _, actual_hash = parent
        if declared.sha256 != actual_hash:
            return f"parent hash mismatch for artifact {declared.artifact_id}"
    return None


def _validate_output_files(manifest_dir: Path, manifest: ManifestV2) -> str | None:
    for item in manifest.files:
        output_path = manifest_dir / item.relative_path
        if not output_path.is_file():
            return f"output missing: {item.relative_path}"
        if output_path.stat().st_size != item.byte_size:
            return f"output byte size mismatch: {item.relative_path}"
        if sha256_file(output_path) != item.sha256:
            return f"output hash mismatch: {item.relative_path}"
    return None
