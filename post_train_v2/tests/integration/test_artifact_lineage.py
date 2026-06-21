from __future__ import annotations

from pathlib import Path

from post_train_v2.src.artifacts.atomic import publish_json
from post_train_v2.src.artifacts.hashing import sha256_file
from post_train_v2.src.artifacts.lineage import check_artifact_status
from post_train_v2.src.artifacts.manifest import (
    ArtifactFile,
    ManifestV2,
    ParentArtifact,
    publish_manifest,
)


def test_artifact_lineage_accepts_complete_parent_config_and_outputs(
    tmp_path: Path,
) -> None:
    parent_path, parent_manifest = _write_manifest(
        tmp_path / "parent",
        stage="build_source",
        filename="source.jsonl",
        payload=b'{"id":"a"}\n',
        parents=[],
        config={"seed": 1},
    )
    child_path, _ = _write_manifest(
        tmp_path / "child",
        stage="validation_split",
        filename="val_200.jsonl",
        payload=b'{"id":"a"}\n',
        parents=[
            ParentArtifact(parent_manifest.artifact_id, sha256_file(parent_path))
        ],
        config={"seed": 2},
    )

    status = check_artifact_status(
        child_path,
        expected_config={"seed": 2},
        parent_manifest_paths=[parent_path],
    )

    assert status.state == "complete"
    assert status.reason == "complete"
    assert status.manifest_path == child_path


def test_artifact_lineage_marks_changed_output_stale(tmp_path: Path) -> None:
    path, _ = _write_manifest(
        tmp_path / "stage",
        stage="build_source",
        filename="source.jsonl",
        payload=b"old\n",
        parents=[],
        config={"seed": 1},
    )
    (path.parent / "source.jsonl").write_bytes(b"new\n")

    status = check_artifact_status(path, expected_config={"seed": 1})

    assert status.state == "stale"
    assert "output hash mismatch" in status.reason


def test_artifact_lineage_rejects_changed_config_and_parent_hash(
    tmp_path: Path,
) -> None:
    parent_path, parent_manifest = _write_manifest(
        tmp_path / "parent",
        stage="build_source",
        filename="source.jsonl",
        payload=b"parent\n",
        parents=[],
        config={"seed": 1},
    )
    child_path, _ = _write_manifest(
        tmp_path / "child",
        stage="validation_split",
        filename="val_200.jsonl",
        payload=b"child\n",
        parents=[ParentArtifact(parent_manifest.artifact_id, "0" * 64)],
        config={"seed": 2},
    )

    assert (
        check_artifact_status(child_path, expected_config={"seed": 3}).state
        == "stale"
    )
    status = check_artifact_status(
        child_path,
        expected_config={"seed": 2},
        parent_manifest_paths=[parent_path],
    )
    assert status.state == "stale"
    assert "parent hash mismatch" in status.reason


def test_artifact_lineage_marks_missing_failed_and_partial_teacher(tmp_path: Path) -> None:
    missing = check_artifact_status(tmp_path / "missing_manifest.json")
    assert missing.state == "missing"

    invalid_path = tmp_path / "invalid.json"
    publish_json(invalid_path, {"not": "a manifest"})
    failed = check_artifact_status(invalid_path)
    assert failed.state == "failed"
    assert "manifest" in failed.reason

    partial_path, _ = _write_manifest(
        tmp_path / "teacher",
        stage="teacher_pool",
        filename="teacher_accepted.jsonl",
        payload=b"partial\n",
        parents=[],
        config={"target_accepted": 20_000},
        completed=False,
    )
    partial = check_artifact_status(
        partial_path,
        expected_config={"target_accepted": 20_000},
    )
    assert partial.state == "stale"
    assert "not marked complete" in partial.reason


def _write_manifest(
    directory: Path,
    *,
    stage: str,
    filename: str,
    payload: bytes,
    parents: list[ParentArtifact],
    config: dict,
    completed: bool = True,
) -> tuple[Path, ManifestV2]:
    directory.mkdir(parents=True)
    data_path = directory / filename
    data_path.write_bytes(payload)
    manifest = ManifestV2.build(
        artifact_type="dataset",
        stage=stage,
        files=[
            ArtifactFile(
                filename,
                sha256_file(data_path),
                data_path.stat().st_size,
                1,
                {"id": "string"},
            )
        ],
        parents=parents,
        config=config,
        stage_metadata={"completed": completed},
        git_revision="fixture",
    )
    manifest_path = directory / f"{stage}_manifest.json"
    publish_manifest(manifest_path, manifest)
    return manifest_path, manifest
