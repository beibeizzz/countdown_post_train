"""Build deterministic validation and accepted-training dataset splits."""

from __future__ import annotations

import errno
import hashlib
import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from post_train_v2.src.artifacts.atomic import publish_jsonl
from post_train_v2.src.artifacts.hashing import sha256_file
from post_train_v2.src.artifacts.manifest import (
    ArtifactFile,
    ManifestV2,
    ParentArtifact,
    load_manifest,
    publish_manifest,
)
from post_train_v2.src.config.loading import load_yaml, require_keys, resolve_repo_path
from post_train_v2.src.countdown.sampling import exclude_ids, stratified_sample
from post_train_v2.src.data.schema import (
    validate_normalized_source,
    validate_sft_record,
)


CONFIG_KEYS = {
    "seed",
    "source_data",
    "source_manifest",
    "teacher_accepted",
    "teacher_manifest",
    "output_dir",
    "val_size",
    "eval_size",
    "sft_size",
    "grpo_size",
}
SOURCE_FIELD_SCHEMA = {
    "id": "string",
    "source_index": "integer",
    "numbers": "array[integer]",
    "target": "integer",
    "gold_expr": "string",
    "prompt": "string",
    "bucket": "object",
}
SFT_FIELD_SCHEMA = {
    **SOURCE_FIELD_SCHEMA,
    "response": "string",
    "validation": "object",
    "provenance": "object",
}
SEED_DERIVATION_VERSION = "sha256-stage-v1"


def derive_stage_seed(seed: int, stage: str) -> int:
    """Derive a stable independent integer seed for a named split stage."""

    _exact_nonnegative_int(seed, "seed")
    if not isinstance(stage, str) or not stage:
        raise ValueError("stage must be a nonempty string")
    payload = f"{SEED_DERIVATION_VERSION}|{seed}|{stage}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def read_jsonl_strict(
    path: str | Path,
    validator: Callable[[Mapping[str, Any]], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Read JSONL objects, ignoring empty lines and reporting malformed lines."""

    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSONL at {path} line {line_number}: {error.msg}"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(
                    f"JSONL row at {path} line {line_number} must be an object"
                )
            try:
                rows.append(validator(value))
            except ValueError as error:
                raise ValueError(
                    f"invalid row at {path} line {line_number}: {error}"
                ) from error
    return rows


def _exact_nonnegative_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative exact integer")
    return value


def _positive_int(value: Any, name: str) -> int:
    normalized = _exact_nonnegative_int(value, name)
    if normalized == 0:
        raise ValueError(f"{name} must be a positive exact integer")
    return normalized


def _logical_path(value: Any, config_dir: Path, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonempty path string")
    path = Path(value)
    if not path.is_absolute():
        return path.as_posix()
    try:
        return Path(os.path.relpath(path, start=config_dir)).as_posix()
    except ValueError as error:
        raise ValueError(
            f"{field} absolute path must share a filesystem root with config"
        ) from error


def _load_config(config_path: str | Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    resolved = resolve_repo_path(config_path)
    config = load_yaml(resolved)
    require_keys(config, *sorted(CONFIG_KEYS))
    if set(config) != CONFIG_KEYS:
        raise ValueError("build_splits config has unexpected keys")
    seed = _exact_nonnegative_int(config["seed"], "seed")
    normalized = {
        "seed": seed,
        "source_data": resolve_repo_path(config["source_data"]),
        "source_manifest": resolve_repo_path(config["source_manifest"]),
        "teacher_accepted": resolve_repo_path(config["teacher_accepted"]),
        "teacher_manifest": resolve_repo_path(config["teacher_manifest"]),
        "output_dir": resolve_repo_path(config["output_dir"]),
        "val_size": _positive_int(config["val_size"], "val_size"),
        "eval_size": _positive_int(config["eval_size"], "eval_size"),
        "sft_size": _positive_int(config["sft_size"], "sft_size"),
        "grpo_size": _positive_int(config["grpo_size"], "grpo_size"),
    }
    logical = {
        key: (
            _logical_path(config[key], resolved.parent, key)
            if key
            in {
                "source_data",
                "source_manifest",
                "teacher_accepted",
                "teacher_manifest",
                "output_dir",
            }
            else normalized[key]
        )
        for key in (
            "seed",
            "source_data",
            "source_manifest",
            "teacher_accepted",
            "teacher_manifest",
            "output_dir",
            "val_size",
            "eval_size",
            "sft_size",
            "grpo_size",
        )
    }
    return resolved, normalized, logical


def _manifest_file(
    manifest: ManifestV2,
    data_path: Path,
    *,
    expected_schema: dict[str, str],
) -> ArtifactFile:
    matches = [
        item for item in manifest.files if item.relative_path == data_path.name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"manifest file entry not found for {data_path.name}"
        )
    item = matches[0]
    actual_hash = sha256_file(data_path)
    if item.sha256 != actual_hash:
        raise ValueError(f"manifest file hash mismatch for {data_path.name}")
    if item.byte_size != data_path.stat().st_size:
        raise ValueError(f"manifest file size mismatch for {data_path.name}")
    if item.field_schema != expected_schema:
        raise ValueError(f"manifest file schema mismatch for {data_path.name}")
    return item


def _require_completed(manifest: ManifestV2, label: str) -> None:
    if manifest.stage_metadata.get("completed") is not True:
        raise ValueError(f"{label} manifest must have completed=true")


def _require_unique_ids(rows: list[dict[str, Any]], label: str) -> None:
    seen: set[str] = set()
    for row in rows:
        row_id = row["id"]
        if row_id in seen:
            raise ValueError(f"duplicate {label} id: {row_id}")
        seen.add(row_id)


def _require_row_count(item: ArtifactFile, rows: list[dict[str, Any]]) -> None:
    if item.row_count != len(rows):
        raise ValueError(
            f"manifest file count mismatch for {item.relative_path}: "
            f"expected {item.row_count}, read {len(rows)}"
        )


def _restore_source_order(
    sampled: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    positions = {row["id"]: index for index, row in enumerate(source_rows)}
    return sorted(sampled, key=lambda row: positions[row["id"]])


def _artifact_file(
    output_dir: Path,
    filename: str,
    count: int,
    schema: dict[str, str],
) -> ArtifactFile:
    path = output_dir / filename
    return ArtifactFile(
        relative_path=filename,
        sha256=sha256_file(path),
        byte_size=path.stat().st_size,
        row_count=count,
        field_schema=schema,
    )


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    unsupported = {
        errno.EINVAL,
        getattr(errno, "ENOTSUP", errno.EINVAL),
        getattr(errno, "EOPNOTSUPP", errno.EINVAL),
    }
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(directory, flags)
    except OSError as error:
        if error.errno in unsupported:
            return
        raise
    try:
        try:
            os.fsync(descriptor)
        except OSError as error:
            if error.errno not in unsupported:
                raise
    finally:
        os.close(descriptor)


def _revoke_manifest(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    _fsync_directory(path.parent)


def _snapshot(paths: list[Path]) -> dict[Path, str]:
    return {path: sha256_file(path) for path in paths}


def _require_unchanged(snapshot: dict[Path, str], label: str) -> None:
    for path, digest in snapshot.items():
        if sha256_file(path) != digest:
            raise ValueError(f"{label} input changed during split build: {path}")


def _parent(manifest: ManifestV2, manifest_path: Path) -> ParentArtifact:
    return ParentArtifact(
        artifact_id=manifest.artifact_id,
        sha256=sha256_file(manifest_path),
    )


def run_validation_splits(config_path: str | Path) -> ManifestV2:
    """Publish validation, fixed-eval, and train-candidate source splits."""

    _, config, logical_config = _load_config(config_path)
    source_path = config["source_data"]
    source_manifest_path = config["source_manifest"]
    output_dir = config["output_dir"]
    snapshots = _snapshot([source_path, source_manifest_path])

    source_manifest = load_manifest(source_manifest_path)
    if source_manifest.stage != "build_source":
        raise ValueError("source manifest stage must be build_source")
    _require_completed(source_manifest, "source")
    source_file = _manifest_file(
        source_manifest,
        source_path,
        expected_schema=SOURCE_FIELD_SCHEMA,
    )
    rows = read_jsonl_strict(source_path, validate_normalized_source)
    _require_unique_ids(rows, "source")
    _require_row_count(source_file, rows)

    val_seed = derive_stage_seed(config["seed"], "validation")
    eval_seed = derive_stage_seed(config["seed"], "evaluation")
    sampled_val = stratified_sample(rows, config["val_size"], val_seed)
    sampled_eval = stratified_sample(sampled_val, config["eval_size"], eval_seed)
    train_candidates = exclude_ids(
        rows, {row["id"] for row in sampled_val}
    )
    val_rows = _restore_source_order(sampled_val, rows)
    eval_rows = _restore_source_order(sampled_eval, rows)
    train_rows = _restore_source_order(train_candidates, rows)

    _require_unchanged(snapshots, "validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "validation_manifest.json"
    _revoke_manifest(manifest_path)
    publish_jsonl(output_dir / "val_200.jsonl", val_rows)
    publish_jsonl(output_dir / "eval_50.jsonl", eval_rows)
    publish_jsonl(output_dir / "train_candidates.jsonl", train_rows)

    manifest = ManifestV2.build(
        artifact_type="dataset",
        stage="build_validation_splits",
        files=[
            _artifact_file(
                output_dir, "val_200.jsonl", len(val_rows), SOURCE_FIELD_SCHEMA
            ),
            _artifact_file(
                output_dir, "eval_50.jsonl", len(eval_rows), SOURCE_FIELD_SCHEMA
            ),
            _artifact_file(
                output_dir,
                "train_candidates.jsonl",
                len(train_rows),
                SOURCE_FIELD_SCHEMA,
            ),
        ],
        parents=[_parent(source_manifest, source_manifest_path)],
        config=logical_config,
        global_seed=config["seed"],
        seed_derivation_version=SEED_DERIVATION_VERSION,
        created_at=source_manifest.created_at,
        stage_metadata={
            "completed": True,
            "counts": {
                "source": len(rows),
                "validation": len(val_rows),
                "evaluation": len(eval_rows),
                "train_candidates": len(train_rows),
            },
            "derived_seeds": {
                "validation": val_seed,
                "evaluation": eval_seed,
            },
            "selected_ids": {
                "validation": [row["id"] for row in sampled_val],
                "evaluation": [row["id"] for row in sampled_eval],
            },
            "source_order_ids": {
                "validation": [row["id"] for row in val_rows],
                "evaluation": [row["id"] for row in eval_rows],
                "train_candidates": [row["id"] for row in train_rows],
            },
        },
    )
    publish_manifest(manifest_path, manifest)
    return manifest


def _validation_ids(manifest: ManifestV2) -> set[str]:
    if manifest.stage != "build_validation_splits":
        raise ValueError(
            "validation manifest stage must be build_validation_splits"
        )
    _require_completed(manifest, "validation")
    selected_ids = manifest.stage_metadata.get("selected_ids")
    if not isinstance(selected_ids, dict):
        raise ValueError("validation manifest selected_ids must be an object")
    values = selected_ids.get("validation")
    if not isinstance(values, list) or any(
        not isinstance(value, str) or not value for value in values
    ):
        raise ValueError(
            "validation manifest selected_ids.validation must be a string list"
        )
    if len(values) != len(set(values)):
        raise ValueError("validation manifest contains duplicate validation ids")
    return set(values)


def run_accepted_splits(config_path: str | Path) -> ManifestV2:
    """Publish independently sampled SFT and GRPO splits from accepted rows."""

    _, config, logical_config = _load_config(config_path)
    accepted_path = config["teacher_accepted"]
    teacher_manifest_path = config["teacher_manifest"]
    output_dir = config["output_dir"]
    validation_manifest_path = output_dir / "validation_manifest.json"
    snapshots = _snapshot(
        [accepted_path, teacher_manifest_path, validation_manifest_path]
    )

    teacher_manifest = load_manifest(teacher_manifest_path)
    if teacher_manifest.stage != "teacher_accepted_pool":
        raise ValueError("teacher manifest stage must be teacher_accepted_pool")
    _require_completed(teacher_manifest, "teacher")
    teacher_file = _manifest_file(
        teacher_manifest,
        accepted_path,
        expected_schema=SFT_FIELD_SCHEMA,
    )
    validation_manifest = load_manifest(validation_manifest_path)
    val_ids = _validation_ids(validation_manifest)
    teacher_manifest.require_parent(
        validation_manifest.artifact_id,
        sha256_file(validation_manifest_path),
    )
    rows = read_jsonl_strict(accepted_path, validate_sft_record)
    _require_unique_ids(rows, "accepted")
    _require_row_count(teacher_file, rows)
    overlap = sorted(row["id"] for row in rows if row["id"] in val_ids)
    if overlap:
        raise ValueError(f"accepted pool contains validation id: {overlap[0]}")
    for size_key, split_name in (
        ("sft_size", "sft_train_8k"),
        ("grpo_size", "grpo_train_4k"),
    ):
        if len(rows) < config[size_key]:
            raise ValueError(
                f"insufficient accepted rows for {split_name}: "
                f"requested {config[size_key]}, available {len(rows)}"
            )

    sft_seed = derive_stage_seed(config["seed"], "sft")
    grpo_seed = derive_stage_seed(config["seed"], "grpo")
    sampled_sft = stratified_sample(rows, config["sft_size"], sft_seed)
    sampled_grpo = stratified_sample(rows, config["grpo_size"], grpo_seed)
    sft_rows = _restore_source_order(sampled_sft, rows)
    grpo_rows = _restore_source_order(sampled_grpo, rows)

    _require_unchanged(snapshots, "accepted")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "accepted_splits_manifest.json"
    _revoke_manifest(manifest_path)
    publish_jsonl(output_dir / "sft_train_8k.jsonl", sft_rows)
    publish_jsonl(output_dir / "grpo_train_4k.jsonl", grpo_rows)

    created_at = max(teacher_manifest.created_at, validation_manifest.created_at)
    manifest = ManifestV2.build(
        artifact_type="dataset",
        stage="build_accepted_splits",
        files=[
            _artifact_file(
                output_dir, "sft_train_8k.jsonl", len(sft_rows), SFT_FIELD_SCHEMA
            ),
            _artifact_file(
                output_dir,
                "grpo_train_4k.jsonl",
                len(grpo_rows),
                SFT_FIELD_SCHEMA,
            ),
        ],
        parents=[
            _parent(teacher_manifest, teacher_manifest_path),
            _parent(validation_manifest, validation_manifest_path),
        ],
        config=logical_config,
        global_seed=config["seed"],
        seed_derivation_version=SEED_DERIVATION_VERSION,
        created_at=created_at,
        stage_metadata={
            "completed": True,
            "counts": {
                "accepted_pool": len(rows),
                "sft_train": len(sft_rows),
                "grpo_train": len(grpo_rows),
            },
            "derived_seeds": {"sft": sft_seed, "grpo": grpo_seed},
            "sampling": "independent_with_overlap_allowed",
            "selected_ids": {
                "sft": [row["id"] for row in sampled_sft],
                "grpo": [row["id"] for row in sampled_grpo],
            },
            "source_order_ids": {
                "sft": [row["id"] for row in sft_rows],
                "grpo": [row["id"] for row in grpo_rows],
            },
            "validation_ids": sorted(val_ids),
        },
    )
    publish_manifest(manifest_path, manifest)
    return manifest
