"""Build deterministic validation and accepted-training dataset splits."""

from __future__ import annotations

import errno
import hashlib
import json
import os
from collections.abc import Callable, Mapping
from contextlib import contextmanager
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
from post_train_v2.src.artifacts.locking import exclusive_output_lock
from post_train_v2.src.config.loading import (
    REPO_ROOT,
    load_yaml,
    require_keys,
    resolve_repo_path,
)
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


def _logical_path(value: Any, _config_dir: Path, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonempty path string")
    path = resolve_repo_path(value)
    try:
        return path.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


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


def _metadata_count(metadata: Mapping[str, Any], key: str, label: str) -> int:
    if key not in metadata:
        raise ValueError(f"{label} metadata missing {key}")
    value = metadata[key]
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} metadata {key} must be an exact integer")
    return value


def _require_complete_source_manifest(
    manifest: ManifestV2,
    source_file: ArtifactFile,
) -> None:
    metadata = manifest.stage_metadata
    if "limit" not in metadata or metadata["limit"] is not None:
        raise ValueError("source manifest limit must be None")
    counts = metadata.get("counts")
    if not isinstance(counts, Mapping):
        raise ValueError("source manifest counts must be an object")
    train_input = _metadata_count(counts, "train_input", "source counts")
    solvable = _metadata_count(counts, "solvable_train", "source counts")
    unsolved = _metadata_count(counts, "unsolved_train", "source counts")
    if train_input != solvable + unsolved:
        raise ValueError(
            "source counts train_input must equal "
            "solvable_train + unsolved_train"
        )
    if source_file.row_count != solvable:
        raise ValueError(
            "source file row_count must equal source counts solvable_train"
        )


def _teacher_completion_counts(
    manifest: ManifestV2,
    teacher_file: ArtifactFile,
) -> tuple[int, int]:
    metadata = manifest.stage_metadata
    _require_completed(manifest, "teacher")
    accepted_count = _metadata_count(
        metadata, "accepted_count", "teacher"
    )
    target_count = _metadata_count(
        metadata, "target_accepted_count", "teacher"
    )
    if target_count == 0:
        raise ValueError(
            "teacher metadata target_accepted_count must be a positive exact integer"
        )
    if accepted_count != teacher_file.row_count:
        raise ValueError(
            "teacher accepted_count must equal teacher file row_count"
        )
    if accepted_count != target_count:
        raise ValueError(
            "teacher completed accepted_count must equal target_accepted_count"
        )
    return accepted_count, target_count


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


@contextmanager
def _split_output_lock(
    output_dir: Path,
    *,
    config_path: Path,
    mode: str,
):
    try:
        with exclusive_output_lock(
            output_dir,
            lock_name=".build_splits.lock",
            metadata={
                "config_path": str(config_path.resolve()),
                "output_dir": str(output_dir.resolve()),
                "mode": mode,
            },
        ):
            yield
    except RuntimeError as error:
        raise RuntimeError(f"split {error}") from error


def _snapshot(paths: list[Path]) -> dict[Path, str]:
    return {path: sha256_file(path) for path in paths}


def _require_unchanged(snapshot: dict[Path, str], label: str) -> None:
    for path, digest in snapshot.items():
        if sha256_file(path) != digest:
            raise ValueError(f"{label} input changed during split build: {path}")


def _parent(manifest: ManifestV2, digest: str) -> ParentArtifact:
    return ParentArtifact(
        artifact_id=manifest.artifact_id,
        sha256=digest,
    )


def run_validation_splits(config_path: str | Path) -> ManifestV2:
    """Publish validation, fixed-eval, and train-candidate source splits."""

    resolved_config, config, logical_config = _load_config(config_path)
    with _split_output_lock(
        config["output_dir"],
        config_path=resolved_config,
        mode="validation",
    ):
        return _run_validation_splits_locked(config, logical_config)


def _run_validation_splits_locked(
    config: Mapping[str, Any],
    logical_config: Mapping[str, Any],
) -> ManifestV2:
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
    _require_complete_source_manifest(source_manifest, source_file)
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
    _require_unchanged(snapshots, "validation")

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
        parents=[_parent(source_manifest, snapshots[source_manifest_path])],
        config=logical_config,
        global_seed=config["seed"],
        seed_derivation_version=SEED_DERIVATION_VERSION,
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
    _require_unchanged(snapshots, "validation")
    publish_manifest(manifest_path, manifest)
    return manifest


def _validation_ids(
    manifest: ManifestV2,
    validation_file: ArtifactFile,
    validation_rows: list[dict[str, Any]],
    configured_size: int,
) -> set[str]:
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
    counts = manifest.stage_metadata.get("counts")
    if not isinstance(counts, Mapping):
        raise ValueError("validation manifest counts must be an object")
    validation_count = _metadata_count(
        counts, "validation", "validation counts"
    )
    if validation_count != configured_size:
        raise ValueError(
            "validation counts.validation must equal configured val_size"
        )
    if len(values) != validation_count:
        raise ValueError(
            "validation selected_ids count must equal counts.validation"
        )
    if validation_file.row_count != validation_count:
        raise ValueError(
            "val_200 file row_count must equal counts.validation"
        )
    if len(validation_rows) != validation_count:
        raise ValueError(
            "val_200 actual row count must equal counts.validation"
        )
    row_ids = {row["id"] for row in validation_rows}
    if set(values) != row_ids:
        raise ValueError(
            "validation selected_ids must match val_200 row ids"
        )
    return set(values)


def run_accepted_splits(config_path: str | Path) -> ManifestV2:
    """Publish independently sampled SFT and GRPO splits from accepted rows."""

    resolved_config, config, logical_config = _load_config(config_path)
    with _split_output_lock(
        config["output_dir"],
        config_path=resolved_config,
        mode="accepted",
    ):
        return _run_accepted_splits_locked(config, logical_config)


def _run_accepted_splits_locked(
    config: Mapping[str, Any],
    logical_config: Mapping[str, Any],
) -> ManifestV2:
    accepted_path = config["teacher_accepted"]
    teacher_manifest_path = config["teacher_manifest"]
    output_dir = config["output_dir"]
    validation_manifest_path = output_dir / "validation_manifest.json"
    validation_path = output_dir / "val_200.jsonl"
    snapshots = _snapshot(
        [
            accepted_path,
            teacher_manifest_path,
            validation_manifest_path,
            validation_path,
        ]
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
    accepted_count, _ = _teacher_completion_counts(
        teacher_manifest, teacher_file
    )
    validation_manifest = load_manifest(validation_manifest_path)
    validation_file = _manifest_file(
        validation_manifest,
        validation_path,
        expected_schema=SOURCE_FIELD_SCHEMA,
    )
    validation_rows = read_jsonl_strict(
        validation_path, validate_normalized_source
    )
    _require_unique_ids(validation_rows, "validation")
    _require_row_count(validation_file, validation_rows)
    val_ids = _validation_ids(
        validation_manifest,
        validation_file,
        validation_rows,
        config["val_size"],
    )
    teacher_manifest.require_parent(
        validation_manifest.artifact_id,
        snapshots[validation_manifest_path],
    )
    rows = read_jsonl_strict(accepted_path, validate_sft_record)
    _require_unique_ids(rows, "accepted")
    _require_row_count(teacher_file, rows)
    if len(rows) != accepted_count:
        raise ValueError(
            "teacher accepted_count must equal actual accepted rows"
        )
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
    _require_unchanged(snapshots, "accepted")

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
            _parent(teacher_manifest, snapshots[teacher_manifest_path]),
            _parent(
                validation_manifest,
                snapshots[validation_manifest_path],
            ),
        ],
        config=logical_config,
        global_seed=config["seed"],
        seed_derivation_version=SEED_DERIVATION_VERSION,
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
    _require_unchanged(snapshots, "accepted")
    publish_manifest(manifest_path, manifest)
    return manifest
