"""Build solver-backed normalized V2 source datasets."""

from __future__ import annotations

import errno
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from post_train_v2.src.artifacts.atomic import publish_jsonl
from post_train_v2.src.artifacts.hashing import (
    sha256_canonical_json,
    sha256_file,
)
from post_train_v2.src.artifacts.manifest import (
    ArtifactFile,
    ManifestV2,
    ParentArtifact,
    publish_manifest,
)
from post_train_v2.src.config.loading import load_yaml, require_keys, resolve_repo_path
from post_train_v2.src.countdown.bucketing import assign_bucket
from post_train_v2.src.countdown.prompts import build_solution_prompt
from post_train_v2.src.countdown.solver import solve_countdown
from post_train_v2.src.data.schema import validate_normalized_source


NORMALIZED_SOURCE_FIELD_SCHEMA = {
    "id": "string",
    "source_index": "integer",
    "numbers": "array[integer]",
    "target": "integer",
    "gold_expr": "string",
    "prompt": "string",
    "bucket": "object",
}
UNSOLVED_SOURCE_FIELD_SCHEMA = {
    "id": "string",
    "source_index": "integer",
    "numbers": "array[integer]",
    "target": "integer",
    "reason": "string",
}


def _exact_nonnegative_int(value: Any, name: str, *, positive: bool = False) -> int:
    if type(value) is int:
        normalized = value
    elif isinstance(value, np.integer):
        normalized = value.item()
    else:
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be a {qualifier} exact integer")
    if normalized < (1 if positive else 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be a {qualifier} exact integer")
    return normalized


def _normalize_numbers(value: Any, name: str = "numbers") -> list[int]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(f"{name} must be a JSON list of exact integers") from error
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list, tuple, numpy array, or JSON list")
    if not value:
        raise ValueError(f"{name} must be nonempty")
    return [
        _exact_nonnegative_int(item, f"{name}[{index}]")
        for index, item in enumerate(value)
    ]


def _build_solved(
    *,
    row_id: str,
    source_index: int,
    numbers: list[int],
    target: int,
    expression: str,
) -> dict[str, Any]:
    return validate_normalized_source(
        {
            "id": row_id,
            "source_index": source_index,
            "numbers": numbers,
            "target": target,
            "gold_expr": expression,
            "prompt": build_solution_prompt(numbers, target),
            "bucket": assign_bucket(numbers, expression),
        }
    )


def _frame_numbers_column(frame: pd.DataFrame) -> str:
    if "nums" in frame.columns:
        return "nums"
    if "numbers" in frame.columns:
        return "numbers"
    raise ValueError("training frame must contain a nums or numbers column")


def build_train_source(
    frame: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Normalize raw train rows, retaining unsolved rows separately."""

    if not isinstance(frame, pd.DataFrame):
        raise ValueError("frame must be a pandas DataFrame")
    if "target" not in frame.columns:
        raise ValueError("training frame must contain a target column")
    numbers_column = _frame_numbers_column(frame)
    solvable: list[dict[str, Any]] = []
    unsolved: list[dict[str, Any]] = []

    for source_index, values in enumerate(
        frame[[numbers_column, "target"]].itertuples(index=False, name=None),
        start=1,
    ):
        raw_numbers, raw_target = values
        numbers = _normalize_numbers(raw_numbers)
        target = _exact_nonnegative_int(raw_target, "target")
        row_id = f"train-{source_index:06d}"
        expression = solve_countdown(numbers, target)
        if expression is None:
            unsolved.append(
                {
                    "id": row_id,
                    "source_index": source_index,
                    "numbers": numbers,
                    "target": target,
                    "reason": "no_solution",
                }
            )
            continue
        solvable.append(
            _build_solved(
                row_id=row_id,
                source_index=source_index,
                numbers=numbers,
                target=target,
                expression=expression,
            )
        )
    return solvable, unsolved


def build_test_source(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Normalize raw test rows and require every row to be solvable."""

    if (
        not isinstance(rows, Sequence)
        or isinstance(rows, (str, bytes, bytearray))
        or isinstance(rows, Mapping)
    ):
        raise ValueError("test rows must be a sequence")
    solved: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for position, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise ValueError(f"test row {position} must be a mapping")
        missing = [
            key for key in ("id", "target") if key not in row
        ]
        if missing or ("numbers" not in row and "nums" not in row):
            missing.append("numbers")
            raise ValueError(
                f"test row {position} missing fields: {', '.join(sorted(set(missing)))}"
            )
        native_id = _exact_nonnegative_int(row["id"], "id", positive=True)
        if native_id in seen_ids:
            raise ValueError(f"duplicate test id: {native_id}")
        seen_ids.add(native_id)
        numbers = _normalize_numbers(
            row["numbers"] if "numbers" in row else row["nums"]
        )
        target = _exact_nonnegative_int(row["target"], "target")
        row_id = f"test-{native_id:06d}"
        expression = solve_countdown(numbers, target)
        if expression is None:
            raise ValueError(f"{row_id} has no solution")
        solved.append(
            _build_solved(
                row_id=row_id,
                source_index=native_id,
                numbers=numbers,
                target=target,
                expression=expression,
            )
        )
    return solved


def _load_test_rows(path: Path) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"test input contains invalid JSON: {path}") from error
    if not isinstance(value, list):
        raise ValueError("test input must contain a JSON list")
    return value


def _validate_limit(limit: int | None) -> int | None:
    if limit is None:
        return None
    if type(limit) is not int or limit <= 0:
        raise ValueError("limit must be a positive exact integer or None")
    return limit


def _artifact_file(
    output_dir: Path,
    filename: str,
    row_count: int,
    field_schema: dict[str, str],
) -> ArtifactFile:
    path = output_dir / filename
    return ArtifactFile(
        relative_path=filename,
        sha256=sha256_file(path),
        byte_size=path.stat().st_size,
        row_count=row_count,
        field_schema=field_schema,
    )


def _raw_parent(kind: str, path: Path) -> ParentArtifact:
    digest = sha256_file(path)
    return ParentArtifact(
        artifact_id=sha256_canonical_json(
            {"kind": kind, "sha256": digest}
        ),
        sha256=digest,
    )


def _logical_manifest_path(value: Any, config_dir: Path, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonempty path string")
    candidate = Path(value)
    if not candidate.is_absolute():
        return candidate.as_posix()
    try:
        return Path(os.path.relpath(candidate, start=config_dir)).as_posix()
    except ValueError as error:
        raise ValueError(
            f"{field} absolute path must share a filesystem root with config"
        ) from error


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


def _revoke_completion_manifest(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    _fsync_directory(path.parent)


def run_build_source(
    config_path: str | Path,
    limit: int | None = None,
) -> ManifestV2:
    """Build and publish normalized source files, with manifest published last."""

    limit = _validate_limit(limit)
    resolved_config_path = resolve_repo_path(config_path)
    config = load_yaml(resolved_config_path)
    require_keys(config, "seed", "train_input", "test_input", "output_dir")
    if set(config) != {"seed", "train_input", "test_input", "output_dir"}:
        raise ValueError("build_source config has unexpected keys")
    seed = _exact_nonnegative_int(config["seed"], "seed")
    train_input = resolve_repo_path(config["train_input"])
    test_input = resolve_repo_path(config["test_input"])
    output_dir = resolve_repo_path(config["output_dir"])
    logical_train_input = _logical_manifest_path(
        config["train_input"],
        resolved_config_path.parent,
        "train_input",
    )
    logical_test_input = _logical_manifest_path(
        config["test_input"],
        resolved_config_path.parent,
        "test_input",
    )
    logical_output = _logical_manifest_path(
        config["output_dir"],
        resolved_config_path.parent,
        "output_dir",
    )

    train_frame = pd.read_parquet(train_input)
    if limit is not None:
        train_frame = train_frame.iloc[:limit]
    solvable_train, unsolved_train = build_train_source(train_frame)
    test_solved = build_test_source(_load_test_rows(test_input))

    output_dir.mkdir(parents=True, exist_ok=True)
    _revoke_completion_manifest(output_dir / "manifest.json")
    publish_jsonl(output_dir / "source_all.jsonl", solvable_train)
    publish_jsonl(output_dir / "solvable_train.jsonl", solvable_train)
    publish_jsonl(output_dir / "unsolved_train.jsonl", unsolved_train)
    publish_jsonl(output_dir / "test_solved.jsonl", test_solved)

    files = [
        _artifact_file(
            output_dir,
            "source_all.jsonl",
            len(solvable_train),
            NORMALIZED_SOURCE_FIELD_SCHEMA,
        ),
        _artifact_file(
            output_dir,
            "solvable_train.jsonl",
            len(solvable_train),
            NORMALIZED_SOURCE_FIELD_SCHEMA,
        ),
        _artifact_file(
            output_dir,
            "unsolved_train.jsonl",
            len(unsolved_train),
            UNSOLVED_SOURCE_FIELD_SCHEMA,
        ),
        _artifact_file(
            output_dir,
            "test_solved.jsonl",
            len(test_solved),
            NORMALIZED_SOURCE_FIELD_SCHEMA,
        ),
    ]
    manifest = ManifestV2.build(
        artifact_type="dataset",
        stage="build_source",
        files=files,
        parents=[
            _raw_parent("raw_train", train_input),
            _raw_parent("raw_test", test_input),
        ],
        config=config,
        global_seed=seed,
        stage_metadata={
            "completed": True,
            "counts": {
                "train_input": len(train_frame),
                "solvable_train": len(solvable_train),
                "unsolved_train": len(unsolved_train),
                "test_solved": len(test_solved),
            },
            "limit": limit,
            "inputs": {
                "train": logical_train_input,
                "test": logical_test_input,
            },
            "output": logical_output,
        },
    )
    publish_manifest(output_dir / "manifest.json", manifest)
    return manifest
