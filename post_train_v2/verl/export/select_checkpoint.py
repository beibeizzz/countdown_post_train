"""Select best and final GRPO actor checkpoints from native validation dumps."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from post_train_v2.src.artifacts.atomic import publish_json
from post_train_v2.src.artifacts.hashing import sha256_canonical_json, sha256_file
from post_train_v2.src.evaluation.scoring import aggregate_rows, score_response

STEP_DUMP_RE = re.compile(r"^step_(?P<step>\d+)\.jsonl$")
CHECKPOINT_RE = re.compile(r"^global_step_(?P<step>\d+)$")


def select_grpo_checkpoints(
    run_dir: str | Path,
    *,
    config: Mapping[str, Any],
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    base = Path(run_dir)
    dumps = _discover_dumps(base)
    checkpoints = _discover_checkpoints(base)
    _require_matching_steps(dumps, checkpoints)

    candidates = [
        _candidate(step, dumps[step], checkpoints[step]) for step in sorted(dumps)
    ]
    best = min(
        candidates,
        key=lambda item: (
            -float(item["metrics"]["accuracy"]),
            -float(item["metrics"]["format_rate"]),
            int(item["step"]),
        ),
    )
    final = max(candidates, key=lambda item: int(item["step"]))

    selection = {
        "config_sha256": sha256_canonical_json(dict(config)),
        "selected_best_step": best["step"],
        "final_step": final["step"],
        "best_checkpoint_path": best["checkpoint_path"],
        "final_checkpoint_path": final["checkpoint_path"],
        "candidates": candidates,
    }
    destination = Path(output_path) if output_path is not None else base / "export" / "selection.json"
    publish_json(destination, selection)
    return selection


def _candidate(step: int, dump_path: Path, checkpoint_path: Path) -> dict[str, Any]:
    scored = []
    for row in _read_jsonl(dump_path):
        scored.append(
            score_response(
                row,
                _require_string(row, "raw_generation"),
                generated_tokens=_require_nonnegative_int(row, "generated_tokens"),
                truncated=_require_bool(row, "truncated"),
            )
        )
    return {
        "step": step,
        "checkpoint_path": str(checkpoint_path),
        "validation_dump_path": str(dump_path),
        "validation_dump_sha256": sha256_file(dump_path),
        "metrics": aggregate_rows(scored),
    }


def _discover_dumps(run_dir: Path) -> dict[int, Path]:
    validation_dir = run_dir / "validation"
    if not validation_dir.is_dir():
        raise FileNotFoundError(f"missing validation directory: {validation_dir}")
    dumps: dict[int, Path] = {}
    for path in validation_dir.iterdir():
        match = STEP_DUMP_RE.match(path.name)
        if match and path.is_file():
            dumps[int(match.group("step"))] = path
    if not dumps:
        raise ValueError(f"no validation dumps found in {validation_dir}")
    return dumps


def _discover_checkpoints(run_dir: Path) -> dict[int, Path]:
    roots = [run_dir / "checkpoints", run_dir]
    checkpoints: dict[int, Path] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.iterdir():
            match = CHECKPOINT_RE.match(path.name)
            if not match or not path.is_dir():
                continue
            actor = path / "actor"
            if actor.is_dir():
                checkpoints[int(match.group("step"))] = actor
    if not checkpoints:
        raise ValueError(f"no actor checkpoints found in {run_dir}")
    return checkpoints


def _require_matching_steps(dumps: Mapping[int, Path], checkpoints: Mapping[int, Path]) -> None:
    dump_steps = set(dumps)
    checkpoint_steps = set(checkpoints)
    missing_checkpoints = sorted(dump_steps - checkpoint_steps)
    missing_dumps = sorted(checkpoint_steps - dump_steps)
    if missing_checkpoints:
        raise ValueError(f"missing checkpoint for validation steps: {missing_checkpoints}")
    if missing_dumps:
        raise ValueError(f"missing validation dump for checkpoint steps: {missing_dumps}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        rows.append(value)
    if not rows:
        raise ValueError(f"validation dump is empty: {path}")
    return rows


def _require_string(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _require_nonnegative_int(row: Mapping[str, Any], key: str) -> int:
    value = row.get(key)
    if type(value) is not int or value < 0:
        raise ValueError(f"{key} must be a nonnegative exact integer")
    return value


def _require_bool(row: Mapping[str, Any], key: str) -> bool:
    value = row.get(key)
    if type(value) is not bool:
        raise ValueError(f"{key} must be a boolean")
    return value
