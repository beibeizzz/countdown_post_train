"""Sidecar utilities for aggregating GRPO rollout metrics from JSONL dumps."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from post_train_v2.src.evaluation.grpo_metrics import aggregate_grpo_metrics


def aggregate_jsonl_file(
    input_path: Path,
    *,
    output_path: Path | None = None,
    group_size: int,
) -> dict[str, float | int]:
    rows = _read_jsonl(input_path)
    metrics = aggregate_grpo_metrics(
        rewards=[_require_number(row, "reward") for row in rows],
        group_size=group_size,
        response_lengths=[_require_int(row, "response_length") for row in rows],
        truncated=[_require_bool(row, "truncated") for row in rows],
        diagnostics=[_diagnostics(row) for row in rows],
    )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(metrics, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
    return metrics


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group-size", type=int, required=True)
    args = parser.parse_args(argv)

    aggregate_jsonl_file(args.input, output_path=args.output, group_size=args.group_size)
    return 0


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        rows.append(value)
    return rows


def _diagnostics(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "answer_correct": row.get("answer_correct"),
        "format_ok": row.get("format_ok"),
        "bucket": row.get("bucket"),
    }


def _require_number(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key)
    if type(value) not in {int, float}:
        raise ValueError(f"{key} must be numeric")
    return float(value)


def _require_int(row: Mapping[str, Any], key: str) -> int:
    value = row.get(key)
    if type(value) is not int or value < 0:
        raise ValueError(f"{key} must be a nonnegative exact integer")
    return value


def _require_bool(row: Mapping[str, Any], key: str) -> bool:
    value = row.get(key)
    if type(value) is not bool:
        raise ValueError(f"{key} must be a boolean")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
