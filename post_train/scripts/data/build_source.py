from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from post_train.src.countdown.bucketing import assign_bucket
from post_train.src.countdown.config import load_yaml_config, resolve_path
from post_train.src.countdown.io import write_jsonl, write_manifest
from post_train.src.countdown.prompts import build_solution_prompt
from post_train.src.countdown.sampling import stratified_sample
from post_train.src.countdown.solver import solve_countdown


DEFAULT_CONFIG = "post_train/configs/data_build.yaml"

ALIASES = {
    "train_source.jsonl": "train_pool.jsonl",
    "val.jsonl": "val_200.jsonl",
    "eval_subset.jsonl": "val_eval_50.jsonl",
    "test.jsonl": "test_with_solver_answers.jsonl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Countdown source data warehouse files.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def normalize_numbers(value: Any) -> list[int]:
    if isinstance(value, str):
        parsed = json.loads(value)
        return normalize_numbers(parsed)

    try:
        values = value.tolist()
    except AttributeError:
        values = value

    if not isinstance(values, (list, tuple)):
        raise ValueError(f"numbers must be a sequence, got {type(value).__name__}")

    return [int(number) for number in values]


def get_numbers(row: dict[str, Any]) -> list[int]:
    if "numbers" in row:
        return normalize_numbers(row["numbers"])
    if "nums" in row:
        return normalize_numbers(row["nums"])
    raise ValueError("row is missing numbers/nums")


def build_solved_record(
    *,
    row_id: str,
    source_index: int,
    numbers: list[int],
    target: int,
    gold_expr: str,
) -> dict[str, Any]:
    bucket = assign_bucket(numbers, gold_expr)
    return {
        "id": row_id,
        "source_index": source_index,
        "numbers": numbers,
        "target": target,
        "gold_expr": gold_expr,
        "prompt": build_solution_prompt(numbers, target),
        "bucket": bucket,
    }


def build_train_rows(train_df: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_rows: list[dict[str, Any]] = []
    unsolved_rows: list[dict[str, Any]] = []

    for row_position, row in enumerate(train_df.to_dict(orient="records"), start=1):
        numbers = get_numbers(row)
        target = int(row["target"])
        gold_expr = solve_countdown(numbers, target)

        if gold_expr is None:
            unsolved_rows.append(
                {
                    "source_index": row_position,
                    "numbers": numbers,
                    "target": target,
                }
            )
            continue

        source_rows.append(
            build_solved_record(
                row_id=f"train-{row_position:06d}",
                source_index=row_position,
                numbers=numbers,
                target=target,
                gold_expr=gold_expr,
            )
        )

    return source_rows, unsolved_rows


def read_test_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"test input must contain a JSON array: {path}")
    return data


def build_test_rows(test_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    solved_rows: list[dict[str, Any]] = []

    for row in test_rows:
        raw_id = int(row["id"])
        numbers = get_numbers(row)
        target = int(row["target"])
        gold_expr = solve_countdown(numbers, target)
        if gold_expr is None:
            raise ValueError(f"unsolved test row id={raw_id}")

        solved_rows.append(
            build_solved_record(
                row_id=f"test-{raw_id:06d}",
                source_index=raw_id,
                numbers=numbers,
                target=target,
                gold_expr=gold_expr,
            )
        )

    return solved_rows


def main() -> None:
    args = parse_args()
    cfg_path = resolve_path(args.config, REPO_ROOT)
    cfg = load_yaml_config(cfg_path)

    train_input = resolve_path(cfg["train_input"], REPO_ROOT)
    test_input = resolve_path(cfg["test_input"], REPO_ROOT)
    output_dir = resolve_path(cfg["output_dir"], REPO_ROOT)
    seed = int(cfg["seed"])

    train_df = pd.read_parquet(train_input)
    if args.limit > 0:
        train_df = train_df.head(args.limit)

    source_rows, unsolved_rows = build_train_rows(train_df)
    val_rows = stratified_sample(source_rows, int(cfg["val_size"]), seed)
    val_ids = {row["id"] for row in val_rows}
    train_pool_rows = [row for row in source_rows if row["id"] not in val_ids]
    eval_subset_rows = stratified_sample(val_rows, int(cfg["eval_subset_size"]), seed + 1)
    test_rows = build_test_rows(read_test_rows(test_input))

    write_jsonl(output_dir / "source_all.jsonl", source_rows)
    write_jsonl(output_dir / "train_pool.jsonl", train_pool_rows)
    write_jsonl(output_dir / "val_200.jsonl", val_rows)
    write_jsonl(output_dir / "val_eval_50.jsonl", eval_subset_rows)
    write_jsonl(output_dir / "test_with_solver_answers.jsonl", test_rows)
    write_jsonl(output_dir / "unsolved_train.jsonl", unsolved_rows)
    write_jsonl(output_dir / "train_source.jsonl", train_pool_rows)
    write_jsonl(output_dir / "val.jsonl", val_rows)
    write_jsonl(output_dir / "eval_subset.jsonl", eval_subset_rows)
    write_jsonl(output_dir / "test.jsonl", test_rows)
    write_manifest(
        output_dir / "manifest.json",
        {
            "name": "data_build",
            "num_source": len(source_rows),
            "num_train_pool": len(train_pool_rows),
            "num_val": len(val_rows),
            "num_eval_subset": len(eval_subset_rows),
            "num_test": len(test_rows),
            "num_unsolved": len(unsolved_rows),
            "seed": seed,
            "aliases": ALIASES,
        },
    )


if __name__ == "__main__":
    main()
