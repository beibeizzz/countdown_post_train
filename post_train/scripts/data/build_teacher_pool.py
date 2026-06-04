from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from post_train.src.countdown.config import load_yaml_config, resolve_path
from post_train.src.countdown.generation import GenerationConfig, VLLMGenerator
from post_train.src.countdown.io import read_jsonl, write_jsonl, write_manifest
from post_train.src.countdown.validation import extract_answer_text, validate_countdown_response


DEFAULT_CONFIG = "post_train/configs/teacher_rollout.yaml"
DEFAULT_INPUT = "post_train/data/processed/train_pool.jsonl"
OUTPUT_DIR = "post_train/data/teacher_rollouts"
ACCEPTED_FILENAME = "teacher_accepted_20k.jsonl"
REJECTED_FILENAME = "teacher_rejected.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build teacher accepted pool.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    return parser.parse_args()


def collect_processed_ids(
    accepted: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
) -> set[Any]:
    validate_unique_ids("teacher rollout outputs", accepted + rejected)
    return {row["id"] for row in accepted + rejected}


def validate_unique_ids(name: str, rows: list[dict[str, Any]]) -> None:
    seen: set[Any] = set()
    for row in rows:
        row_id = row["id"]
        if row_id in seen:
            raise ValueError(f"{name} contains duplicate id: {row_id}")
        seen.add(row_id)


def validate_resume_state(
    accepted: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    target: int,
) -> None:
    validate_unique_ids("teacher rollout outputs", accepted + rejected)
    if len(accepted) > target:
        raise ValueError(
            f"existing accepted rows count {len(accepted)} exceeds target {target}"
        )


def validate_source_rows(source_rows: list[dict[str, Any]]) -> None:
    validate_unique_ids("source input", source_rows)


def build_teacher_payload(row: dict[str, Any], response: str) -> dict[str, Any]:
    text = response.strip()
    result = validate_countdown_response(text, row["numbers"], int(row["target"]))
    return {
        **row,
        "response": text,
        "teacher_expr": extract_answer_text(text),
        "validation": {
            "ok": result.ok,
            "error": result.error,
            "value": result.value,
        },
    }


def process_teacher_responses(
    *,
    rows: list[dict[str, Any]],
    responses: list[str],
    accepted: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    processed_ids: set[Any],
    target: int,
) -> None:
    for row, response in zip(rows, responses, strict=True):
        if len(accepted) >= target:
            break

        payload = build_teacher_payload(row, response)
        if payload["validation"]["ok"]:
            accepted.append(payload)
        else:
            rejected.append(payload)
        processed_ids.add(row["id"])


def batched(rows: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    return [rows[index : index + batch_size] for index in range(0, len(rows), batch_size)]


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temp_path = path.with_name(f"{path.name}.tmp")
    write_jsonl(temp_path, rows)
    temp_path.replace(path)


def atomic_write_manifest(path: Path, payload: dict[str, Any]) -> None:
    temp_path = path.with_name(f"{path.name}.tmp")
    write_manifest(temp_path, payload)
    temp_path.replace(path)


def main() -> None:
    args = parse_args()

    cfg_path = resolve_path(args.config, REPO_ROOT)
    cfg = load_yaml_config(cfg_path)
    input_path = resolve_path(args.input, REPO_ROOT)
    model_path = resolve_path(cfg["model_path"], REPO_ROOT)
    output_dir = resolve_path(OUTPUT_DIR, REPO_ROOT)

    accepted_path = output_dir / ACCEPTED_FILENAME
    rejected_path = output_dir / REJECTED_FILENAME

    source_rows = read_jsonl(input_path)
    validate_source_rows(source_rows)
    accepted = read_jsonl(accepted_path) if accepted_path.exists() else []
    rejected = read_jsonl(rejected_path) if rejected_path.exists() else []

    target = int(cfg["stop_after_accepted"])
    validate_resume_state(accepted, rejected, target)
    processed_ids = collect_processed_ids(accepted, rejected)
    generation_config = GenerationConfig(
        max_new_tokens=int(cfg["max_new_tokens"]),
        temperature=float(cfg["temperature"]),
        top_p=float(cfg["top_p"]),
        enable_thinking=bool(cfg.get("enable_thinking", False)),
    )

    remaining_rows = [
        row
        for row in source_rows
        if row["id"] not in processed_ids
    ]

    if len(accepted) < target and remaining_rows:
        generator = VLLMGenerator(str(model_path))
        for batch in batched(remaining_rows, int(cfg["batch_size"])):
            if len(accepted) >= target:
                break

            prompts = [row["prompt"] for row in batch]
            responses = generator.generate(prompts, generation_config)
            process_teacher_responses(
                rows=batch,
                responses=responses,
                accepted=accepted,
                rejected=rejected,
                processed_ids=processed_ids,
                target=target,
            )

            atomic_write_jsonl(accepted_path, accepted[:target])
            atomic_write_jsonl(rejected_path, rejected)

    atomic_write_jsonl(accepted_path, accepted[:target])
    atomic_write_jsonl(rejected_path, rejected)
    atomic_write_manifest(
        output_dir / "manifest.json",
        {
            "name": "teacher_accepted_pool",
            "model": str(model_path),
            "num_accepted": len(accepted[:target]),
            "num_rejected": len(rejected),
            "max_new_tokens": generation_config.max_new_tokens,
            "enable_thinking": generation_config.enable_thinking,
        },
    )


if __name__ == "__main__":
    main()
