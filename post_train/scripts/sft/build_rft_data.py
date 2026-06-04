from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from post_train.src.countdown.config import load_yaml_config, resolve_path
from post_train.src.countdown.generation import GenerationConfig, VLLMGenerator
from post_train.src.countdown.io import read_jsonl, write_jsonl, write_manifest
from post_train.src.countdown.validation import extract_answer_text, validate_countdown_response


DEFAULT_CONFIG = "post_train/configs/rft.yaml"
REJECTED_FILENAME = "rft_rejected.jsonl"


@dataclass(frozen=True)
class Rollout:
    prompt: str
    source_index: int
    sample_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build accepted RFT data with vLLM rollouts.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--limit", type=int, default=None, help="Limit source prompts for smoke runs.")
    return parser.parse_args()


def validate_source_rows(rows: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    for source_index, row in enumerate(rows):
        row_id = str(row.get("id", "")).strip()
        if not row_id:
            raise ValueError(f"source row {source_index} must contain a non-empty id")
        if row_id in seen:
            raise ValueError(f"source rows contain duplicate id: {row_id}")
        seen.add(row_id)


def build_rollouts(
    rows: list[dict[str, Any]],
    num_samples_per_prompt: int,
) -> list[Rollout]:
    if num_samples_per_prompt < 1:
        raise ValueError("num_samples_per_prompt must be at least 1")

    return [
        Rollout(
            prompt=str(row["prompt"]),
            source_index=source_index,
            sample_index=sample_index,
        )
        for source_index, row in enumerate(rows)
        for sample_index in range(num_samples_per_prompt)
    ]


def classify_rft_responses(
    rows: list[dict[str, Any]],
    rollouts: list[Rollout],
    responses: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for rollout, response in zip(rollouts, responses, strict=True):
        row = rows[rollout.source_index]
        source_id = str(row["id"])
        text = response.strip()
        result = validate_countdown_response(text, row["numbers"], int(row["target"]))
        validation = {
            "ok": result.ok,
            "error": result.error,
            "value": result.value,
        }

        if result.ok:
            accepted.append(
                {
                    "id": f"{source_id}-rft-{rollout.sample_index}",
                    "prompt": row["prompt"],
                    "response": text,
                    "numbers": row["numbers"],
                    "target": row["target"],
                    "source": "rft",
                    "source_id": source_id,
                    "teacher_expr": extract_answer_text(text),
                    "validation": validation,
                }
            )
        else:
            rejected.append(
                {
                    "id": f"{source_id}-rft-{rollout.sample_index}",
                    "prompt": row["prompt"],
                    "response": text,
                    "numbers": row["numbers"],
                    "target": row["target"],
                    "source_index": rollout.source_index,
                    "source_id": source_id,
                    "sample_index": rollout.sample_index,
                    "teacher_expr": extract_answer_text(text),
                    "validation": validation,
                }
            )

    return accepted, rejected


def batched(items: list[Rollout], batch_size: int) -> list[list[Rollout]]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def build_generation_config(cfg: dict[str, Any]) -> GenerationConfig:
    return GenerationConfig(
        max_new_tokens=int(cfg["max_new_tokens"]),
        temperature=float(cfg["temperature"]),
        top_p=float(cfg["top_p"]),
        enable_thinking=bool(cfg.get("enable_thinking", False)),
    )


def main() -> None:
    args = parse_args()

    cfg_path = resolve_path(args.config, REPO_ROOT)
    cfg = load_yaml_config(cfg_path)

    train_prompts_path = resolve_path(cfg["train_prompts"], REPO_ROOT)
    accepted_path = resolve_path(cfg["accepted_output"], REPO_ROOT)
    rejected_path = accepted_path.with_name(REJECTED_FILENAME)
    output_dir = resolve_path(cfg.get("output_dir", accepted_path.parent), REPO_ROOT)
    model_path = resolve_path(cfg["base_model_path"], REPO_ROOT)

    rows = read_jsonl(train_prompts_path)
    if args.limit is not None:
        if args.limit < 0:
            raise ValueError("--limit must be non-negative")
        rows = rows[: args.limit]
    validate_source_rows(rows)

    num_samples_per_prompt = int(cfg["num_samples_per_prompt"])
    rollouts = build_rollouts(rows, num_samples_per_prompt)
    generation_config = build_generation_config(cfg)

    responses: list[str] = []
    if rollouts:
        generator = VLLMGenerator(str(model_path))
        for batch in batched(rollouts, int(cfg.get("batch_size", len(rollouts)))):
            responses.extend(
                generator.generate([rollout.prompt for rollout in batch], generation_config)
            )

    accepted, rejected = classify_rft_responses(rows, rollouts, responses)

    write_jsonl(accepted_path, accepted)
    write_jsonl(rejected_path, rejected)
    write_manifest(
        output_dir / "manifest.json",
        {
            "name": "rft_data",
            "num_input": len(rows),
            "num_rollouts": len(rollouts),
            "num_accepted": len(accepted),
            "num_rejected": len(rejected),
            "num_samples_per_prompt": num_samples_per_prompt,
            "model": str(model_path),
            "base_model_path": str(model_path),
            "max_new_tokens": generation_config.max_new_tokens,
            "enable_thinking": generation_config.enable_thinking,
        },
    )


if __name__ == "__main__":
    main()
