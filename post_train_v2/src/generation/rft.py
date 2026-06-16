"""RFT rollout expansion and response selection."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from post_train_v2.src.artifacts.atomic import publish_json, publish_jsonl
from post_train_v2.src.config.loading import load_yaml, resolve_repo_path
from post_train_v2.src.countdown.validation import (
    serialize_fraction,
    validate_countdown_response,
)
from post_train_v2.src.data.schema import (
    NORMALIZED_SOURCE_KEYS,
    validate_normalized_source,
    validate_sft_record,
)
from post_train_v2.src.data.splits import read_jsonl_strict
from post_train_v2.src.generation.parallel_vllm import (
    ParallelVLLMEngine,
    PositionedPrompt,
    WorkerSpec,
)
from post_train_v2.src.generation.seeding import derive_request_seed


@dataclass(frozen=True)
class RFTRequest:
    position: int
    prompt: str
    seed: int
    metadata: dict[str, int | str]


def build_rollout_requests(
    source_rows: Sequence[Mapping[str, Any]],
    *,
    rollouts_per_prompt: int,
    seed: int,
) -> list[RFTRequest]:
    if type(rollouts_per_prompt) is not int or rollouts_per_prompt <= 0:
        raise ValueError("rollouts_per_prompt must be a positive integer")
    requests: list[RFTRequest] = []
    for source_position, row in enumerate(source_rows):
        for rollout_index in range(rollouts_per_prompt):
            position = source_position * rollouts_per_prompt + rollout_index
            row_id = str(row["id"])
            requests.append(
                RFTRequest(
                    position=position,
                    prompt=str(row["prompt"]),
                    seed=derive_request_seed(seed, "rft", row_id, rollout_index),
                    metadata={
                        "id": row_id,
                        "source_index": int(row["source_index"]),
                        "rollout_index": rollout_index,
                    },
                )
            )
    return requests


def normalize_rollout_sources(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        if set(row) == NORMALIZED_SOURCE_KEYS:
            normalized.append(validate_normalized_source(row))
        else:
            sft_row = validate_sft_record(row)
            normalized.append(
                {key: sft_row[key] for key in NORMALIZED_SOURCE_KEYS}
            )
    return normalized


def select_rft_rows(
    source_rows: Sequence[Mapping[str, Any]],
    positioned_responses: Sequence[tuple[int, str]],
    *,
    rollouts_per_prompt: int,
    max_correct_per_prompt: int = 2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sources = list(source_rows)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_text: dict[str, set[str]] = defaultdict(set)
    accepted_counts: dict[str, int] = defaultdict(int)

    for position, raw_response in sorted(positioned_responses):
        source_position = position // rollouts_per_prompt
        rollout_index = position % rollouts_per_prompt
        source = dict(sources[source_position])
        response = _normalize_response(raw_response)
        if response in seen_text[source["id"]]:
            continue
        seen_text[source["id"]].add(response)
        validation = validate_countdown_response(
            response,
            list(source["numbers"]),
            int(source["target"]),
        )
        row = {
            **source,
            "response": response,
            "validation": _validation_dict(validation),
            "provenance": {
                "stage": "rft_rollout",
                "source_id": source["id"],
                "source_index": source["source_index"],
                "rollout_index": rollout_index,
            },
        }
        if validation.ok and accepted_counts[source["id"]] < max_correct_per_prompt:
            accepted.append(row)
            accepted_counts[source["id"]] += 1
        elif not validation.ok:
            rejected.append(row)
    return accepted, rejected


def run_rft_rollout(config_path: str | Path, *, limit: int | None = None) -> dict[str, Any]:
    config = load_yaml(config_path)
    raw_rows = read_jsonl_strict(
        resolve_repo_path(config["input_path"]),
        lambda row: dict(row),
    )
    source_rows = normalize_rollout_sources(raw_rows)
    if limit is not None:
        source_rows = source_rows[:limit]
    requests = build_rollout_requests(
        source_rows,
        rollouts_per_prompt=int(config["rollouts_per_prompt"]),
        seed=int(config["seed"]),
    )
    worker_specs = [
        WorkerSpec(
            worker_index=index,
            device=int(device),
            cache_root=str(Path(config["cache_root"]) / f"worker-{index}"),
        )
        for index, device in enumerate(config["devices"])
    ]
    with ParallelVLLMEngine(
        model_path=str(resolve_repo_path(config["model_path"])),
        worker_specs=worker_specs,
        gpu_memory_utilization=float(config["gpu_memory_utilization"]),
        max_model_len=int(config["max_model_len"]),
        seed=int(config["seed"]),
        max_new_tokens=int(config["max_new_tokens"]),
        temperature=float(config["temperature"]),
        top_p=float(config["top_p"]),
        enable_thinking=False,
        timeout_seconds=float(config["worker_timeout_seconds"]),
    ).start() as engine:
        positioned_responses = engine.generate(
            1,
            [
                PositionedPrompt(
                    position=request.position,
                    prompt=request.prompt,
                    seed=request.seed,
                )
                for request in requests
            ],
        )

    accepted, rejected = select_rft_rows(
        source_rows,
        positioned_responses,
        rollouts_per_prompt=int(config["rollouts_per_prompt"]),
    )
    output_dir = resolve_repo_path(config["output_dir"])
    publish_jsonl(output_dir / "rft_accepted.jsonl", accepted)
    publish_jsonl(output_dir / "rft_rejected.jsonl", rejected)
    manifest = {
        "stage": "rft_rollout",
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "question_coverage": len({row["id"] for row in accepted}),
        "rollouts_per_prompt": int(config["rollouts_per_prompt"]),
    }
    publish_json(output_dir / "manifest.json", manifest)
    return manifest


def _normalize_response(response: str) -> str:
    return response.replace("\r\n", "\n").replace("\r", "\n").strip()


def _validation_dict(result) -> dict[str, Any]:
    return {
        "ok": result.ok,
        "value": serialize_fraction(result.value),
        "used_numbers": result.used_numbers,
        "expression": result.expression,
        "error": result.error,
    }
