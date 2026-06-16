"""DPO rejected candidate classification and pair selection."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Literal

from post_train_v2.src.artifacts.atomic import publish_jsonl
from post_train_v2.src.artifacts.hashing import sha256_file
from post_train_v2.src.artifacts.manifest import ArtifactFile, ManifestV2, publish_manifest
from post_train_v2.src.config.loading import load_yaml, resolve_repo_path
from post_train_v2.src.countdown.prompts import build_dpo_forced_wrong_prompt
from post_train_v2.src.countdown.validation import (
    serialize_fraction,
    validate_countdown_response,
)
from post_train_v2.src.data.schema import DPO_REJECTED_CATEGORIES, validate_dpo_record, validate_sft_record
from post_train_v2.src.data.splits import read_jsonl_strict
from post_train_v2.src.generation.parallel_vllm import ParallelVLLMEngine, PositionedPrompt, WorkerSpec
from post_train_v2.src.generation.seeding import derive_request_seed
from post_train_v2.src.generation.metadata import GenerationRecord

GenerationRoute = Literal["forced_wrong", "high_temp"]
RejectedCategory = Literal[
    "wrong_value",
    "number_mismatch",
    "invalid_expression",
    "missing_answer_tag",
    "truncated",
    "unexpected_correct",
]
ELIGIBLE_CATEGORIES = {
    "wrong_value",
    "number_mismatch",
    "invalid_expression",
    "missing_answer_tag",
    "truncated",
}
CATEGORY_PRIORITY = [
    "wrong_value",
    "number_mismatch",
    "invalid_expression",
    "missing_answer_tag",
    "truncated",
]
QUOTA_WEIGHTS = {
    "wrong_value": 0.70,
    "number_mismatch": 0.15,
    "invalid_expression": 0.10,
    "missing_answer_tag": 0.03,
    "truncated": 0.02,
}
DPO_CANDIDATE_SCHEMA = {
    "source_id": "string",
    "candidate_id": "string",
    "generation_route": "string",
    "rejected": "string",
    "rejected_category": "string",
    "validation": "object",
    "rollout_index": "integer",
}
DPO_PAIR_SCHEMA = {
    "prompt": "string",
    "chosen": "string",
    "rejected": "string",
    "rejected_category": "string",
    "generation_route": "string",
    "provenance": "object",
}


@dataclass(frozen=True)
class DPOCandidate:
    source_id: str
    candidate_id: str
    generation_route: GenerationRoute
    rejected: str
    rejected_category: RejectedCategory
    validation: dict[str, Any]
    rollout_index: int

    @property
    def eligible(self) -> bool:
        return self.rejected_category in ELIGIBLE_CATEGORIES


@dataclass(frozen=True)
class DPOSelectionResult:
    pairs: list[dict[str, Any]]
    quotas: dict[str, int]
    category_counts: dict[str, int]
    route_counts: dict[str, int]
    shortfall: int


@dataclass(frozen=True)
class DPOGenerationRequest:
    position: int
    prompt: str
    seed: int
    metadata: dict[str, Any]


def classify_dpo_candidate(
    *,
    source: dict[str, Any],
    record: GenerationRecord,
    generation_route: GenerationRoute,
    rollout_index: int,
) -> DPOCandidate:
    if generation_route not in {"forced_wrong", "high_temp"}:
        raise ValueError("generation_route must be forced_wrong or high_temp")
    if type(rollout_index) is not int or rollout_index < 0:
        raise ValueError("rollout_index must be a nonnegative integer")
    source_id = str(source["id"])
    candidate_id = f"{source_id}:{generation_route}:{rollout_index}"

    if record.truncated:
        category: RejectedCategory = "truncated"
        validation = {
            "ok": False,
            "value": None,
            "used_numbers": [],
            "expression": None,
            "error": "truncated",
        }
    else:
        result = validate_countdown_response(
            record.text,
            list(source["numbers"]),
            int(source["target"]),
        )
        if result.ok:
            category = "unexpected_correct"
        else:
            category = result.error  # type: ignore[assignment]
        validation = {
            "ok": result.ok,
            "value": serialize_fraction(result.value),
            "used_numbers": result.used_numbers,
            "expression": result.expression,
            "error": category,
        }

    return DPOCandidate(
        source_id=source_id,
        candidate_id=candidate_id,
        generation_route=generation_route,
        rejected=record.text,
        rejected_category=category,
        validation=validation,
        rollout_index=rollout_index,
    )


def validate_chosen_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chosen_rows = []
    for row in rows:
        try:
            chosen_rows.append(validate_sft_record(row))
        except ValueError as error:
            raise ValueError("chosen response must validate") from error
    for row in chosen_rows:
        result = validate_countdown_response(
            row["response"],
            row["numbers"],
            row["target"],
        )
        if not result.ok:
            raise ValueError(f"chosen response must validate: {row['id']}")
    return chosen_rows


def build_dpo_generation_requests(
    rows: list[dict[str, Any]],
    *,
    rollouts_per_route: int,
    seed: int,
) -> list[DPOGenerationRequest]:
    if type(rollouts_per_route) is not int or rollouts_per_route <= 0:
        raise ValueError("rollouts_per_route must be a positive integer")
    requests: list[DPOGenerationRequest] = []
    position = 0
    for row in rows:
        for route in ("forced_wrong", "high_temp"):
            for rollout_index in range(rollouts_per_route):
                prompt = (
                    build_dpo_forced_wrong_prompt(
                        row["numbers"],
                        row["target"],
                        row["response"],
                    )
                    if route == "forced_wrong"
                    else row["prompt"]
                )
                requests.append(
                    DPOGenerationRequest(
                        position=position,
                        prompt=prompt,
                        seed=derive_request_seed(seed, "dpo", row["id"], position),
                        metadata={
                            "source_id": row["id"],
                            "generation_route": route,
                            "rollout_index": rollout_index,
                        },
                    )
                )
                position += 1
    return requests


def run_build_dpo_data(config_path: str | Path, *, limit: int | None = None) -> ManifestV2:
    config = load_yaml(config_path)
    rows = read_jsonl_strict(resolve_repo_path(config["input_path"]), validate_sft_record)
    if limit is not None:
        rows = rows[:limit]
    rows = validate_chosen_rows(rows)
    requests = build_dpo_generation_requests(
        rows,
        rollouts_per_route=int(config["rollouts_per_route"]),
        seed=int(config["seed"]),
    )
    by_position = {request.position: request for request in requests}
    worker_specs = [
        WorkerSpec(index, int(device), str(Path(config["cache_root"]) / f"worker-{index}"))
        for index, device in enumerate(config["devices"])
    ]
    with ParallelVLLMEngine(
        model_path=str(resolve_repo_path(config["model_path"])),
        worker_specs=worker_specs,
        gpu_memory_utilization=float(config["gpu_memory_utilization"]),
        max_model_len=int(config["max_model_len"]),
        seed=int(config["seed"]),
        max_new_tokens=int(config["max_new_tokens"]),
        temperature=1.0,
        top_p=float(config["top_p"]),
        enable_thinking=False,
        timeout_seconds=float(config["worker_timeout_seconds"]),
    ).start() as engine:
        positioned_records = engine.generate(
            1,
            [
                PositionedPrompt(
                    request.position,
                    request.prompt,
                    request.seed,
                )
                for request in requests
            ],
            include_metadata=True,
        )
    row_by_id = {row["id"]: row for row in rows}
    candidates = []
    for position, record in positioned_records:
        request = by_position[position]
        source = row_by_id[request.metadata["source_id"]]
        candidate = classify_dpo_candidate(
            source=source,
            record=record,
            generation_route=request.metadata["generation_route"],
            rollout_index=request.metadata["rollout_index"],
        )
        candidates.append(_candidate_to_dict(candidate))
    selection = select_dpo_pairs(
        rows,
        [
            _candidate_from_dict(row)
            for row in candidates
        ],
        target_size=int(config["target_pairs"]),
        seed=int(config["seed"]),
    )
    return publish_dpo_outputs(
        output_dir=resolve_repo_path(config["output_dir"]),
        candidates=candidates,
        pairs=selection.pairs,
        config=config,
        selection_summary={
            "category_counts": selection.category_counts,
            "route_counts": selection.route_counts,
            "shortfall": selection.shortfall,
        },
    )


def publish_dpo_outputs(
    *,
    output_dir: str | Path,
    candidates: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    config: dict[str, Any],
    selection_summary: dict[str, Any],
) -> ManifestV2:
    for candidate in candidates:
        category = candidate["rejected_category"]
        if category not in DPO_REJECTED_CATEGORIES:
            raise ValueError(f"unexpected DPO rejected category: {category}")
    pairs = [validate_dpo_record(pair) for pair in pairs]
    output_dir = Path(output_dir)
    publish_jsonl(output_dir / "dpo_candidates.jsonl", candidates)
    publish_jsonl(output_dir / "dpo_pairs.jsonl", pairs)
    files = (
        _artifact_file(output_dir, "dpo_candidates.jsonl", len(candidates), DPO_CANDIDATE_SCHEMA),
        _artifact_file(output_dir, "dpo_pairs.jsonl", len(pairs), DPO_PAIR_SCHEMA),
    )
    manifest = ManifestV2.build(
        artifact_type="dpo_pairs",
        stage="dpo_pair_build",
        files=files,
        parents=(),
        config=dict(config),
        model_path=str(config.get("model_path")) if config.get("model_path") else None,
        seed_derivation_version="sha256-stage-v1",
        stage_metadata=selection_summary,
    )
    publish_manifest(output_dir / "manifest.json", manifest)
    return manifest


def compute_category_quotas(target_size: int) -> dict[str, int]:
    if type(target_size) is not int or target_size <= 0:
        raise ValueError("target_size must be a positive integer")
    quotas = {
        category: int(target_size * QUOTA_WEIGHTS[category])
        for category in CATEGORY_PRIORITY
    }
    remainder = target_size - sum(quotas.values())
    for category in CATEGORY_PRIORITY:
        if remainder <= 0:
            break
        quotas[category] += 1
        remainder -= 1
    return quotas


def select_dpo_pairs(
    chosen_rows: list[dict[str, Any]],
    candidates: list[DPOCandidate],
    *,
    target_size: int,
    seed: int,
) -> DPOSelectionResult:
    quotas = compute_category_quotas(target_size)
    chosen_by_id = {row["id"]: row for row in chosen_rows}
    selected: list[DPOCandidate] = []
    used_sources: set[str] = set()
    eligible = [
        candidate
        for candidate in candidates
        if candidate.eligible and candidate.source_id in chosen_by_id
    ]

    for category in CATEGORY_PRIORITY:
        needed = quotas[category]
        selected_for_category = _balanced_route_candidates(
            [
                candidate
                for candidate in eligible
                if candidate.rejected_category == category
                and candidate.source_id not in used_sources
            ],
            seed=seed,
        )[:needed]
        selected.extend(selected_for_category)
        used_sources.update(candidate.source_id for candidate in selected_for_category)

    if len(selected) < target_size:
        remaining = [
            candidate
            for category in CATEGORY_PRIORITY
            for candidate in _stable_order(
                [
                    item
                    for item in eligible
                    if item.rejected_category == category
                    and item.source_id not in used_sources
                    and item not in selected
                ],
                seed=seed,
            )
        ]
        for candidate in remaining:
            if len(selected) >= target_size:
                break
            selected.append(candidate)
            used_sources.add(candidate.source_id)

    ordered = sorted(
        selected,
        key=lambda item: (
            CATEGORY_PRIORITY.index(item.rejected_category),
            _stable_key(item, seed),
        ),
    )
    pairs = [_pair_from_candidate(chosen_by_id[item.source_id], item) for item in ordered]
    return DPOSelectionResult(
        pairs=pairs,
        quotas=quotas,
        category_counts=_counts(pair["rejected_category"] for pair in pairs),
        route_counts=_counts(pair["generation_route"] for pair in pairs),
        shortfall=max(0, target_size - len(pairs)),
    )


def _balanced_route_candidates(
    candidates: list[DPOCandidate],
    *,
    seed: int,
) -> list[DPOCandidate]:
    by_route = {
        "forced_wrong": _stable_order(
            [item for item in candidates if item.generation_route == "forced_wrong"],
            seed=seed,
        ),
        "high_temp": _stable_order(
            [item for item in candidates if item.generation_route == "high_temp"],
            seed=seed,
        ),
    }
    result: list[DPOCandidate] = []
    while by_route["forced_wrong"] or by_route["high_temp"]:
        next_route = (
            "forced_wrong"
            if len([item for item in result if item.generation_route == "forced_wrong"])
            <= len([item for item in result if item.generation_route == "high_temp"])
            else "high_temp"
        )
        other_route = "high_temp" if next_route == "forced_wrong" else "forced_wrong"
        route = next_route if by_route[next_route] else other_route
        result.append(by_route[route].pop(0))
    return result


def _stable_order(candidates: list[DPOCandidate], *, seed: int) -> list[DPOCandidate]:
    return sorted(candidates, key=lambda item: _stable_key(item, seed))


def _stable_key(candidate: DPOCandidate, seed: int) -> str:
    payload = f"{seed}|{candidate.source_id}|{candidate.candidate_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _pair_from_candidate(row: dict[str, Any], candidate: DPOCandidate) -> dict[str, Any]:
    return {
        "prompt": row["prompt"],
        "chosen": row["response"],
        "rejected": candidate.rejected,
        "rejected_category": candidate.rejected_category,
        "generation_route": candidate.generation_route,
        "provenance": {
            "source_id": candidate.source_id,
            "candidate_id": candidate.candidate_id,
            "rollout_index": candidate.rollout_index,
            "validation": candidate.validation,
        },
    }


def _counts(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def _candidate_to_dict(candidate: DPOCandidate) -> dict[str, Any]:
    return {
        "source_id": candidate.source_id,
        "candidate_id": candidate.candidate_id,
        "generation_route": candidate.generation_route,
        "rejected": candidate.rejected,
        "rejected_category": candidate.rejected_category,
        "validation": candidate.validation,
        "rollout_index": candidate.rollout_index,
    }


def _candidate_from_dict(row: dict[str, Any]) -> DPOCandidate:
    return DPOCandidate(
        source_id=row["source_id"],
        candidate_id=row["candidate_id"],
        generation_route=row["generation_route"],
        rejected=row["rejected"],
        rejected_category=row["rejected_category"],
        validation=row["validation"],
        rollout_index=row["rollout_index"],
    )


def _artifact_file(
    output_dir: Path,
    filename: str,
    row_count: int,
    schema: dict[str, str],
) -> ArtifactFile:
    path = output_dir / filename
    return ArtifactFile(
        relative_path=filename,
        sha256=sha256_file(path),
        byte_size=path.stat().st_size,
        row_count=row_count,
        field_schema=schema,
    )
