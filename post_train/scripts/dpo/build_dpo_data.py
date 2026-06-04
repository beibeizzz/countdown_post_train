from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from post_train.src.countdown.config import load_yaml_config, resolve_path
from post_train.src.countdown.generation import GenerationConfig, VLLMGenerator
from post_train.src.countdown.io import read_jsonl, write_jsonl, write_manifest
from post_train.src.countdown.prompts import build_dpo_forced_wrong_prompt
from post_train.src.countdown.validation import extract_answer_text, validate_countdown_expression


DEFAULT_CONFIG = "post_train/configs/dpo_data.yaml"
PAIR_FILENAME = "dpo_train.jsonl"
CANDIDATE_FILENAME = "dpo_rejected_candidates.jsonl"
MANIFEST_FILENAME = "manifest.json"

MALFORMED_CATEGORIES = {"invalid_expression", "missing_answer_tag", "truncated"}
CATEGORY_PRIORITY = {
    "wrong_value": 0,
    "number_mismatch": 1,
    "invalid_expression": 2,
    "missing_answer_tag": 3,
    "truncated": 4,
}


@dataclass(frozen=True)
class RouteRequest:
    source_index: int
    route: str
    prompt: str


@dataclass(frozen=True)
class Candidate:
    source_index: int
    source_id: str
    route: str
    text: str
    category: str
    prompt: str = ""
    chosen: str = ""
    numbers: list[int] | None = None
    target: int | None = None
    validation: dict[str, Any] | None = None
    truncated: bool = False
    truncation_source: str = "unavailable"
    truncation_evidence: dict[str, Any] | None = None


@dataclass(frozen=True)
class GenerationRecord:
    text: str
    finish_reason: str | None = None
    token_count: int | None = None
    stop_reason: str | None = None
    truncation_source: str = "unavailable"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DPO pairs from chosen SFT data and rejected rollouts.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--limit", type=int, default=None, help="Limit chosen rows for smoke runs.")
    return parser.parse_args()


def classify_rejected(text: str, numbers: list[int], target: int, truncated: bool) -> str:
    if truncated:
        return "truncated"
    expr = extract_answer_text(text)
    if expr is None:
        return "missing_answer_tag"
    result = validate_countdown_expression(expr, numbers, target)
    if result.ok:
        return "unexpected_correct"
    if result.error == "number_mismatch":
        return "number_mismatch"
    if result.error == "wrong_value":
        return "wrong_value"
    return "invalid_expression"


def validate_unique_chosen_ids(rows: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    for index, row in enumerate(rows):
        row_id = str(row.get("id", "")).strip()
        if not row_id:
            raise ValueError(f"chosen row {index} must contain a non-empty id")
        if row_id in seen:
            raise ValueError(f"chosen rows contain duplicate id: {row_id}")
        seen.add(row_id)


def validate_chosen_schema(rows: list[dict[str, Any]]) -> None:
    required = {"id", "prompt", "response", "numbers", "target"}
    for index, row in enumerate(rows):
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"chosen row {index} is missing required keys: {missing}")
        if not isinstance(row["numbers"], list):
            raise ValueError(f"chosen row {index} numbers must be a list")


def validate_config(cfg: dict[str, Any]) -> None:
    required = [
        "target_pairs",
        "forced_wrong_fraction",
        "high_temp_fraction",
        "forced_wrong_temperature",
        "high_temp_temperature",
        "top_p",
        "max_new_tokens",
        "batch_size",
        "malformed_cap_fraction",
        "preferred_wrong_value_min_fraction",
    ]
    for key in required:
        if key not in cfg:
            raise ValueError(f"config missing required key: {key}")

    int_settings = ["target_pairs", "max_new_tokens", "batch_size"]
    for key in int_settings:
        value = int(cfg[key])
        if key == "target_pairs":
            if value < 0:
                raise ValueError(f"{key} must be non-negative")
        elif value < 1:
            raise ValueError(f"{key} must be at least 1")

    fraction_settings = [
        "forced_wrong_fraction",
        "high_temp_fraction",
        "malformed_cap_fraction",
        "preferred_wrong_value_min_fraction",
    ]
    for key in fraction_settings:
        value = float(cfg[key])
        if not 0 <= value <= 1:
            raise ValueError(f"{key} must be between 0 and 1")

    if float(cfg["forced_wrong_fraction"]) + float(cfg["high_temp_fraction"]) <= 0:
        raise ValueError("at least one rejected generation route fraction must be positive")

    for key in ["forced_wrong_temperature", "high_temp_temperature"]:
        if float(cfg[key]) < 0:
            raise ValueError(f"{key} must be non-negative")

    top_p = float(cfg["top_p"])
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be greater than 0 and at most 1")

    oversample_factor = float(cfg.get("candidate_oversample_factor", 2.0))
    if oversample_factor <= 0:
        raise ValueError("candidate_oversample_factor must be positive")


def build_generation_config(cfg: dict[str, Any], temperature: float) -> GenerationConfig:
    return GenerationConfig(
        max_new_tokens=int(cfg["max_new_tokens"]),
        temperature=temperature,
        top_p=float(cfg["top_p"]),
        enable_thinking=bool(cfg.get("enable_thinking", False)),
    )


def batched(items: list[RouteRequest], batch_size: int) -> list[list[RouteRequest]]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def build_route_requests(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> list[RouteRequest]:
    target_pairs = int(cfg["target_pairs"])
    if target_pairs < 0:
        raise ValueError("target_pairs must be non-negative")
    if not rows or target_pairs == 0:
        return []

    route_specs = [
        ("forced_wrong", float(cfg.get("forced_wrong_fraction", 0.5))),
        ("high_temp", float(cfg.get("high_temp_fraction", 0.5))),
    ]
    enabled_routes = [route for route, route_fraction in route_specs if route_fraction > 0]
    if not enabled_routes:
        return []
    candidate_budget = min(
        len(rows) * len(enabled_routes),
        math.ceil(target_pairs * float(cfg.get("candidate_oversample_factor", 2.0))),
    )
    route_budgets = allocate_route_budgets(route_specs, candidate_budget, len(rows))

    requests: list[RouteRequest] = []
    route_counts = {route: 0 for route in enabled_routes}
    for source_index, row in enumerate(rows):
        for route in enabled_routes:
            if len(requests) >= candidate_budget:
                return requests
            if route_counts[route] >= route_budgets[route]:
                continue
            if route == "forced_wrong":
                prompt = build_dpo_forced_wrong_prompt(
                    list(row["numbers"]),
                    int(row["target"]),
                    str(row["response"]),
                )
            else:
                prompt = str(row["prompt"])
            requests.append(
                RouteRequest(
                    source_index=source_index,
                    route=route,
                    prompt=prompt,
                )
            )
            route_counts[route] += 1
    return requests


def allocate_route_budgets(
    route_specs: list[tuple[str, float]],
    candidate_budget: int,
    max_per_route: int,
) -> dict[str, int]:
    enabled = [(route, route_fraction) for route, route_fraction in route_specs if route_fraction > 0]
    if not enabled or candidate_budget <= 0:
        return {}
    total_fraction = sum(route_fraction for _, route_fraction in enabled)
    budgets: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    remaining = candidate_budget
    for route, route_fraction in enabled:
        raw_budget = candidate_budget * route_fraction / total_fraction
        budget = min(max_per_route, math.floor(raw_budget))
        budgets[route] = budget
        remaining -= budget
        remainders.append((raw_budget - math.floor(raw_budget), route))

    for _, route in sorted(remainders, reverse=True):
        if remaining <= 0:
            break
        if budgets[route] >= max_per_route:
            continue
        budgets[route] += 1
        remaining -= 1

    while remaining > 0:
        progressed = False
        for route, _ in enabled:
            if remaining <= 0:
                break
            if budgets[route] >= max_per_route:
                continue
            budgets[route] += 1
            remaining -= 1
            progressed = True
        if not progressed:
            break
    return budgets


def generation_record_from_metadata(metadata: dict[str, Any] | str) -> GenerationRecord:
    if isinstance(metadata, str):
        return GenerationRecord(text=metadata)
    return GenerationRecord(
        text=str(metadata.get("text", "")),
        finish_reason=metadata.get("finish_reason"),
        token_count=metadata.get("token_count"),
        stop_reason=metadata.get("stop_reason"),
    )


def truncation_status(record: GenerationRecord, max_new_tokens: int) -> tuple[bool, str, dict[str, Any]]:
    evidence = {
        "finish_reason": record.finish_reason,
        "token_count": record.token_count,
        "stop_reason": record.stop_reason,
        "max_new_tokens": max_new_tokens,
    }
    if record.finish_reason is None and record.token_count is None and record.stop_reason is None:
        return False, "unavailable", evidence
    if str(record.finish_reason).lower() in {"length", "max_tokens"}:
        return True, "finish_reason", evidence
    if record.token_count is not None and record.token_count >= max_new_tokens:
        if record.finish_reason is None and record.stop_reason is None:
            return True, "token_count", evidence
    return False, "metadata", evidence


def validation_detail(text: str, numbers: list[int], target: int) -> dict[str, Any]:
    expr = extract_answer_text(text)
    if expr is None:
        return {
            "ok": False,
            "error": "missing_answer_tag",
            "value": None,
            "extracted_expr": None,
        }
    result = validate_countdown_expression(expr, numbers, target)
    return {
        "ok": result.ok,
        "error": result.error,
        "value": result.value,
        "extracted_expr": expr,
    }


def build_candidates(
    rows: list[dict[str, Any]],
    requests: list[RouteRequest],
    responses: list[GenerationRecord | dict[str, Any] | str],
    max_new_tokens: int,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    for request, response in zip(requests, responses, strict=True):
        row = rows[request.source_index]
        record = response if isinstance(response, GenerationRecord) else generation_record_from_metadata(response)
        text = record.text.strip()
        truncated, truncation_source, truncation_evidence = truncation_status(record, max_new_tokens)
        validation = validation_detail(text, list(row["numbers"]), int(row["target"]))
        category = classify_rejected(
            text,
            list(row["numbers"]),
            int(row["target"]),
            truncated=truncated,
        )
        candidates.append(
            Candidate(
                source_index=request.source_index,
                source_id=str(row["id"]),
                route=request.route,
                text=text,
                category=category,
                prompt=str(row["prompt"]),
                chosen=str(row["response"]),
                numbers=list(row["numbers"]),
                target=int(row["target"]),
                validation=validation,
                truncated=truncated,
                truncation_source=truncation_source,
                truncation_evidence=truncation_evidence,
            )
        )
    return candidates


def select_dpo_pairs(
    rows: list[dict[str, Any]],
    candidates: list[Candidate],
    target_pairs: int,
    malformed_cap_fraction: float,
    preferred_wrong_value_min_fraction: float,
) -> list[dict[str, Any]]:
    if target_pairs < 0:
        raise ValueError("target_pairs must be non-negative")
    if not 0 <= malformed_cap_fraction <= 1:
        raise ValueError("malformed_cap_fraction must be between 0 and 1")
    if not 0 <= preferred_wrong_value_min_fraction <= 1:
        raise ValueError("preferred_wrong_value_min_fraction must be between 0 and 1")

    eligible = [candidate for candidate in candidates if candidate.category != "unexpected_correct"]
    ordered = sorted(
        enumerate(eligible),
        key=lambda item: (
            CATEGORY_PRIORITY.get(item[1].category, 99),
            item[1].source_index,
            item[1].route,
            item[0],
        ),
    )
    malformed_cap = math.floor(target_pairs * malformed_cap_fraction)
    malformed_count = 0
    pairs: list[dict[str, Any]] = []

    for _, candidate in ordered:
        if len(pairs) >= target_pairs:
            break
        if candidate.category in MALFORMED_CATEGORIES:
            if malformed_count >= malformed_cap:
                continue
            malformed_count += 1

        row = rows[candidate.source_index]
        pairs.append(
            {
                "id": f"{candidate.source_id}-dpo-{len(pairs)}",
                "prompt": row["prompt"],
                "chosen": row["response"],
                "rejected": candidate.text.strip(),
                "numbers": row["numbers"],
                "target": row["target"],
                "source_id": candidate.source_id,
                "rejected_route": candidate.route,
                "rejected_category": candidate.category,
            }
        )

    required_wrong_value = math.ceil(len(pairs) * preferred_wrong_value_min_fraction)
    selected_wrong_value = sum(1 for pair in pairs if pair["rejected_category"] == "wrong_value")
    available_wrong_value = sum(1 for candidate in eligible if candidate.category == "wrong_value")
    if available_wrong_value >= required_wrong_value and selected_wrong_value < required_wrong_value:
        raise RuntimeError(
            "wrong_value selection invariant failed: enough wrong_value candidates were available "
            f"({available_wrong_value}) but only {selected_wrong_value}/{required_wrong_value} were selected"
        )

    return pairs


def candidate_to_row(rows: list[dict[str, Any]], candidate: Candidate) -> dict[str, Any]:
    row = rows[candidate.source_index]
    return {
        "source_id": candidate.source_id,
        "source_index": candidate.source_index,
        "prompt": candidate.prompt or row["prompt"],
        "chosen": candidate.chosen or row["response"],
        "numbers": candidate.numbers if candidate.numbers is not None else row["numbers"],
        "target": candidate.target if candidate.target is not None else row["target"],
        "rejected_route": candidate.route,
        "rejected": candidate.text,
        "rejected_category": candidate.category,
        "validation": candidate.validation or validation_detail(
            candidate.text,
            list(row["numbers"]),
            int(row["target"]),
        ),
        "truncated": candidate.truncated,
        "truncation_source": candidate.truncation_source,
        "truncation_evidence": candidate.truncation_evidence or {},
    }


def fraction(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return count / total


def build_manifest(
    cfg: dict[str, Any],
    model_path: Path | str,
    rows: list[dict[str, Any]],
    candidates: list[Candidate],
    pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    category_counts = Counter(candidate.category for candidate in candidates)
    route_counts = Counter(candidate.route for candidate in candidates)
    eligible_candidates = [candidate for candidate in candidates if candidate.category != "unexpected_correct"]
    eligible_category_counts = Counter(candidate.category for candidate in eligible_candidates)
    pair_category_counts = Counter(pair["rejected_category"] for pair in pairs)
    malformed_pairs = sum(pair_category_counts[category] for category in MALFORMED_CATEGORIES)
    wrong_value_pairs = pair_category_counts["wrong_value"]
    required_wrong_value = math.ceil(len(pairs) * float(cfg["preferred_wrong_value_min_fraction"]))
    wrong_value_shortfall = max(0, required_wrong_value - wrong_value_pairs)

    return {
        "name": "dpo_data",
        "category_counts": dict(sorted(category_counts.items())),
        "route_counts": dict(sorted(route_counts.items())),
        "eligible_category_counts": dict(sorted(eligible_category_counts.items())),
        "pair_category_counts": dict(sorted(pair_category_counts.items())),
        "num_chosen_input": len(rows),
        "num_candidates": len(candidates),
        "num_eligible_candidates": len(eligible_candidates),
        "num_pairs": len(pairs),
        "target_pairs": int(cfg["target_pairs"]),
        "pair_shortfall": max(0, int(cfg["target_pairs"]) - len(pairs)),
        "malformed_fraction": fraction(malformed_pairs, len(pairs)),
        "wrong_value_fraction": fraction(wrong_value_pairs, len(pairs)),
        "wrong_value_min_satisfied": wrong_value_shortfall == 0,
        "wrong_value_shortfall": wrong_value_shortfall,
        "preferred_wrong_value_min_fraction": float(cfg["preferred_wrong_value_min_fraction"]),
        "malformed_cap_fraction": float(cfg["malformed_cap_fraction"]),
        "candidate_oversample_factor": float(cfg.get("candidate_oversample_factor", 2.0)),
        "model": str(model_path),
        "max_new_tokens": int(cfg["max_new_tokens"]),
        "enable_thinking": bool(cfg.get("enable_thinking", False)),
    }


def generate_route(
    generator: VLLMGenerator,
    requests: list[RouteRequest],
    cfg: dict[str, Any],
    temperature: float,
) -> list[GenerationRecord]:
    responses: list[GenerationRecord] = []
    generation_config = build_generation_config(cfg, temperature)
    for batch in batched(requests, int(cfg["batch_size"])):
        metadata_records = generator.generate_with_metadata(
            [request.prompt for request in batch],
            generation_config,
        )
        responses.extend(generation_record_from_metadata(record) for record in metadata_records)
    return responses


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be non-negative")

    cfg_path = resolve_path(args.config, REPO_ROOT)
    cfg = load_yaml_config(cfg_path)
    validate_config(cfg)

    chosen_path = resolve_path(cfg["chosen_data"], REPO_ROOT)
    output_dir = resolve_path(cfg["output_dir"], REPO_ROOT)
    model_path = resolve_path(cfg["model_path"], REPO_ROOT)
    pair_path = output_dir / PAIR_FILENAME
    candidate_path = output_dir / CANDIDATE_FILENAME

    rows = read_jsonl(chosen_path)
    if args.limit is not None:
        rows = rows[: args.limit]
    validate_unique_chosen_ids(rows)
    validate_chosen_schema(rows)

    requests = build_route_requests(rows, cfg)

    response_by_request_index: dict[int, GenerationRecord] = {}
    ordered_requests: list[RouteRequest] = []
    if requests:
        generator = VLLMGenerator(str(model_path))
        request_items = list(enumerate(requests))
        forced_items = [(index, request) for index, request in request_items if request.route == "forced_wrong"]
        high_temp_items = [(index, request) for index, request in request_items if request.route == "high_temp"]
        forced_responses = generate_route(
            generator,
            [request for _, request in forced_items],
            cfg,
            float(cfg["forced_wrong_temperature"]),
        )
        high_temp_responses = generate_route(
            generator,
            [request for _, request in high_temp_items],
            cfg,
            float(cfg["high_temp_temperature"]),
        )
        for (index, _), response in zip(forced_items, forced_responses, strict=True):
            response_by_request_index[index] = response
        for (index, _), response in zip(high_temp_items, high_temp_responses, strict=True):
            response_by_request_index[index] = response
        ordered_requests = requests
    responses = [response_by_request_index[index] for index in range(len(ordered_requests))]

    candidates = build_candidates(
        rows,
        ordered_requests,
        responses,
        int(cfg["max_new_tokens"]),
    )
    pairs = select_dpo_pairs(
        rows,
        candidates,
        int(cfg["target_pairs"]),
        float(cfg["malformed_cap_fraction"]),
        float(cfg["preferred_wrong_value_min_fraction"]),
    )

    write_jsonl(pair_path, pairs)
    write_jsonl(candidate_path, [candidate_to_row(rows, candidate) for candidate in candidates])
    write_manifest(output_dir / MANIFEST_FILENAME, build_manifest(cfg, model_path, rows, candidates, pairs))


if __name__ == "__main__":
    main()
