"""DPO rejected candidate classification and pair selection."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Literal

from post_train_v2.src.countdown.validation import (
    serialize_fraction,
    validate_countdown_response,
)
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
