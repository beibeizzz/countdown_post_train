"""DPO rejected candidate classification and pair selection."""

from __future__ import annotations

from dataclasses import dataclass
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
