from __future__ import annotations

import pytest

from post_train_v2.src.countdown.prompts import build_solution_prompt
from post_train_v2.src.generation.dpo import (
    DPOCandidate,
    classify_dpo_candidate,
)
from post_train_v2.src.generation.metadata import GenerationRecord


def source_row() -> dict:
    numbers = [1, 1, 1, 1]
    target = 4
    return {
        "id": "row-1",
        "source_index": 1,
        "numbers": numbers,
        "target": target,
        "gold_expr": "1+1+1+1",
        "prompt": build_solution_prompt(numbers, target),
        "bucket": {},
    }


@pytest.mark.parametrize(
    ("text", "category"),
    (
        ("<answer>1+1+1-1</answer>", "wrong_value"),
        ("<answer>1+1+1+1+1</answer>", "number_mismatch"),
        ("<answer>1+/1</answer>", "invalid_expression"),
        ("1+1+1+1", "missing_answer_tag"),
    ),
)
def test_classify_dpo_candidate_validation_categories(text: str, category: str):
    candidate = classify_dpo_candidate(
        source=source_row(),
        record=GenerationRecord(
            text=text,
            finish_reason="stop",
            token_count=4,
            stop_reason=None,
        ),
        generation_route="forced_wrong",
        rollout_index=0,
    )

    assert isinstance(candidate, DPOCandidate)
    assert candidate.rejected_category == category
    assert candidate.generation_route == "forced_wrong"
    assert candidate.source_id == "row-1"
    assert candidate.candidate_id == "row-1:forced_wrong:0"
    assert candidate.validation["error"] == category


def test_truncated_takes_precedence_over_validation_result():
    candidate = classify_dpo_candidate(
        source=source_row(),
        record=GenerationRecord(
            text="<answer>1+1+1+1</answer>",
            finish_reason="length",
            token_count=256,
            stop_reason=None,
            truncated=True,
        ),
        generation_route="high_temp",
        rollout_index=3,
    )

    assert candidate.rejected_category == "truncated"
    assert candidate.validation["error"] == "truncated"


def test_unexpected_correct_is_not_eligible():
    candidate = classify_dpo_candidate(
        source=source_row(),
        record=GenerationRecord(
            text="<answer>1+1+1+1</answer>",
            finish_reason="stop",
            token_count=5,
            stop_reason=None,
        ),
        generation_route="high_temp",
        rollout_index=1,
    )

    assert candidate.rejected_category == "unexpected_correct"
    assert candidate.eligible is False
