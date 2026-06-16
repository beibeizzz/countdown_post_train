"""Framework-neutral Countdown rule reward."""

from __future__ import annotations

from dataclasses import dataclass

from post_train_v2.src.countdown.validation import (
    extract_answer_text,
    serialize_fraction,
    validate_countdown_response,
)


@dataclass(frozen=True)
class RewardResult:
    score: float
    format_ok: bool
    answer_correct: bool
    error: str | None
    expression: str | None
    value: str | None


def score_response(text: str, numbers: list[int], target: int) -> RewardResult:
    expression = extract_answer_text(text)
    format_ok = expression is not None
    validation = validate_countdown_response(text, numbers, target)
    answer_correct = validation.ok
    score = (0.2 if format_ok else 0.0) + (1.0 if answer_correct else 0.0)
    return RewardResult(
        score=score,
        format_ok=format_ok,
        answer_correct=answer_correct,
        error=validation.error,
        expression=expression,
        value=serialize_fraction(validation.value),
    )
