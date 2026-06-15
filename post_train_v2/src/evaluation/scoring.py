"""Framework-neutral Countdown evaluation scoring."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from post_train_v2.src.countdown.validation import (
    extract_answer_text,
    serialize_fraction,
    validate_countdown_response,
)


def score_response(
    row: Mapping[str, Any],
    raw_generation: str,
    *,
    generated_tokens: int,
    truncated: bool,
) -> dict[str, Any]:
    if not isinstance(raw_generation, str):
        raise ValueError("raw_generation must be a string")
    if type(generated_tokens) is not int or generated_tokens < 0:
        raise ValueError("generated_tokens must be a nonnegative exact integer")
    if type(truncated) is not bool:
        raise ValueError("truncated must be a boolean")

    result = validate_countdown_response(
        raw_generation,
        list(row["numbers"]),
        row["target"],
    )
    expression = extract_answer_text(raw_generation)
    valid_expression = result.error not in {
        "missing_answer_tag",
        "invalid_expression",
    }
    return {
        "id": row["id"],
        "prompt": row["prompt"],
        "raw_generation": raw_generation,
        "extracted_expr": expression,
        "format_ok": expression is not None,
        "valid_expression": valid_expression,
        "correct": result.ok,
        "error": result.error,
        "value": serialize_fraction(result.value),
        "generated_tokens": generated_tokens,
        "truncated": truncated,
    }


def aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    count = len(rows)
    if count == 0:
        return {
            "count": 0,
            "accuracy": 0.0,
            "format_rate": 0.0,
            "valid_expression_rate": 0.0,
            "avg_generated_tokens": 0.0,
            "max_generated_tokens": 0,
            "truncated_count": 0,
            "truncated_rate": 0.0,
        }

    token_counts: list[int] = []
    for index, row in enumerate(rows):
        tokens = row.get("generated_tokens")
        if type(tokens) is not int or tokens < 0:
            raise ValueError(
                f"rows[{index}].generated_tokens must be a nonnegative exact integer"
            )
        token_counts.append(tokens)

    truncated_count = sum(row.get("truncated") is True for row in rows)
    return {
        "count": count,
        "accuracy": sum(row.get("correct") is True for row in rows) / count,
        "format_rate": sum(row.get("format_ok") is True for row in rows) / count,
        "valid_expression_rate": (
            sum(row.get("valid_expression") is True for row in rows) / count
        ),
        "avg_generated_tokens": sum(token_counts) / count,
        "max_generated_tokens": max(token_counts),
        "truncated_count": truncated_count,
        "truncated_rate": truncated_count / count,
    }

