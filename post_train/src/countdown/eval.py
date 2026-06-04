from __future__ import annotations

from typing import Any

from post_train.src.countdown.validation import extract_answer_text, validate_countdown_response


def score_generation(
    row: dict[str, Any],
    raw_generation: str,
    generated_tokens: int,
    truncated: bool,
) -> dict[str, Any]:
    extracted_expr = extract_answer_text(raw_generation)
    result = validate_countdown_response(raw_generation, row["numbers"], int(row["target"]))

    return {
        "id": row.get("id"),
        "prompt": row.get("prompt"),
        "raw_generation": raw_generation,
        "extracted_expr": extracted_expr,
        "format_ok": extracted_expr is not None,
        "valid": result.ok,
        "correct": result.ok,
        "error": result.error,
        "generated_tokens": generated_tokens,
        "truncated": truncated,
    }


def aggregate_eval_rows(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    if not rows:
        return {
            "accuracy": 0,
            "format_rate": 0,
            "valid_expression_rate": 0,
            "avg_generated_tokens": 0,
            "max_generated_tokens": 0,
            "truncated_count": 0,
        }

    row_count = len(rows)
    generated_tokens = [int(row.get("generated_tokens", 0)) for row in rows]

    return {
        "accuracy": sum(1 for row in rows if row.get("correct")) / row_count,
        "format_rate": sum(1 for row in rows if row.get("format_ok")) / row_count,
        "valid_expression_rate": sum(1 for row in rows if row.get("valid")) / row_count,
        "avg_generated_tokens": sum(generated_tokens) / row_count,
        "max_generated_tokens": max(generated_tokens),
        "truncated_count": sum(1 for row in rows if row.get("truncated")),
    }
