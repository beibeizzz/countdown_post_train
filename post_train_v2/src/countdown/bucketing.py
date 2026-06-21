"""Countdown difficulty bucketing based on input count and expression shape."""

from __future__ import annotations

from post_train_v2.src.countdown.solver import expression_metadata


def assign_bucket(numbers: list[int], expr: str) -> dict[str, int | bool | str]:
    metadata = expression_metadata(expr, num_count=len(numbers))
    score = 0

    if metadata["num_count"] >= 4:
        score += 1
    if metadata["num_count"] >= 5:
        score += 2
    if metadata["has_subtraction"]:
        score += 1
    if metadata["has_division"]:
        score += 2
    if metadata["expr_depth"] >= 4:
        score += 1
    if metadata["expr_len"] >= 18:
        score += 1

    if score <= 1:
        complexity = "easy"
    elif score <= 3:
        complexity = "medium"
    else:
        complexity = "hard"

    return {
        **metadata,
        "score": score,
        "complexity": complexity,
        "bucket_key": f"{metadata['num_count']}_{complexity}",
    }
