from __future__ import annotations

from post_train.src.countdown.solver import expression_metadata


def assign_bucket(numbers: list[int], expr: str) -> dict:
    meta = expression_metadata(expr, num_count=len(numbers))
    score = 0

    if meta["num_count"] >= 4:
        score += 1
    if meta["num_count"] >= 5:
        score += 2
    if meta["has_subtraction"]:
        score += 1
    if meta["has_division"]:
        score += 2
    if meta["expr_depth"] >= 4:
        score += 1
    if meta["expr_len"] >= 18:
        score += 1

    if score <= 1:
        complexity = "easy"
    elif score <= 3:
        complexity = "medium"
    else:
        complexity = "hard"

    return {
        **meta,
        "score": score,
        "complexity": complexity,
        "bucket_key": f"{meta['num_count']}_{complexity}",
    }
