"""Thin verl adapter for the Countdown rule reward."""

from __future__ import annotations

from typing import Any

from post_train_v2.src.rewards.countdown import score_response


def compute_score(
    *,
    data_source: str,
    solution_str: str,
    ground_truth: dict[str, Any],
    extra_info: dict[str, Any] | None = None,
    **_kwargs,
) -> dict[str, Any]:
    del extra_info
    if data_source != "countdown":
        return {
            "score": 0.0,
            "format_ok": False,
            "answer_correct": False,
            "error": "unsupported_data_source",
            "expression": None,
            "value": None,
        }
    result = score_response(
        solution_str,
        list(ground_truth["numbers"]),
        int(ground_truth["target"]),
    )
    return {
        "score": float(result.score),
        "format_ok": bool(result.format_ok),
        "answer_correct": bool(result.answer_correct),
        "error": result.error,
        "expression": result.expression,
        "value": result.value,
    }
