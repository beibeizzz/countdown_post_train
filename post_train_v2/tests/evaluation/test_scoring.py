from __future__ import annotations

from post_train_v2.src.countdown.bucketing import assign_bucket
from post_train_v2.src.countdown.prompts import build_solution_prompt
from post_train_v2.src.evaluation.scoring import aggregate_rows, score_response


def source_row() -> dict:
    numbers = [1, 2, 3]
    target = 6
    expression = "1+2+3"
    return {
        "id": "train-000001",
        "source_index": 1,
        "numbers": numbers,
        "target": target,
        "gold_expr": expression,
        "prompt": build_solution_prompt(numbers, target),
        "bucket": assign_bucket(numbers, expression),
    }


def test_score_response_keeps_full_generation_and_exact_diagnostics():
    scored = score_response(
        source_row(),
        "Reasoning\n<answer>1+2+3</answer>",
        generated_tokens=12,
        truncated=False,
    )

    assert scored["correct"] is True
    assert scored["format_ok"] is True
    assert scored["valid_expression"] is True
    assert scored["value"] == "6/1"
    assert scored["raw_generation"].startswith("Reasoning")
    assert scored["generated_tokens"] == 12


def test_aggregate_rows_includes_truncation_rate_and_empty_contract():
    rows = [
        {
            "correct": True,
            "format_ok": True,
            "valid_expression": True,
            "generated_tokens": 10,
            "truncated": False,
        },
        {
            "correct": False,
            "format_ok": False,
            "valid_expression": False,
            "generated_tokens": 256,
            "truncated": True,
        },
    ]

    metrics = aggregate_rows(rows)

    assert metrics["count"] == 2
    assert metrics["accuracy"] == 0.5
    assert metrics["format_rate"] == 0.5
    assert metrics["valid_expression_rate"] == 0.5
    assert metrics["truncated_count"] == 1
    assert metrics["truncated_rate"] == 0.5
    assert metrics["avg_generated_tokens"] == 133.0
    assert aggregate_rows([]) == {
        "count": 0,
        "accuracy": 0.0,
        "format_rate": 0.0,
        "valid_expression_rate": 0.0,
        "avg_generated_tokens": 0.0,
        "max_generated_tokens": 0,
        "truncated_count": 0,
        "truncated_rate": 0.0,
    }

