from __future__ import annotations

from post_train_v2.src.countdown.bucketing import assign_bucket
from post_train_v2.src.countdown.prompts import build_solution_prompt
from post_train_v2.src.generation.rft import (
    build_rollout_requests,
    normalize_rollout_sources,
    select_rft_rows,
)


def source_row():
    numbers = [1, 1, 1, 1]
    target = 4
    gold_expr = "1+1+1+1"
    return {
        "id": "source-1",
        "source_index": 7,
        "numbers": numbers,
        "target": target,
        "gold_expr": gold_expr,
        "prompt": build_solution_prompt(numbers, target),
        "bucket": assign_bucket(numbers, gold_expr),
    }


def test_build_rollout_requests_expands_four_per_source_with_stable_metadata():
    requests = build_rollout_requests([source_row()], rollouts_per_prompt=4, seed=123)

    assert [request.position for request in requests] == [0, 1, 2, 3]
    assert [request.prompt for request in requests] == [source_row()["prompt"]] * 4
    assert [request.metadata["source_index"] for request in requests] == [7] * 4
    assert [request.metadata["rollout_index"] for request in requests] == [0, 1, 2, 3]
    assert len({request.seed for request in requests}) == 4


def test_select_rft_rows_deduplicates_exact_text_and_keeps_earliest_two_correct():
    source = source_row()
    responses = [
        (0, "  <answer>1+1+1+1</answer>\r\n"),
        (1, "<answer>(1+1)+(1+1)</answer>"),
        (2, "<answer>1+1+1+1</answer>"),
        (3, "<answer>1+1+1-1</answer>"),
    ]

    accepted, rejected = select_rft_rows([source], responses, rollouts_per_prompt=4)

    assert [row["provenance"]["rollout_index"] for row in accepted] == [0, 1]
    assert [row["response"] for row in accepted] == [
        "<answer>1+1+1+1</answer>",
        "<answer>(1+1)+(1+1)</answer>",
    ]
    assert [row["provenance"]["rollout_index"] for row in rejected] == [3]


def test_select_rft_rows_keeps_expression_equivalent_distinct_responses():
    source = source_row()
    responses = [
        (0, "<answer>1+1+1+1</answer>"),
        (1, "We can add all numbers.\n<answer>1+1+1+1</answer>"),
    ]

    accepted, rejected = select_rft_rows([source], responses, rollouts_per_prompt=2)

    assert len(accepted) == 2
    assert rejected == []


def test_normalize_rollout_sources_accepts_sft_records():
    row = {
        **source_row(),
        "response": "<answer>1+1+1+1</answer>",
        "validation": {
            "ok": True,
            "value": "4/1",
            "used_numbers": [1, 1, 1, 1],
            "expression": "1+1+1+1",
            "error": None,
        },
        "provenance": {"stage": "teacher"},
    }

    assert normalize_rollout_sources([row]) == [source_row()]
