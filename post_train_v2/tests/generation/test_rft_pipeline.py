from __future__ import annotations

from post_train_v2.src.generation.rft import (
    build_rollout_requests,
    select_rft_rows,
)


def source_row():
    return {
        "id": "source-1",
        "source_index": 7,
        "numbers": [1, 1, 1, 1],
        "target": 4,
        "gold_expr": "1+1+1+1",
        "prompt": "Using the numbers [1, 1, 1, 1], create an equation that equals 4.",
        "bucket": {
            "num_count": 4,
            "expr_depth": 1,
            "expr_len": 7,
            "has_division": False,
            "has_subtraction": False,
            "score": 1,
            "complexity": "easy",
            "bucket_key": "easy:4:1",
        },
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
