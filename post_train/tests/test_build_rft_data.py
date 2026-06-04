import pytest

from post_train.scripts.sft.build_rft_data import (
    build_rollouts,
    classify_rft_responses,
    validate_source_rows,
)


def test_build_rollouts_duplicates_prompts_and_tracks_source_sample_index():
    rows = [
        {"id": "sft-000001", "prompt": "prompt 1"},
        {"id": "sft-000002", "prompt": "prompt 2"},
    ]

    rollouts = build_rollouts(rows, num_samples_per_prompt=2)

    assert [item.prompt for item in rollouts] == [
        "prompt 1",
        "prompt 1",
        "prompt 2",
        "prompt 2",
    ]
    assert [(item.source_index, item.sample_index) for item in rollouts] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]


def test_validate_source_rows_rejects_duplicate_source_ids_before_rollouts():
    rows = [
        {"id": "sft-000001", "prompt": "prompt 1"},
        {"id": "sft-000001", "prompt": "prompt 2"},
    ]

    with pytest.raises(ValueError, match="duplicate id.*sft-000001"):
        validate_source_rows(rows)


def test_validate_source_rows_rejects_empty_source_ids_before_rollouts():
    rows = [
        {"id": "sft-000001", "prompt": "prompt 1"},
        {"id": "", "prompt": "prompt 2"},
    ]

    with pytest.raises(ValueError, match="non-empty id"):
        validate_source_rows(rows)


def test_classify_rft_responses_writes_accepted_rows_and_rejected_summary():
    rows = [
        {
            "id": "sft-000001",
            "prompt": "solve 24",
            "numbers": [7, 3, 8, 2],
            "target": 24,
        },
        {
            "id": "sft-000002",
            "prompt": "solve 11",
            "numbers": [1, 2, 3, 4],
            "target": 11,
        },
    ]
    rollouts = build_rollouts(rows, num_samples_per_prompt=1)
    responses = [
        "reasoning\n<answer> (7-3)*(8-2) </answer>\n",
        "<answer> 1+2+3+4 </answer>",
    ]

    accepted, rejected = classify_rft_responses(rows, rollouts, responses)

    assert accepted == [
        {
            "id": "sft-000001-rft-0",
            "prompt": "solve 24",
            "response": "reasoning\n<answer> (7-3)*(8-2) </answer>",
            "numbers": [7, 3, 8, 2],
            "target": 24,
            "source": "rft",
            "source_id": "sft-000001",
            "teacher_expr": "(7-3)*(8-2)",
            "validation": {
                "ok": True,
                "error": None,
                "value": 24,
            },
        }
    ]
    assert rejected == [
        {
            "id": "sft-000002-rft-0",
            "prompt": "solve 11",
            "response": "<answer> 1+2+3+4 </answer>",
            "numbers": [1, 2, 3, 4],
            "target": 11,
            "source_index": 1,
            "source_id": "sft-000002",
            "sample_index": 0,
            "teacher_expr": "1+2+3+4",
            "validation": {
                "ok": False,
                "error": "wrong_value",
                "value": 10,
            },
        }
    ]
