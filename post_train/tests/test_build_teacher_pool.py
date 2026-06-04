from pathlib import Path
from uuid import uuid4

import pytest

from post_train.scripts.data.build_teacher_pool import (
    atomic_write_jsonl,
    build_teacher_payload,
    collect_processed_ids,
    process_teacher_responses,
    validate_resume_state,
    validate_source_rows,
)
from post_train.src.countdown.io import read_jsonl


def test_collect_processed_ids_includes_accepted_and_rejected_rows():
    accepted = [{"id": "train-000001"}, {"id": "train-000002"}]
    rejected = [{"id": "train-000003"}]

    assert collect_processed_ids(accepted, rejected) == {
        "train-000001",
        "train-000002",
        "train-000003",
    }


def test_validate_resume_state_rejects_duplicate_ids_across_outputs():
    accepted = [{"id": "train-000001"}]
    rejected = [{"id": "train-000001"}]

    with pytest.raises(ValueError, match="duplicate id.*train-000001"):
        validate_resume_state(accepted, rejected, target=10)


def test_validate_resume_state_rejects_accepted_count_over_target():
    accepted = [{"id": "train-000001"}, {"id": "train-000002"}]

    with pytest.raises(ValueError, match="accepted rows.*exceeds target"):
        validate_resume_state(accepted, [], target=1)


def test_validate_source_rows_rejects_duplicate_ids():
    rows = [{"id": "train-000001"}, {"id": "train-000001"}]

    with pytest.raises(ValueError, match="duplicate id.*train-000001"):
        validate_source_rows(rows)


def test_atomic_write_jsonl_replaces_existing_file():
    path = Path("C:/tmp") / f"teacher_accepted_20k_{uuid4().hex}.jsonl"
    temp_path = path.with_name(f"{path.name}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        path.write_text('{"id": "old"}\n', encoding="utf-8")

        atomic_write_jsonl(path, [{"id": "new"}])

        assert read_jsonl(path) == [{"id": "new"}]
        assert not temp_path.exists()
    finally:
        if path.exists():
            path.unlink()
        if temp_path.exists():
            temp_path.unlink()


def test_build_teacher_payload_preserves_row_and_records_validation():
    row = {
        "id": "train-000001",
        "numbers": [7, 3, 8, 2],
        "target": 24,
        "prompt": "solve",
    }

    payload = build_teacher_payload(
        row,
        "reasoning\n<answer> (7-3)*(8-2) </answer>\n",
    )

    assert payload["id"] == row["id"]
    assert payload["prompt"] == row["prompt"]
    assert payload["response"] == "reasoning\n<answer> (7-3)*(8-2) </answer>"
    assert payload["teacher_expr"] == "(7-3)*(8-2)"
    assert payload["validation"] == {
        "ok": True,
        "error": None,
        "value": 24,
    }


def test_process_teacher_responses_stops_before_validating_after_target():
    rows = [
        {
            "id": "train-000001",
            "numbers": [7, 3, 8, 2],
            "target": 24,
            "prompt": "solve",
        },
        {
            "id": "train-000002",
            "numbers": [1, 2, 3, 4],
            "target": 99,
            "prompt": "solve",
        },
    ]
    responses = [
        "<answer> (7-3)*(8-2) </answer>",
        "<answer> 1+2+3+4 </answer>",
    ]
    accepted = []
    rejected = []
    processed_ids = set()

    process_teacher_responses(
        rows=rows,
        responses=responses,
        accepted=accepted,
        rejected=rejected,
        processed_ids=processed_ids,
        target=1,
    )

    assert [row["id"] for row in accepted] == ["train-000001"]
    assert rejected == []
    assert processed_ids == {"train-000001"}
