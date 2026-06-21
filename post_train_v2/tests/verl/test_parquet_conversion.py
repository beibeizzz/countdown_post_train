from __future__ import annotations

import pytest

from post_train_v2.src.countdown.bucketing import assign_bucket
from post_train_v2.src.countdown.prompts import build_solution_prompt
from post_train_v2.verl.data.conversion import (
    convert_source_rows,
    source_to_verl_record,
    validate_unique_verl_ids,
)


def source_row(row_id: str = "row-1", source_index: int = 1) -> dict:
    numbers = [1, 1, 1, 1]
    target = 4
    gold_expr = "1+1+1+1"
    return {
        "id": row_id,
        "source_index": source_index,
        "numbers": numbers,
        "target": target,
        "gold_expr": gold_expr,
        "prompt": build_solution_prompt(numbers, target),
        "bucket": assign_bucket(numbers, gold_expr),
    }


def test_source_to_verl_record_matches_expected_schema():
    source = source_row()

    record = source_to_verl_record(source)

    assert record == {
        "data_source": "countdown",
        "prompt": [{"role": "user", "content": source["prompt"]}],
        "ability": "arithmetic",
        "reward_model": {
            "style": "rule",
            "ground_truth": {"numbers": source["numbers"], "target": source["target"]},
        },
        "extra_info": {
            "id": source["id"],
            "source_index": source["source_index"],
            "bucket": source["bucket"],
            "gold_expr": source["gold_expr"],
        },
    }


def test_convert_source_rows_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="duplicate id"):
        validate_unique_verl_ids([source_row("dup"), source_row("dup")])


def test_convert_source_rows_rejects_row_count_mismatch():
    with pytest.raises(ValueError, match="row-count mismatch"):
        convert_source_rows([source_row("a")], expected_count=2)


def test_convert_source_rows_validates_arrow_friendly_schema():
    rows = convert_source_rows([source_row("a")], expected_count=1)

    assert rows[0]["data_source"] == "countdown"
    assert rows[0]["reward_model"]["style"] == "rule"
