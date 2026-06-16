from __future__ import annotations

import json

import pytest

from post_train_v2.src.artifacts.manifest import load_manifest
from post_train_v2.src.countdown.bucketing import assign_bucket
from post_train_v2.src.countdown.prompts import build_dpo_forced_wrong_prompt, build_solution_prompt
from post_train_v2.src.generation.dpo import (
    build_dpo_generation_requests,
    publish_dpo_outputs,
    validate_chosen_rows,
)
from post_train_v2.src.generation.metadata import GenerationRecord


def chosen_row(row_id: str = "row-1", response: str = "<answer>1+1+1+1</answer>"):
    numbers = [1, 1, 1, 1]
    target = 4
    gold_expr = "1+1+1+1"
    return {
        "id": row_id,
        "source_index": 1,
        "numbers": numbers,
        "target": target,
        "gold_expr": gold_expr,
        "prompt": build_solution_prompt(numbers, target),
        "bucket": assign_bucket(numbers, gold_expr),
        "response": response,
        "validation": {
            "ok": True,
            "value": "4/1",
            "used_numbers": numbers,
            "expression": gold_expr,
            "error": None,
        },
        "provenance": {"stage": "teacher"},
    }


def test_validate_chosen_rows_rejects_bad_chosen_before_generation():
    with pytest.raises(ValueError, match="chosen response must validate"):
        validate_chosen_rows(
            [chosen_row(response="<answer>1+1+1-1</answer>")]
        )


def test_build_dpo_generation_requests_halves_routes_and_derives_seeds():
    row = chosen_row()
    requests = build_dpo_generation_requests([row], rollouts_per_route=2, seed=123)

    routes = [request.metadata["generation_route"] for request in requests]
    assert routes == ["forced_wrong", "forced_wrong", "high_temp", "high_temp"]
    assert requests[0].prompt == build_dpo_forced_wrong_prompt(
        row["numbers"],
        row["target"],
        row["response"],
    )
    assert requests[2].prompt == row["prompt"]
    assert len({request.seed for request in requests}) == 4
    assert [request.metadata["rollout_index"] for request in requests] == [0, 1, 0, 1]


def test_publish_dpo_outputs_preserves_five_category_names_and_manifest(tmp_path):
    row = chosen_row()
    candidates = [
        {
            "source_id": "row-1",
            "candidate_id": f"c-{index}",
            "generation_route": "forced_wrong",
            "rejected": f"bad-{index}",
            "rejected_category": category,
            "validation": {"ok": False, "error": category},
            "rollout_index": index,
        }
        for index, category in enumerate(
            [
                "wrong_value",
                "number_mismatch",
                "invalid_expression",
                "missing_answer_tag",
                "truncated",
            ]
        )
    ]
    pairs = [
        {
            "prompt": row["prompt"],
            "chosen": row["response"],
            "rejected": candidate["rejected"],
            "rejected_category": candidate["rejected_category"],
            "generation_route": candidate["generation_route"],
            "provenance": {"source_id": "row-1"},
        }
        for candidate in candidates
    ]

    manifest = publish_dpo_outputs(
        output_dir=tmp_path,
        candidates=candidates,
        pairs=pairs,
        config={"seed": 123},
        selection_summary={
            "category_counts": {item["rejected_category"]: 1 for item in candidates},
            "route_counts": {"forced_wrong": 5},
            "shortfall": 0,
        },
    )

    loaded = load_manifest(tmp_path / "manifest.json")
    candidate_rows = [
        json.loads(line)
        for line in (tmp_path / "dpo_candidates.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert loaded.artifact_id == manifest.artifact_id
    assert sorted({row["rejected_category"] for row in candidate_rows}) == [
        "invalid_expression",
        "missing_answer_tag",
        "number_mismatch",
        "truncated",
        "wrong_value",
    ]
