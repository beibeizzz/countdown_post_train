import pytest

from post_train.scripts.dpo.build_dpo_data import (
    Candidate,
    GenerationRecord,
    build_candidates,
    build_manifest,
    build_route_requests,
    candidate_to_row,
    classify_rejected,
    select_dpo_pairs,
    validate_config,
    validate_unique_chosen_ids,
)


def test_classify_rejected_prioritizes_truncation_before_parsing():
    assert classify_rejected("", [1, 2, 3], 6, truncated=True) == "truncated"


def test_classify_rejected_maps_validation_errors_to_required_categories():
    assert classify_rejected("<answer> 1 + 2 + 3 </answer>", [1, 2, 3], 7, False) == "wrong_value"
    assert classify_rejected("<answer> 1 + 2 </answer>", [1, 2, 3], 6, False) == "number_mismatch"
    assert classify_rejected("<answer> 1 + </answer>", [1, 2, 3], 6, False) == "invalid_expression"
    assert classify_rejected("no final answer", [1, 2, 3], 6, False) == "missing_answer_tag"
    assert classify_rejected("<answer> 1 + 2 + 3 </answer>", [1, 2, 3], 6, False) == "unexpected_correct"


def test_select_dpo_pairs_prefers_wrong_value_and_caps_malformed():
    row = {
        "id": "chosen-1",
        "prompt": "solve",
        "response": "chosen",
        "numbers": [1, 2, 3],
        "target": 6,
    }
    candidates = [
        Candidate(source_index=0, source_id="chosen-1", route="high_temp", text="bad missing", category="missing_answer_tag"),
        Candidate(source_index=0, source_id="chosen-1", route="high_temp", text="<answer> 1 + 2 </answer>", category="number_mismatch"),
        Candidate(source_index=0, source_id="chosen-1", route="forced_wrong", text="<answer> 1 + 2 + 3 </answer>", category="unexpected_correct"),
        Candidate(source_index=0, source_id="chosen-1", route="forced_wrong", text="<answer> 1 + 2 - 3 </answer>", category="wrong_value"),
    ]

    pairs = select_dpo_pairs(
        [row],
        candidates,
        target_pairs=3,
        malformed_cap_fraction=0.25,
        preferred_wrong_value_min_fraction=0.70,
    )

    assert [pair["rejected_category"] for pair in pairs] == ["wrong_value", "number_mismatch"]
    assert pairs[0]["rejected"] == "<answer> 1 + 2 - 3 </answer>"
    assert pairs[0]["chosen"] == "chosen"
    assert pairs[0]["prompt"] == "solve"


def test_select_dpo_pairs_uses_limited_malformed_when_stronger_categories_are_exhausted():
    row = {
        "id": "chosen-1",
        "prompt": "solve",
        "response": "chosen",
        "numbers": [1, 2, 3],
        "target": 6,
    }
    candidates = [
        Candidate(source_index=0, source_id="chosen-1", route="high_temp", text="missing", category="missing_answer_tag"),
        Candidate(source_index=0, source_id="chosen-1", route="forced_wrong", text="<answer> 1 + </answer>", category="invalid_expression"),
    ]

    pairs = select_dpo_pairs(
        [row],
        candidates,
        target_pairs=4,
        malformed_cap_fraction=0.50,
        preferred_wrong_value_min_fraction=0.70,
    )

    assert len(pairs) == 2
    assert {pair["rejected_category"] for pair in pairs} == {"invalid_expression", "missing_answer_tag"}


def test_validate_unique_chosen_ids_rejects_duplicates():
    with pytest.raises(ValueError, match="duplicate"):
        validate_unique_chosen_ids([{"id": "same"}, {"id": "same"}])


def make_rows(count):
    return [
        {
            "id": f"chosen-{index}",
            "prompt": f"solve {index}",
            "response": f"chosen {index}",
            "numbers": [1, 2, 3],
            "target": 6,
        }
        for index in range(count)
    ]


def test_build_route_requests_oversamples_beyond_target_when_rows_are_available():
    requests = build_route_requests(
        make_rows(10),
        {
            "target_pairs": 4,
            "forced_wrong_fraction": 0.5,
            "high_temp_fraction": 0.5,
            "candidate_oversample_factor": 2.0,
        },
    )

    assert len(requests) == 8
    assert [request.source_index for request in requests] == [0, 0, 1, 1, 2, 2, 3, 3]
    assert [request.route for request in requests[:2]] == ["forced_wrong", "high_temp"]


def test_build_manifest_reports_wrong_value_minimum_status_and_shortfall():
    cfg = {
        "preferred_wrong_value_min_fraction": 0.75,
        "malformed_cap_fraction": 0.1,
        "target_pairs": 4,
        "max_new_tokens": 256,
        "enable_thinking": False,
    }
    pairs = [
        {"rejected_category": "wrong_value"},
        {"rejected_category": "number_mismatch"},
        {"rejected_category": "number_mismatch"},
        {"rejected_category": "number_mismatch"},
    ]

    manifest = build_manifest(cfg, "model", make_rows(4), [], pairs)

    assert manifest["wrong_value_min_satisfied"] is False
    assert manifest["wrong_value_shortfall"] == 2


def test_validate_config_rejects_invalid_numeric_settings():
    valid = {
        "target_pairs": 10,
        "forced_wrong_fraction": 0.5,
        "high_temp_fraction": 0.5,
        "candidate_oversample_factor": 2.0,
        "forced_wrong_temperature": 0.3,
        "high_temp_temperature": 1.0,
        "top_p": 0.95,
        "max_new_tokens": 256,
        "batch_size": 64,
        "malformed_cap_fraction": 0.1,
        "preferred_wrong_value_min_fraction": 0.7,
    }

    for key, value in [
        ("forced_wrong_fraction", -0.1),
        ("top_p", 1.5),
        ("max_new_tokens", 0),
        ("batch_size", 0),
    ]:
        cfg = dict(valid)
        cfg[key] = value
        with pytest.raises(ValueError, match=key):
            validate_config(cfg)


def test_candidate_log_row_includes_debug_context_and_truncation_evidence():
    rows = make_rows(1)
    records = [
        GenerationRecord(
            text="<answer> 1 + 2 + 3 </answer>",
            finish_reason="length",
            token_count=256,
            truncation_source="finish_reason",
        )
    ]
    requests = build_route_requests(
        rows,
        {
            "target_pairs": 1,
            "forced_wrong_fraction": 1.0,
            "high_temp_fraction": 0.0,
            "candidate_oversample_factor": 1.0,
        },
    )

    candidate = build_candidates(rows, requests, records, max_new_tokens=256)[0]
    row = candidate_to_row(rows, candidate)

    assert row["prompt"] == "solve 0"
    assert row["chosen"] == "chosen 0"
    assert row["numbers"] == [1, 2, 3]
    assert row["target"] == 6
    assert row["truncated"] is True
    assert row["truncation_source"] == "finish_reason"
    assert row["validation"]["extracted_expr"] == "1 + 2 + 3"
