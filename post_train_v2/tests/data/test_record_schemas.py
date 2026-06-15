from copy import deepcopy
from fractions import Fraction

import pytest

from post_train_v2.src.data import (
    validate_dpo_record,
    validate_normalized_source,
    validate_sft_record,
    validate_unique_ids,
    validate_verl_record,
)


def normalized_source() -> dict:
    return {
        "id": "train-000001",
        "source_index": 1,
        "numbers": [1, 2],
        "target": 3,
        "gold_expr": "1+2",
        "prompt": "Use 1 and 2 to make 3.",
        "bucket": {
            "num_count": 2,
            "expr_depth": 2,
            "expr_len": 3,
            "has_division": False,
            "has_subtraction": False,
            "score": 0,
            "complexity": "easy",
            "bucket_key": "2_easy",
        },
    }


def validation_result() -> dict:
    return {
        "ok": True,
        "value": "3/1",
        "used_numbers": [1, 2],
        "expression": "1+2",
        "error": None,
    }


def provenance() -> dict:
    return {
        "stage": "teacher",
        "attempt": 0,
        "metrics": [1, 0.5, None, True],
    }


def sft_record() -> dict:
    return {
        **normalized_source(),
        "response": "Reasoning\n<answer>1+2</answer>",
        "validation": validation_result(),
        "provenance": provenance(),
    }


def dpo_record(category: str = "wrong_value") -> dict:
    return {
        "prompt": "Use 1 and 2 to make 3.",
        "chosen": "<answer>1+2</answer>",
        "rejected": "<answer>1*2</answer>",
        "rejected_category": category,
        "generation_route": "forced_wrong",
        "provenance": {
            **provenance(),
            "problem_id": "train-000001",
            "candidate_id": "candidate-000001",
        },
    }


def verl_record() -> dict:
    return {
        "data_source": "countdown",
        "prompt": [
            {"role": "system", "content": "Solve exactly."},
            {"role": "user", "content": "Use 1 and 2 to make 3."},
        ],
        "ability": "countdown",
        "reward_model": {
            "style": "rule",
            "ground_truth": {"numbers": [1, 2], "target": 3},
        },
        "extra_info": {
            "id": "train-000001",
            "source_index": 1,
            "tags": ["easy", None],
        },
    }


def assert_deep_copy(original: dict, result: dict, nested_path: tuple[str, ...]) -> None:
    assert result == original
    assert result is not original
    original_nested = original
    result_nested = result
    for key in nested_path:
        original_nested = original_nested[key]
        result_nested = result_nested[key]
    assert result_nested is not original_nested


def test_normalized_source_returns_canonical_deep_copy_without_mutating_input():
    row = normalized_source()
    before = deepcopy(row)

    result = validate_normalized_source(row)

    assert row == before
    assert_deep_copy(row, result, ("bucket",))
    assert result["numbers"] is not row["numbers"]
    result["bucket"]["complexity"] = "hard"
    result["numbers"].append(99)
    assert row == before


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda row: row.pop("source_index"), "source_index"),
        (lambda row: row.update({"unexpected": True}), "keys"),
        (lambda row: row.update({"id": ""}), "id"),
        (lambda row: row.update({"source_index": True}), "source_index"),
        (lambda row: row.update({"source_index": 1.0}), "source_index"),
        (lambda row: row.update({"numbers": []}), "numbers"),
        (lambda row: row.update({"numbers": [1, True]}), "numbers"),
        (lambda row: row.update({"numbers": [1, 2.0]}), "numbers"),
        (lambda row: row.update({"target": -1}), "target"),
        (lambda row: row.update({"prompt": ""}), "prompt"),
    ],
)
def test_normalized_source_rejects_missing_extra_and_wrong_types(mutation, match):
    row = normalized_source()
    mutation(row)

    with pytest.raises(ValueError, match=match):
        validate_normalized_source(row)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda bucket: bucket.update({"typo": 1}), "bucket keys"),
        (lambda bucket: bucket.update({"expr_depth": True}), "expr_depth"),
        (lambda bucket: bucket.update({"expr_len": 3.0}), "expr_len"),
        (lambda bucket: bucket.update({"has_division": 0}), "has_division"),
        (lambda bucket: bucket.update({"score": -1}), "score"),
        (lambda bucket: bucket.update({"complexity": "unknown"}), "complexity"),
        (lambda bucket: bucket.update({"bucket_key": "2_hard"}), "bucket_key"),
    ],
)
def test_normalized_source_enforces_bucket_types_and_consistency(mutation, match):
    row = normalized_source()
    mutation(row["bucket"])

    with pytest.raises(ValueError, match=match):
        validate_normalized_source(row)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("num_count", 3),
        ("expr_depth", 99),
        ("expr_len", 99),
        ("has_division", True),
        ("has_subtraction", True),
        ("score", 1),
        ("complexity", "medium"),
        ("bucket_key", "2_medium"),
    ],
)
def test_normalized_source_rejects_each_forged_bucket_dimension(field, value):
    row = normalized_source()
    row["bucket"][field] = value

    with pytest.raises(ValueError, match="canonical bucket"):
        validate_normalized_source(row)


@pytest.mark.parametrize(
    ("gold_expr", "match"),
    [
        ("1+", "gold_expr.*invalid_expression"),
        ("1+1", "gold_expr.*number_mismatch"),
        ("1*2", "gold_expr.*wrong_value"),
    ],
)
def test_normalized_source_requires_gold_expr_to_be_a_correct_solution(
    gold_expr, match
):
    row = normalized_source()
    row["gold_expr"] = gold_expr

    with pytest.raises(ValueError, match=match):
        validate_normalized_source(row)


def test_normalized_source_is_not_restricted_to_arrow_int64():
    large = 2**80
    row = normalized_source()
    row.update(
        {
            "numbers": [large, 1],
            "target": large + 1,
            "gold_expr": f"{large}+1",
        }
    )
    row["bucket"] = {
        "num_count": 2,
        "expr_depth": 2,
        "expr_len": len(row["gold_expr"]),
        "has_division": False,
        "has_subtraction": False,
        "score": 1,
        "complexity": "easy",
        "bucket_key": "2_easy",
    }

    assert validate_normalized_source(row) == row


def test_sft_record_returns_deep_canonical_copy():
    row = sft_record()
    before = deepcopy(row)

    result = validate_sft_record(row)

    assert row == before
    assert_deep_copy(row, result, ("validation",))
    assert result["provenance"] is not row["provenance"]
    assert result["provenance"]["metrics"] is not row["provenance"]["metrics"]
    result["validation"]["used_numbers"].append(99)
    result["provenance"]["metrics"].append("changed")
    assert row == before


@pytest.mark.parametrize("value", ["48/2", "3/0", "3", "03/1", "3/-1"])
def test_sft_validation_requires_reduced_canonical_fraction_strings(value):
    row = sft_record()
    row["validation"]["value"] = value

    with pytest.raises(ValueError, match="validation.value"):
        validate_sft_record(row)


@pytest.mark.parametrize(
    "validation",
    [
        {
            "ok": True,
            "value": None,
            "used_numbers": [1, 2],
            "expression": "1+2",
            "error": None,
        },
        {
            "ok": True,
            "value": "3/1",
            "used_numbers": [1, 2],
            "expression": None,
            "error": None,
        },
        {
            "ok": True,
            "value": "3/1",
            "used_numbers": [1, 2],
            "expression": "1+2",
            "error": "wrong_value",
        },
        {
            "ok": False,
            "value": None,
            "used_numbers": [],
            "expression": None,
            "error": None,
        },
    ],
)
def test_sft_validation_enforces_success_and_failure_semantics(validation):
    row = sft_record()
    row["validation"] = validation

    with pytest.raises(ValueError, match="validation"):
        validate_sft_record(row)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ok", 1),
        ("used_numbers", [1, 2.0]),
        ("expression", 3),
        ("error", "truncated"),
    ],
)
def test_sft_validation_rejects_wrong_primitive_types(field, value):
    row = sft_record()
    row["validation"][field] = value

    with pytest.raises(ValueError, match=f"validation.{field}"):
        validate_sft_record(row)


def test_sft_validation_rejects_expression_falsely_declared_correct():
    row = sft_record()
    row["validation"]["expression"] = "1*2"

    with pytest.raises(ValueError, match="validation.*actual"):
        validate_sft_record(row)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ok", True),
        ("value", "9/1"),
        ("used_numbers", [2, 1]),
        ("error", "number_mismatch"),
    ],
)
def test_sft_validation_declared_fields_must_match_expression_result(field, value):
    row = sft_record()
    row["validation"] = {
        "ok": False,
        "value": "2/1",
        "used_numbers": [1, 2],
        "expression": "1*2",
        "error": "wrong_value",
    }
    row["validation"][field] = value

    with pytest.raises(ValueError, match=f"validation.{field}.*actual"):
        validate_sft_record(row)


@pytest.mark.parametrize(
    "validation",
    [
        {
            "ok": False,
            "value": "2/1",
            "used_numbers": [1, 2],
            "expression": "1*2",
            "error": "wrong_value",
        },
        {
            "ok": False,
            "value": None,
            "used_numbers": [],
            "expression": "1+",
            "error": "invalid_expression",
        },
        {
            "ok": False,
            "value": None,
            "used_numbers": [],
            "expression": None,
            "error": "missing_answer_tag",
        },
    ],
)
def test_sft_validation_accepts_exact_v2_validator_results(validation):
    row = sft_record()
    row["validation"] = validation

    assert validate_sft_record(row) == row


@pytest.mark.parametrize(
    "bad_value",
    [
        Fraction(1, 2),
        b"bytes",
        float("nan"),
        float("inf"),
        {"nested": Fraction(1, 2)},
        {1: "non-string-key"},
        ("tuple",),
    ],
)
def test_sft_provenance_rejects_non_json_primitives(bad_value):
    row = sft_record()
    row["provenance"]["bad"] = bad_value

    with pytest.raises(ValueError, match="provenance"):
        validate_sft_record(row)


def test_sft_provenance_remains_json_friendly_not_arrow_restricted():
    row = sft_record()
    row["provenance"]["heterogeneous"] = [1, "x", {"nested": True}]

    assert validate_sft_record(row) == row


@pytest.mark.parametrize(
    "category",
    [
        "wrong_value",
        "number_mismatch",
        "invalid_expression",
        "missing_answer_tag",
        "truncated",
    ],
)
def test_dpo_accepts_exact_rejected_category_vocabulary(category):
    row = dpo_record(category)

    result = validate_dpo_record(row)

    assert result == row
    assert set(result) == {
        "prompt",
        "chosen",
        "rejected",
        "rejected_category",
        "generation_route",
        "provenance",
    }


def test_dpo_rejects_unknown_category_and_identical_responses():
    with pytest.raises(ValueError, match="rejected_category"):
        validate_dpo_record(dpo_record("wrong-answer"))

    row = dpo_record()
    row["rejected"] = row["chosen"]
    with pytest.raises(ValueError, match="chosen"):
        validate_dpo_record(row)


def test_dpo_requires_exact_keys_nonempty_strings_and_deep_copies_provenance():
    row = dpo_record()
    row["extra"] = "typo"
    with pytest.raises(ValueError, match="keys"):
        validate_dpo_record(row)

    for field in (
        "prompt",
        "chosen",
        "rejected",
        "rejected_category",
        "generation_route",
    ):
        row = dpo_record()
        row[field] = ""
        with pytest.raises(ValueError, match=field):
            validate_dpo_record(row)

    row = dpo_record()
    result = validate_dpo_record(row)
    assert result["provenance"] is not row["provenance"]
    assert result["provenance"]["metrics"] is not row["provenance"]["metrics"]


def test_dpo_rejects_top_level_id_even_when_it_matches_provenance():
    row = dpo_record()
    row["id"] = row["provenance"]["problem_id"]

    with pytest.raises(ValueError, match="DPO record keys"):
        validate_dpo_record(row)


def test_verl_accepts_structured_chat_and_ground_truth_with_deep_copy():
    row = verl_record()
    before = deepcopy(row)

    result = validate_verl_record(row)

    assert row == before
    assert_deep_copy(row, result, ("reward_model",))
    assert result["prompt"] is not row["prompt"]
    assert result["prompt"][0] is not row["prompt"][0]
    assert result["extra_info"] is not row["extra_info"]
    result["reward_model"]["ground_truth"]["numbers"].append(99)
    result["prompt"][0]["content"] = "changed"
    assert row == before


def test_canonical_verl_record_converts_to_pyarrow_table():
    pyarrow = pytest.importorskip("pyarrow")
    row = validate_verl_record(verl_record())

    table = pyarrow.Table.from_pylist([row])

    assert table.num_rows == 1
    assert table.column_names == list(row)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("numbers", [1, 2**80]),
        ("target", 2**80),
    ],
)
def test_verl_ground_truth_rejects_values_outside_arrow_int64(field, value):
    row = verl_record()
    row["reward_model"]["ground_truth"][field] = value

    with pytest.raises(ValueError, match=f"ground_truth.{field}.*int64"):
        validate_verl_record(row)


def test_verl_ground_truth_int64_boundary_converts_with_pyarrow():
    pyarrow = pytest.importorskip("pyarrow")
    row = verl_record()
    maximum = 2**63 - 1
    row["reward_model"]["ground_truth"] = {
        "numbers": [0, maximum],
        "target": maximum,
    }

    canonical = validate_verl_record(row)
    table = pyarrow.Table.from_pylist([canonical])

    assert table.to_pylist() == [canonical]


@pytest.mark.parametrize(
    "bad_value",
    [
        [1, "x"],
        [{"value": 1}, {"value": "x"}],
        [[1], ["x"]],
        [True, 1],
    ],
)
def test_verl_extra_info_rejects_incompatible_arrow_list_types(bad_value):
    row = verl_record()
    row["extra_info"]["bad"] = bad_value

    with pytest.raises(ValueError, match="extra_info.*Arrow"):
        validate_verl_record(row)


def test_verl_extra_info_accepts_arrow_mergeable_numeric_and_null_lists():
    row = verl_record()
    row["extra_info"]["numeric"] = [1, 2.5, None]
    row["extra_info"]["objects"] = [
        {"value": 1, "label": None},
        {"value": 2.5, "label": "x"},
    ]

    result = validate_verl_record(row)

    assert result == row


def test_verl_extra_info_merges_mapping_list_keys_as_nullable_arrow_struct():
    pyarrow = pytest.importorskip("pyarrow")
    row = verl_record()
    row["extra_info"]["items"] = [{"left": 1}, {"right": 2}]

    canonical = validate_verl_record(row)
    table = pyarrow.Table.from_pylist([canonical])

    assert table.to_pylist()[0]["extra_info"]["items"] == [
        {"left": 1, "right": None},
        {"left": None, "right": 2},
    ]


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda row: row.update({"extra": 1}), "keys"),
        (lambda row: row.update({"data_source": ""}), "data_source"),
        (lambda row: row.update({"prompt": []}), "prompt"),
        (
            lambda row: row["prompt"][0].update({"name": "extra"}),
            "prompt.*keys",
        ),
        (lambda row: row["prompt"][0].update({"role": ""}), "role"),
        (lambda row: row["prompt"][0].update({"content": 1}), "content"),
        (
            lambda row: row["reward_model"].update({"typo": True}),
            "reward_model keys",
        ),
        (
            lambda row: row["reward_model"]["ground_truth"].update({"extra": 1}),
            "ground_truth keys",
        ),
        (
            lambda row: row["reward_model"]["ground_truth"].update(
                {"numbers": [1, True]}
            ),
            "ground_truth.numbers",
        ),
        (
            lambda row: row["reward_model"]["ground_truth"].update({"target": 3.0}),
            "ground_truth.target",
        ),
    ],
)
def test_verl_rejects_malformed_chat_and_ground_truth(mutation, match):
    row = verl_record()
    mutation(row)

    with pytest.raises(ValueError, match=match):
        validate_verl_record(row)


@pytest.mark.parametrize(
    "bad_value",
    [
        Fraction(1, 2),
        b"bytes",
        float("nan"),
        float("-inf"),
        {"nested": Fraction(1, 2)},
        {1: "non-string-key"},
        ("tuple",),
    ],
)
def test_verl_extra_info_rejects_non_json_or_arrow_friendly_values(bad_value):
    row = verl_record()
    row["extra_info"]["bad"] = bad_value

    with pytest.raises(ValueError, match="extra_info"):
        validate_verl_record(row)


def test_validate_unique_ids_preserves_order_and_returns_unaliased_rows():
    rows = [
        {"id": "b", "nested": {"values": [2]}},
        {"id": "a", "nested": {"values": [1]}},
    ]
    before = deepcopy(rows)

    result = validate_unique_ids(rows, "training")

    assert result == rows
    assert rows == before
    assert result is not rows
    assert result[0] is not rows[0]
    assert result[0]["nested"] is not rows[0]["nested"]
    result[0]["nested"]["values"].append(99)
    assert rows == before


@pytest.mark.parametrize(
    ("rows", "match"),
    [
        ([{"value": 1}], "training.*missing id"),
        ([{"id": ""}], "training.*id"),
        ([{"id": 1}], "training.*id"),
        ([{"id": "same"}, {"id": "same"}], "training.*duplicate id.*same"),
        (["not-a-mapping"], "training.*mapping"),
    ],
)
def test_validate_unique_ids_reports_missing_invalid_and_duplicate_ids(rows, match):
    with pytest.raises(ValueError, match=match):
        validate_unique_ids(rows, "training")


@pytest.mark.parametrize("rows", [None, {"id": "x"}, "rows"])
def test_validate_unique_ids_requires_a_sequence(rows):
    with pytest.raises(ValueError, match="training.*sequence"):
        validate_unique_ids(rows, "training")
