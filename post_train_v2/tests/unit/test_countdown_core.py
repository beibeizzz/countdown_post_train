from fractions import Fraction

import pytest

from post_train_v2.src.countdown.bucketing import assign_bucket
from post_train_v2.src.countdown.prompts import (
    build_chat_messages,
    build_dpo_forced_wrong_prompt,
    build_solution_prompt,
)
from post_train_v2.src.countdown.solver import expression_metadata, solve_countdown
from post_train_v2.src.countdown.validation import (
    extract_answer_text,
    has_complete_answer_tag,
    serialize_fraction,
    validate_countdown_expression,
    validate_countdown_response,
)


def test_solution_prompt_contract():
    prompt = build_solution_prompt([1, 1, 1, 1], 4)

    assert "Use each number exactly once" in prompt
    assert "Only use +, -, *, / and parentheses" in prompt
    assert "<answer> equation </answer>" in prompt
    assert "Division must be exact" not in prompt


def test_forced_wrong_prompt_and_chat_messages():
    chosen = "Reasoning. <answer> ((7-3)*(8-2)) </answer>"
    prompt = build_dpo_forced_wrong_prompt([7, 3, 8, 2], 24, chosen)

    assert "plausible but mathematically wrong" in prompt
    assert "Use the same numbers exactly once" in prompt
    assert "different from 24" in prompt
    assert chosen in prompt
    assert build_chat_messages(prompt) == [{"role": "user", "content": prompt}]


def test_extract_and_complete_answer_apis_use_last_complete_tag():
    text = "draft <answer> 1+1 </answer>\nfinal <answer> (7-3)*(8-2) </answer>"

    assert extract_answer_text(text) == "(7-3)*(8-2)"
    assert has_complete_answer_tag(text) is True
    assert has_complete_answer_tag("<answer> 1+1") is False


def test_fractional_intermediate_is_valid():
    result = validate_countdown_expression(
        "(85-(45/(69-74)))",
        [85, 45, 69, 74],
        94,
    )

    assert result.ok is True
    assert result.value == Fraction(94, 1)
    assert result.error is None


@pytest.mark.parametrize(
    "expression,numbers,target",
    [
        ("True", [1], 1),
        ("1.5", [1], 1),
        ("-1", [1], -1),
        ("2**3", [2, 3], 8),
        ("abs(1)", [1], 1),
        ("1/0", [1, 0], 1),
    ],
)
def test_validation_rejects_invalid_expression_forms(expression, numbers, target):
    result = validate_countdown_expression(expression, numbers, target)

    assert result.ok is False
    assert result.value is None
    assert result.error == "invalid_expression"


def test_validation_preserves_number_mismatch_and_wrong_value_semantics():
    mismatch = validate_countdown_expression("(7-3)*6", [7, 3, 8, 2], 24)
    repeated = validate_countdown_expression("1+1+3", [1, 2, 3], 5)
    wrong = validate_countdown_expression("(7-3)*(8-2)", [7, 3, 8, 2], 25)

    assert mismatch.error == "number_mismatch"
    assert mismatch.value == Fraction(24, 1)
    assert repeated.error == "number_mismatch"
    assert repeated.used_numbers == [1, 1, 3]
    assert wrong.error == "wrong_value"
    assert wrong.value == Fraction(24, 1)


def test_response_requires_complete_answer_tag():
    missing = validate_countdown_response("answer: 1+1", [1, 1], 2)
    valid = validate_countdown_response("<answer> 1+1 </answer>", [1, 1], 2)

    assert missing.error == "missing_answer_tag"
    assert valid.ok is True
    assert valid.value == Fraction(2, 1)


def test_fraction_serialization_is_artifact_safe():
    assert serialize_fraction(None) is None
    assert serialize_fraction(Fraction(94, 1)) == "94/1"
    assert serialize_fraction(Fraction(-3, 7)) == "-3/7"


def test_solver_returns_fully_parenthesized_valid_gold_expression():
    expression = solve_countdown([3, 3, 8, 8], 24)

    assert expression is not None
    assert expression.startswith("(")
    assert expression.endswith(")")
    assert expression.count("(") == 3
    assert expression.count(")") == 3
    assert validate_countdown_expression(expression, [3, 3, 8, 8], 24).ok


def test_solver_returns_none_for_unsolved_instance():
    assert solve_countdown([1, 1], 3) is None


def test_expression_metadata_and_bucketing_preserve_v1_semantics():
    expression = "(100+((75/(23-15))+6))"
    metadata = expression_metadata(expression, num_count=5)
    bucket = assign_bucket([100, 75, 23, 15, 6], expression)

    assert metadata == {
        "num_count": 5,
        "expr_depth": 5,
        "expr_len": len(expression),
        "has_division": True,
        "has_subtraction": True,
    }
    assert bucket["complexity"] == "hard"
    assert bucket["bucket_key"] == "5_hard"
    assert bucket["score"] >= 4
