import post_train.src.countdown.validation as validation
from post_train.src.countdown.validation import (
    extract_answer_text,
    validate_countdown_expression,
    validate_countdown_response,
)


def test_extract_answer_text_returns_last_answer():
    text = "draft <answer> 1+1 </answer>\nfinal <answer> (7-3)*(8-2) </answer>"

    assert extract_answer_text(text) == "(7-3)*(8-2)"


def test_validation_accepts_correct_expression():
    result = validate_countdown_expression("(7-3)*(8-2)", [7, 3, 8, 2], 24)

    assert result.ok is True
    assert result.value == 24
    assert result.error is None


def test_validation_allows_intermediate_fraction():
    result = validate_countdown_expression("6/(1+1)", [6, 1, 1], 3)

    assert result.ok is True
    assert result.value == 3


def test_validation_rejects_number_mismatch():
    result = validate_countdown_expression("(7-3)*6", [7, 3, 8, 2], 24)

    assert result.ok is False
    assert result.error == "number_mismatch"


def test_validation_rejects_wrong_value():
    result = validate_countdown_expression("(7-3)*(8-2)", [7, 3, 8, 2], 25)

    assert result.ok is False
    assert result.error == "wrong_value"


def test_validation_rejects_unsupported_syntax():
    result = validate_countdown_expression("__import__('os').system('echo bad')", [1], 1)

    assert result.ok is False
    assert result.error == "invalid_expression"


def test_validation_response_rejects_missing_answer_tag():
    result = validate_countdown_response("no answer here", [1], 1)

    assert result.ok is False
    assert result.error == "missing_answer_tag"


def test_validation_rejects_bool_constant():
    result = validate_countdown_expression("True", [1], 1)

    assert result.ok is False
    assert result.error == "invalid_expression"


def test_validation_rejects_float_constant():
    result = validate_countdown_expression("1.5", [1], 1)

    assert result.ok is False
    assert result.error == "invalid_expression"


def test_validation_rejects_unary_minus():
    result = validate_countdown_expression("-1", [1], -1)

    assert result.ok is False
    assert result.error == "invalid_expression"


def test_validation_rejects_division_by_zero():
    result = validate_countdown_expression("1/0", [1, 0], 1)

    assert result.ok is False
    assert result.error == "invalid_expression"


def test_validation_rejects_recursion_error(monkeypatch):
    def raise_recursion_error(node):
        raise RecursionError

    monkeypatch.setattr(validation, "_eval_node", raise_recursion_error)

    result = validate_countdown_expression("1", [1], 1)

    assert result.ok is False
    assert result.value is None
    assert result.used_numbers == []
    assert result.expression == "1"
    assert result.error == "invalid_expression"
