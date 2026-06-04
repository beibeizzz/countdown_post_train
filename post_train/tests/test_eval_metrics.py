from post_train.src.countdown.eval import aggregate_eval_rows, score_generation
from post_train.scripts.eval.evaluate_model import is_truncated


def test_score_generation_marks_correct_response_valid():
    row = {
        "id": "row-1",
        "prompt": "Make 24 from 7, 3, 8, 2.",
        "numbers": [7, 3, 8, 2],
        "target": 24,
    }

    scored = score_generation(
        row,
        "reasoning\n<answer>(7-3)*(8-2)</answer>",
        generated_tokens=12,
        truncated=False,
    )

    assert scored["id"] == "row-1"
    assert scored["prompt"] == row["prompt"]
    assert scored["raw_generation"] == "reasoning\n<answer>(7-3)*(8-2)</answer>"
    assert scored["extracted_expr"] == "(7-3)*(8-2)"
    assert scored["format_ok"] is True
    assert scored["valid"] is True
    assert scored["correct"] is True
    assert scored["error"] is None
    assert scored["generated_tokens"] == 12
    assert scored["truncated"] is False


def test_aggregate_eval_rows_counts_accuracy_format_and_truncation():
    rows = [
        {
            "correct": True,
            "format_ok": True,
            "valid": True,
            "generated_tokens": 10,
            "truncated": False,
        },
        {
            "correct": False,
            "format_ok": False,
            "valid": False,
            "generated_tokens": 20,
            "truncated": True,
        },
    ]

    metrics = aggregate_eval_rows(rows)

    assert metrics["accuracy"] == 0.5
    assert metrics["format_rate"] == 0.5
    assert metrics["valid_expression_rate"] == 0.5
    assert metrics["avg_generated_tokens"] == 15
    assert metrics["max_generated_tokens"] == 20
    assert metrics["truncated_count"] == 1


def test_aggregate_eval_rows_empty_returns_zero_metrics():
    assert aggregate_eval_rows([]) == {
        "accuracy": 0,
        "format_rate": 0,
        "valid_expression_rate": 0,
        "avg_generated_tokens": 0,
        "max_generated_tokens": 0,
        "truncated_count": 0,
    }


def test_score_generation_reports_wrong_value():
    row = {
        "id": "row-2",
        "prompt": "Make 25 from 7, 3, 8, 2.",
        "numbers": [7, 3, 8, 2],
        "target": 25,
    }

    scored = score_generation(row, "<answer>(7-3)*(8-2)</answer>", generated_tokens=8, truncated=True)

    assert scored["format_ok"] is True
    assert scored["valid"] is False
    assert scored["correct"] is False
    assert scored["error"] == "wrong_value"
    assert scored["truncated"] is True


def test_is_truncated_requires_max_length_without_eos():
    assert is_truncated([1, 2, 3], max_new_tokens=3, eos_token_id=None) is True
    assert is_truncated([1, 2], max_new_tokens=3, eos_token_id=None) is False


def test_is_truncated_handles_int_and_list_eos_ids():
    assert is_truncated([1, 2, 99], max_new_tokens=3, eos_token_id=99) is False
    assert is_truncated([1, 2, 99], max_new_tokens=3, eos_token_id=[98, 99]) is False
    assert is_truncated([1, 2, 3], max_new_tokens=3, eos_token_id=[98, 99]) is True
