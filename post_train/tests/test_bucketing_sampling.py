import pytest

from post_train.src.countdown import solver
from post_train.src.countdown.bucketing import assign_bucket
from post_train.src.countdown.sampling import stratified_sample
from post_train.src.countdown.solver import expression_metadata, solve_countdown
from post_train.src.countdown.validation import validate_countdown_expression


def test_solve_countdown_finds_valid_expression():
    expr = solve_countdown([7, 3, 8, 2], 24)

    assert expr is not None
    assert validate_countdown_expression(expr, [7, 3, 8, 2], 24).ok is True


def test_solve_countdown_handles_duplicate_numbers():
    expr = solve_countdown([1, 1, 1, 1], 4)

    assert expr is not None
    assert validate_countdown_expression(expr, [1, 1, 1, 1], 4).ok is True


def test_solve_countdown_finds_fraction_intermediate_solution():
    expr = solve_countdown([3, 3, 8, 8], 24)

    assert expr is not None
    assert validate_countdown_expression(expr, [3, 3, 8, 8], 24).ok is True


def test_solve_countdown_caches_repeated_rows(monkeypatch):
    calls = 0
    original_search = solver._search

    def counting_search(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_search(*args, **kwargs)

    monkeypatch.setattr(solver, "_search", counting_search)

    first_expr = solve_countdown([2, 2, 3, 3], 10)
    first_call_count = calls
    second_expr = solve_countdown([3, 2, 3, 2], 10)

    assert first_expr is not None
    assert second_expr == first_expr
    assert first_call_count > 0
    assert calls == first_call_count
    assert validate_countdown_expression(second_expr, [2, 2, 3, 3], 10).ok is True


def test_expression_metadata_detects_division_and_depth():
    meta = expression_metadata("6/(1+1)", num_count=3)

    assert meta["has_division"] is True
    assert meta["expr_depth"] >= 2
    assert meta["expr_len"] == len("6/(1+1)")


def test_assign_bucket_marks_five_number_division_hard():
    bucket = assign_bucket(numbers=[100, 75, 23, 15, 6], expr="(100+75)/(23-15+6)")

    assert bucket["num_count"] == 5
    assert bucket["complexity"] == "hard"


def test_stratified_sample_is_reproducible():
    rows = [
        {"id": "a1", "bucket": {"bucket_key": "3_easy"}},
        {"id": "a2", "bucket": {"bucket_key": "3_easy"}},
        {"id": "b1", "bucket": {"bucket_key": "4_medium"}},
        {"id": "b2", "bucket": {"bucket_key": "4_medium"}},
        {"id": "c1", "bucket": {"bucket_key": "5_hard"}},
        {"id": "c2", "bucket": {"bucket_key": "5_hard"}},
    ]

    first = stratified_sample(rows, size=3, seed=42)
    second = stratified_sample(rows, size=3, seed=42)

    assert first == second
    assert len(first) == 3
    assert {row["bucket"]["bucket_key"] for row in first} == {"3_easy", "4_medium", "5_hard"}


def test_stratified_sample_size_zero_returns_empty():
    rows = [{"bucket": {}}]

    assert stratified_sample(rows, size=0, seed=42) == []


def test_stratified_sample_size_at_least_rows_returns_all_rows_after_validation():
    rows = [
        {"id": "a1", "bucket": {"bucket_key": "3_easy"}},
        {"id": "b1", "bucket": {"bucket_key": "4_medium"}},
    ]

    assert stratified_sample(rows, size=2, seed=42) == rows
    assert stratified_sample(rows, size=3, seed=42) == rows


def test_stratified_sample_missing_bucket_key_raises_value_error():
    rows = [{"id": "a1", "bucket": {}}]

    with pytest.raises(ValueError):
        stratified_sample(rows, size=1, seed=42)


def test_stratified_sample_missing_id_raises_value_error():
    rows = [{"bucket": {"bucket_key": "3_easy"}}]

    with pytest.raises(ValueError):
        stratified_sample(rows, size=1, seed=42)


def test_stratified_sample_id_none_raises_value_error():
    rows = [{"id": None, "bucket": {"bucket_key": "3_easy"}}]

    with pytest.raises(ValueError):
        stratified_sample(rows, size=1, seed=42)


def test_stratified_sample_empty_id_raises_value_error():
    rows = [{"id": "", "bucket": {"bucket_key": "3_easy"}}]

    with pytest.raises(ValueError):
        stratified_sample(rows, size=1, seed=42)


def test_stratified_sample_duplicate_ids_raise_value_error():
    rows = [
        {"id": "a1", "bucket": {"bucket_key": "3_easy"}},
        {"id": "a1", "bucket": {"bucket_key": "4_medium"}},
    ]

    with pytest.raises(ValueError):
        stratified_sample(rows, size=1, seed=42)


def test_stratified_sample_bucket_key_none_raises_value_error():
    rows = [{"id": "a1", "bucket": {"bucket_key": None}}]

    with pytest.raises(ValueError):
        stratified_sample(rows, size=1, seed=42)


def test_stratified_sample_empty_bucket_key_raises_value_error():
    rows = [{"id": "a1", "bucket": {"bucket_key": ""}}]

    with pytest.raises(ValueError):
        stratified_sample(rows, size=1, seed=42)


def test_stratified_sample_underfilled_bucket_returns_requested_size_without_duplicate_ids():
    rows = [
        {"id": "a1", "bucket": {"bucket_key": "3_easy"}},
        {"id": "b1", "bucket": {"bucket_key": "4_medium"}},
        {"id": "b2", "bucket": {"bucket_key": "4_medium"}},
        {"id": "b3", "bucket": {"bucket_key": "4_medium"}},
        {"id": "b4", "bucket": {"bucket_key": "4_medium"}},
    ]

    sample = stratified_sample(rows, size=4, seed=42)
    sample_ids = [row["id"] for row in sample]

    assert len(sample) == 4
    assert len(sample_ids) == len(set(sample_ids))
