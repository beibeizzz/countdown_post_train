from copy import deepcopy
import json
import random

import pytest

from post_train_v2.src.countdown.sampling import (
    build_validation_splits,
    exclude_ids,
    stratified_sample,
)


def _rows(count: int = 12) -> list[dict]:
    bucket_keys = ("3_easy", "4_medium", "5_hard")
    return [
        {
            "id": f"row-{index:02d}",
            "source_index": index,
            "bucket": {"bucket_key": bucket_keys[index % len(bucket_keys)]},
        }
        for index in range(count)
    ]


def _ids(rows: list[dict]) -> list[str]:
    return [row["id"] for row in rows]


def test_stratified_sample_is_balanced_and_seed_deterministic():
    rows = _rows()

    first = stratified_sample(rows, size=6, seed=42)
    second = stratified_sample(rows, size=6, seed=42)

    assert json.dumps(_ids(first)) == json.dumps(_ids(second))
    assert {
        key: sum(row["bucket"]["bucket_key"] == key for row in first)
        for key in ("3_easy", "4_medium", "5_hard")
    } == {"3_easy": 2, "4_medium": 2, "5_hard": 2}


def test_stratified_sample_is_independent_of_input_order():
    rows = _rows()

    forward = stratified_sample(rows, size=7, seed=9)
    reversed_input = stratified_sample(list(reversed(rows)), size=7, seed=9)

    assert _ids(forward) == _ids(reversed_input)


def test_sampling_does_not_modify_input_rows():
    rows = _rows()
    original = deepcopy(rows)

    stratified_sample(rows, size=5, seed=1)

    assert rows == original


def test_sampling_results_do_not_share_mutable_objects_with_input():
    rows = _rows()
    sample = stratified_sample(rows, size=5, seed=1)
    sampled_id = sample[0]["id"]
    input_row = next(row for row in rows if row["id"] == sampled_id)

    sample[0]["source_index"] = -1
    sample[0]["bucket"]["bucket_key"] = "mutated"

    assert input_row["source_index"] != -1
    assert input_row["bucket"]["bucket_key"] != "mutated"


def test_validation_split_results_do_not_share_mutable_objects():
    rows = _rows()
    splits = build_validation_splits(
        rows,
        validation_size=8,
        eval_size=3,
        seed=17,
    )
    shared_id = splits.eval_rows[0]["id"]
    val_row = next(row for row in splits.val_rows if row["id"] == shared_id)
    input_row = next(row for row in rows if row["id"] == shared_id)
    train_row = splits.train_candidates[0]
    train_input_row = next(row for row in rows if row["id"] == train_row["id"])

    splits.eval_rows[0]["source_index"] = -1
    splits.eval_rows[0]["bucket"]["bucket_key"] = "mutated"
    train_row["source_index"] = -2
    train_row["bucket"]["bucket_key"] = "train-mutated"

    assert val_row["source_index"] != -1
    assert val_row["bucket"]["bucket_key"] != "mutated"
    assert input_row["source_index"] != -1
    assert input_row["bucket"]["bucket_key"] != "mutated"
    assert train_input_row["source_index"] != -2
    assert train_input_row["bucket"]["bucket_key"] != "train-mutated"


def test_seed_determines_remainder_bucket_order_without_lexical_bias():
    rows = _rows(3)
    bucket_keys = sorted(row["bucket"]["bucket_key"] for row in rows)
    selected_by_seed = {}

    for seed in range(6):
        expected_order = list(bucket_keys)
        random.Random(seed).shuffle(expected_order)
        selected_bucket = stratified_sample(rows, size=1, seed=seed)[0]["bucket"][
            "bucket_key"
        ]
        selected_by_seed[seed] = selected_bucket
        assert selected_bucket == expected_order[0]

    assert set(selected_by_seed.values()) == set(bucket_keys)


@pytest.mark.parametrize("size", [-1, 13])
def test_stratified_sample_rejects_out_of_range_size(size):
    with pytest.raises(ValueError, match="size"):
        stratified_sample(_rows(), size=size, seed=1)


def test_stratified_sample_allows_zero_and_full_size():
    rows = _rows(3)

    assert stratified_sample(rows, size=0, seed=1) == []
    assert _ids(stratified_sample(rows, size=3, seed=1)) == [
        "row-00",
        "row-01",
        "row-02",
    ]


@pytest.mark.parametrize(
    "rows,match",
    [
        ([{"bucket": {"bucket_key": "3_easy"}}], "missing id"),
        ([{"id": "", "bucket": {"bucket_key": "3_easy"}}], "missing id"),
        (
            [
                {"id": "same", "bucket": {"bucket_key": "3_easy"}},
                {"id": "same", "bucket": {"bucket_key": "4_medium"}},
            ],
            "duplicate row id",
        ),
        ([{"id": "x", "bucket": {}}], "bucket.bucket_key"),
        ([{"id": "x", "bucket": {"bucket_key": ""}}], "bucket.bucket_key"),
    ],
)
def test_sampling_rejects_missing_or_duplicate_identity_fields(rows, match):
    with pytest.raises(ValueError, match=match):
        stratified_sample(rows, size=0, seed=1)


def test_exclude_ids_removes_validation_rows_and_validates_exclusions():
    rows = _rows(6)

    remaining = exclude_ids(rows, {"row-01", "row-04"})

    assert _ids(remaining) == ["row-00", "row-02", "row-03", "row-05"]
    with pytest.raises(ValueError, match="missing excluded id"):
        exclude_ids(rows, {"does-not-exist"})


def test_build_validation_splits_makes_eval_subset_and_excludes_validation():
    rows = _rows()
    original = deepcopy(rows)

    splits = build_validation_splits(
        rows,
        validation_size=8,
        eval_size=3,
        seed=17,
    )

    validation_ids = set(_ids(splits.val_rows))
    eval_ids = set(_ids(splits.eval_rows))
    train_ids = set(_ids(splits.train_candidates))
    assert len(validation_ids) == 8
    assert len(eval_ids) == 3
    assert eval_ids <= validation_ids
    assert validation_ids.isdisjoint(train_ids)
    assert validation_ids | train_ids == set(_ids(rows))
    assert rows == original


def test_build_validation_splits_is_stable_for_same_input_set():
    rows = _rows()

    first = build_validation_splits(rows, validation_size=8, eval_size=3, seed=17)
    second = build_validation_splits(
        [rows[5], *rows[:5], *rows[6:]],
        validation_size=8,
        eval_size=3,
        seed=17,
    )

    assert _ids(first.val_rows) == _ids(second.val_rows)
    assert _ids(first.eval_rows) == _ids(second.eval_rows)
    assert _ids(first.train_candidates) == _ids(second.train_candidates)


@pytest.mark.parametrize(
    "validation_size,eval_size",
    [
        (-1, 0),
        (13, 0),
        (4, -1),
        (4, 5),
    ],
)
def test_build_validation_splits_rejects_invalid_sizes(validation_size, eval_size):
    with pytest.raises(ValueError):
        build_validation_splits(
            _rows(),
            validation_size=validation_size,
            eval_size=eval_size,
            seed=1,
        )
