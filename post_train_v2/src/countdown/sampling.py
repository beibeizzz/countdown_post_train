"""Deterministic, balanced sampling for Countdown datasets."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Any


Row = dict[str, Any]


@dataclass(frozen=True)
class ValidationSplits:
    val_rows: list[Row]
    eval_rows: list[Row]
    train_candidates: list[Row]


def stratified_sample(
    rows: Sequence[Row],
    size: int,
    seed: int,
) -> list[Row]:
    normalized = _normalize_rows(rows)
    _validate_size(size, len(normalized), "size")
    _validate_seed(seed)
    if size == 0:
        return []
    if size == len(normalized):
        return normalized

    buckets: dict[str, list[Row]] = defaultdict(list)
    for row in normalized:
        buckets[row["bucket"]["bucket_key"]].append(row)

    bucket_keys = sorted(buckets)
    random_source = random.Random(seed)
    shuffled_buckets: dict[str, list[Row]] = {}
    for bucket_key in bucket_keys:
        bucket_rows = list(buckets[bucket_key])
        random_source.shuffle(bucket_rows)
        shuffled_buckets[bucket_key] = bucket_rows

    quotas = _balanced_quotas(bucket_keys, shuffled_buckets, size)
    selected: list[Row] = []
    leftovers: list[Row] = []
    for bucket_key in bucket_keys:
        quota = quotas[bucket_key]
        selected.extend(shuffled_buckets[bucket_key][:quota])
        leftovers.extend(shuffled_buckets[bucket_key][quota:])

    shortfall = size - len(selected)
    if shortfall:
        selected.extend(leftovers[:shortfall])
    random_source.shuffle(selected)
    return selected


def exclude_ids(
    rows: Sequence[Row],
    excluded_ids: Collection[str],
) -> list[Row]:
    normalized = _normalize_rows(rows)
    excluded = set(excluded_ids)
    if len(excluded) != len(excluded_ids):
        raise ValueError("duplicate excluded id")
    if any(not isinstance(row_id, str) or not row_id for row_id in excluded):
        raise ValueError("excluded ids must be non-empty strings")

    available = {row["id"] for row in normalized}
    missing = sorted(excluded - available)
    if missing:
        raise ValueError(f"missing excluded id: {missing[0]}")
    return [row for row in normalized if row["id"] not in excluded]


def build_validation_splits(
    rows: Sequence[Row],
    validation_size: int = 200,
    eval_size: int = 50,
    seed: int = 0,
) -> ValidationSplits:
    normalized = _normalize_rows(rows)
    _validate_size(validation_size, len(normalized), "validation_size")
    _validate_size(eval_size, validation_size, "eval_size")
    _validate_seed(seed)

    val_rows = stratified_sample(normalized, validation_size, seed)
    eval_rows = stratified_sample(val_rows, eval_size, seed + 1)
    train_candidates = exclude_ids(
        normalized,
        {row["id"] for row in val_rows},
    )
    return ValidationSplits(
        val_rows=val_rows,
        eval_rows=eval_rows,
        train_candidates=train_candidates,
    )


def _normalize_rows(rows: Sequence[Row]) -> list[Row]:
    normalized: list[Row] = []
    seen_ids: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("each row must be a mapping")

        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id:
            raise ValueError("row is missing id")
        if row_id in seen_ids:
            raise ValueError(f"duplicate row id: {row_id}")
        seen_ids.add(row_id)

        bucket = row.get("bucket")
        if not isinstance(bucket, dict):
            raise ValueError("row is missing bucket.bucket_key")
        bucket_key = bucket.get("bucket_key")
        if not isinstance(bucket_key, str) or not bucket_key:
            raise ValueError("row is missing bucket.bucket_key")
        normalized.append(row)

    return sorted(normalized, key=lambda row: row["id"])


def _validate_size(size: int, maximum: int, name: str) -> None:
    if isinstance(size, bool) or not isinstance(size, int):
        raise ValueError(f"{name} must be an integer")
    if size < 0 or size > maximum:
        raise ValueError(f"{name} must be between 0 and {maximum}")


def _validate_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")


def _balanced_quotas(
    bucket_keys: list[str],
    buckets: dict[str, list[Row]],
    size: int,
) -> dict[str, int]:
    quotas = {bucket_key: 0 for bucket_key in bucket_keys}
    bucket_order = {bucket_key: index for index, bucket_key in enumerate(bucket_keys)}

    for _ in range(size):
        eligible = [
            bucket_key
            for bucket_key in bucket_keys
            if quotas[bucket_key] < len(buckets[bucket_key])
        ]
        bucket_key = min(
            eligible,
            key=lambda key: (quotas[key], bucket_order[key]),
        )
        quotas[bucket_key] += 1
    return quotas
