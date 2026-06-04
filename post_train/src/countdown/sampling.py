from __future__ import annotations

import random
from collections import defaultdict
from typing import Any


def _balanced_quotas(
    bucket_keys: list[str],
    buckets: dict[str, list[dict[str, Any]]],
    size: int,
) -> dict[str, int]:
    quotas = {bucket_key: 0 for bucket_key in bucket_keys}

    for _ in range(size):
        eligible = [
            bucket_key
            for bucket_key in bucket_keys
            if quotas[bucket_key] < len(buckets[bucket_key])
        ]
        if not eligible:
            break

        bucket_key = min(
            eligible,
            key=lambda key: (quotas[key], bucket_keys.index(key)),
        )
        quotas[bucket_key] += 1

    return quotas


def stratified_sample(rows: list[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    if size <= 0:
        return []

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_ids: set[Any] = set()
    for row in rows:
        if row.get("id") in (None, ""):
            raise ValueError("row is missing id")
        row_id = row["id"]
        if row_id in seen_ids:
            raise ValueError(f"duplicate row id: {row_id}")
        seen_ids.add(row_id)

        try:
            bucket_key = row["bucket"]["bucket_key"]
        except (KeyError, TypeError) as exc:
            raise ValueError("row is missing bucket.bucket_key") from exc
        if bucket_key in (None, ""):
            raise ValueError("row is missing bucket.bucket_key")
        buckets[bucket_key].append(row)

    if size >= len(rows):
        return list(rows)

    bucket_keys = sorted(buckets)
    rng = random.Random(seed)
    shuffled_buckets = {
        bucket_key: rng.sample(buckets[bucket_key], k=len(buckets[bucket_key]))
        for bucket_key in bucket_keys
    }

    quotas = _balanced_quotas(bucket_keys, shuffled_buckets, size)
    selected: list[dict[str, Any]] = []
    leftovers: list[dict[str, Any]] = []

    for bucket_key in bucket_keys:
        quota = quotas[bucket_key]
        bucket_rows = shuffled_buckets[bucket_key]
        selected.extend(bucket_rows[:quota])
        leftovers.extend(bucket_rows[quota:])

    shortfall = size - len(selected)
    if shortfall > 0:
        selected.extend(leftovers[:shortfall])

    rng.shuffle(selected)
    return selected[:size]
