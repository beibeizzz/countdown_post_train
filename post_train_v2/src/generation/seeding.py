"""Stable request-level seed derivation for generation stages."""

from __future__ import annotations

import hashlib


def derive_request_seed(
    global_seed: int,
    stage: str,
    sample_id: str,
    rollout_index: int,
) -> int:
    if type(global_seed) is not int or global_seed < 0:
        raise ValueError("global_seed must be a nonnegative exact integer")
    if not isinstance(stage, str) or not stage:
        raise ValueError("stage must be a nonempty string")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("sample_id must be a nonempty string")
    if type(rollout_index) is not int or rollout_index < 0:
        raise ValueError("rollout_index must be a nonnegative exact integer")

    payload = (
        f"{global_seed}|{stage}|{sample_id}|{rollout_index}".encode("utf-8")
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")
