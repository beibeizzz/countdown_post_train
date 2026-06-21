"""GRPO rollout metric aggregation helpers."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from math import sqrt
from typing import Any


def aggregate_grpo_metrics(
    *,
    rewards: Sequence[float],
    group_size: int,
    response_lengths: Sequence[int],
    truncated: Sequence[bool],
    diagnostics: Sequence[Mapping[str, Any]] | None = None,
    entropy: float | None = None,
) -> dict[str, float | int]:
    _validate_common_inputs(rewards, group_size, response_lengths, truncated, diagnostics)

    count = len(rewards)
    if count == 0:
        metrics: dict[str, float | int] = {
            "count": 0,
            "reward": 0.0,
            "reward_std": 0.0,
            "group_reward_std": 0.0,
            "frac_reward_zero_std": 0.0,
            "all_correct_group_fraction": 0.0,
            "all_wrong_group_fraction": 0.0,
            "accuracy": 0.0,
            "format_rate": 0.0,
            "avg_response_length": 0.0,
            "max_response_length": 0,
            "truncated_count": 0,
            "truncated_rate": 0.0,
            "kl": 0.0,
        }
        if entropy is not None:
            metrics["entropy"] = float(entropy)
        return metrics

    reward_values = [float(reward) for reward in rewards]
    length_values = [int(length) for length in response_lengths]
    group_rewards = _chunks(reward_values, group_size)
    group_stds = [_population_std(group) for group in group_rewards]

    metrics = {
        "count": count,
        "reward": sum(reward_values) / count,
        "reward_std": _population_std(reward_values),
        "group_reward_std": sum(group_stds) / len(group_stds),
        "frac_reward_zero_std": sum(std == 0.0 for std in group_stds) / len(group_stds),
        "all_correct_group_fraction": (
            sum(all(reward >= 1.2 for reward in group) for group in group_rewards)
            / len(group_rewards)
        ),
        "all_wrong_group_fraction": (
            sum(all(reward <= 0.0 for reward in group) for group in group_rewards)
            / len(group_rewards)
        ),
        "avg_response_length": sum(length_values) / count,
        "max_response_length": max(length_values),
        "truncated_count": sum(flag is True for flag in truncated),
        "truncated_rate": sum(flag is True for flag in truncated) / count,
        "kl": 0.0,
    }

    if diagnostics is not None:
        metrics.update(_diagnostic_metrics(reward_values, diagnostics))

    if "accuracy" not in metrics:
        metrics["accuracy"] = sum(reward >= 1.2 for reward in reward_values) / count
    if "format_rate" not in metrics:
        metrics["format_rate"] = sum(reward >= 0.2 for reward in reward_values) / count
    if entropy is not None:
        metrics["entropy"] = float(entropy)
    return metrics


def _diagnostic_metrics(
    rewards: Sequence[float],
    diagnostics: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    count = len(diagnostics)
    metrics: dict[str, float] = {
        "accuracy": sum(row.get("answer_correct") is True for row in diagnostics) / count,
        "format_rate": sum(row.get("format_ok") is True for row in diagnostics) / count,
    }

    bucket_rows: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(diagnostics):
        bucket = row.get("bucket")
        if isinstance(bucket, str) and bucket:
            bucket_rows[bucket].append(index)

    for bucket, indexes in sorted(bucket_rows.items()):
        bucket_count = len(indexes)
        prefix = f"bucket/{bucket}"
        metrics[f"{prefix}/accuracy"] = (
            sum(diagnostics[index].get("answer_correct") is True for index in indexes)
            / bucket_count
        )
        metrics[f"{prefix}/format_rate"] = (
            sum(diagnostics[index].get("format_ok") is True for index in indexes)
            / bucket_count
        )
        metrics[f"{prefix}/reward"] = (
            sum(rewards[index] for index in indexes) / bucket_count
        )
    return metrics


def _population_std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _chunks(values: Sequence[float], size: int) -> list[list[float]]:
    return [list(values[index : index + size]) for index in range(0, len(values), size)]


def _validate_common_inputs(
    rewards: Sequence[float],
    group_size: int,
    response_lengths: Sequence[int],
    truncated: Sequence[bool],
    diagnostics: Sequence[Mapping[str, Any]] | None,
) -> None:
    if type(group_size) is not int or group_size <= 0:
        raise ValueError("group_size must be a positive exact integer")
    if len(rewards) != len(response_lengths) or len(rewards) != len(truncated):
        raise ValueError("rewards, response_lengths, and truncated must have equal length")
    if len(rewards) % group_size != 0:
        raise ValueError("rewards length must be divisible by group_size")
    if diagnostics is not None and len(diagnostics) != len(rewards):
        raise ValueError("diagnostics length must match rewards length")

    for index, reward in enumerate(rewards):
        if type(reward) not in {int, float}:
            raise ValueError(f"rewards[{index}] must be numeric")
    for index, length in enumerate(response_lengths):
        if type(length) is not int or length < 0:
            raise ValueError(
                f"response_lengths[{index}] must be a nonnegative exact integer"
            )
    for index, flag in enumerate(truncated):
        if type(flag) is not bool:
            raise ValueError(f"truncated[{index}] must be a boolean")
