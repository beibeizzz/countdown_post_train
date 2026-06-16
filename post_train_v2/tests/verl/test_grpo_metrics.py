from __future__ import annotations

import pytest

from post_train_v2.src.evaluation.grpo_metrics import aggregate_grpo_metrics


def test_aggregate_grpo_metrics_preserves_zero_std_groups():
    metrics = aggregate_grpo_metrics(
        rewards=[1.2, 1.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
        group_size=4,
        response_lengths=[8, 9, 10, 11, 12, 13, 14, 256],
        truncated=[False, False, False, False, False, False, False, True],
    )

    assert metrics["reward_std"] == pytest.approx(0.5092887197)
    assert metrics["group_reward_std"] == pytest.approx(0.2772634127)
    assert metrics["frac_reward_zero_std"] == pytest.approx(0.5)
    assert metrics["all_correct_group_fraction"] == pytest.approx(0.0)
    assert metrics["all_wrong_group_fraction"] == pytest.approx(0.5)
    assert metrics["kl"] == 0.0
    assert metrics["truncated_count"] == 1
    assert metrics["avg_response_length"] == pytest.approx(41.625)


def test_aggregate_grpo_metrics_includes_bucket_metrics_and_omits_entropy_by_default():
    metrics = aggregate_grpo_metrics(
        rewards=[1.2, 0.0],
        group_size=1,
        response_lengths=[8, 9],
        truncated=[False, False],
        diagnostics=[
            {"answer_correct": True, "format_ok": True, "bucket": "easy"},
            {"answer_correct": False, "format_ok": False, "bucket": "hard"},
        ],
    )

    assert "entropy" not in metrics
    assert metrics["accuracy"] == 0.5
    assert metrics["format_rate"] == 0.5
    assert metrics["bucket/easy/accuracy"] == 1.0
    assert metrics["bucket/hard/reward"] == 0.0
