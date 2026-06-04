from post_train.scripts.grpo.train_grpo import (
    build_metric_row,
    compute_rewards,
    ensure_policy_examples_available,
    grpo_metric_summary,
    validate_supported_grpo_config,
)


def test_compute_rewards_adds_format_and_answer_rewards_without_mutating_inputs():
    rows = [
        {"numbers": [7, 3, 8, 2], "target": 24, "prompt": "make 24"},
        {"numbers": [1, 2, 3], "target": 7, "prompt": "make 7"},
        {"numbers": [1, 2, 3], "target": 7, "prompt": "make 7"},
    ]
    completions = [
        "Work it out.\n<answer>(7-3)*(8-2)</answer>",
        "No tagged answer here",
        "<answer>1+2+3</answer>",
    ]

    rewarded = compute_rewards(rows, completions, format_reward=0.2, answer_reward=1.0)

    assert [row["reward"] for row in rewarded] == [1.2, 0.0, 0.2]
    assert [row["format_ok"] for row in rewarded] == [True, False, True]
    assert [row["correct"] for row in rewarded] == [True, False, False]
    assert "reward" not in rows[0]
    assert rewarded[0]["completion"] == completions[0]


def test_compute_rewards_requires_matching_row_and_completion_counts():
    rows = [{"numbers": [1], "target": 1}]

    try:
        compute_rewards(rows, [], format_reward=0.2, answer_reward=1.0)
    except ValueError as exc:
        assert "same length" in str(exc)
    else:
        raise AssertionError("Expected mismatched rows/completions to fail")


def test_grpo_metric_summary_uses_population_std_and_group_chunks():
    summary = grpo_metric_summary([0.0, 1.0, 1.0, 1.0, 0.5], group_size=2)

    assert summary == {
        "reward_std": 0.4,
        "group_reward_std": 0.25,
        "frac_reward_zero_std": 0.5,
    }


def test_grpo_metric_summary_handles_empty_rewards():
    assert grpo_metric_summary([], group_size=4) == {
        "reward_std": 0.0,
        "group_reward_std": 0.0,
        "frac_reward_zero_std": 0.0,
    }


def test_build_metric_row_includes_required_logging_fields():
    rewarded_rows = [
        {"reward": 1.2, "format_ok": True, "correct": True, "token_count": 12, "truncated": False},
        {"reward": 0.2, "format_ok": True, "correct": False, "token_count": 20, "truncated": True},
    ]

    metric_row = build_metric_row(
        loss=0.75,
        rewarded_rows=rewarded_rows,
        group_size=2,
        approx_kl=0.0,
        entropy=None,
        learning_rate=3e-7,
    )

    assert metric_row == {
        "loss": 0.75,
        "mean_reward": 0.7,
        "reward_std": 0.5,
        "group_reward_std": 0.5,
        "frac_reward_zero_std": 0.0,
        "accuracy": 0.5,
        "format_rate": 1.0,
        "approx_kl": 0.0,
        "entropy": None,
        "avg_gen_tokens": 16.0,
        "max_gen_tokens": 20,
        "truncated_count": 1,
        "rollout_count": 2,
        "learning_rate": 3e-7,
    }


def test_validate_supported_grpo_config_accepts_zero_kl_coeff():
    validate_supported_grpo_config({"kl_coeff": 0.0})
    validate_supported_grpo_config({})


def test_validate_supported_grpo_config_rejects_nonzero_kl_coeff():
    try:
        validate_supported_grpo_config({"kl_coeff": 0.01})
    except ValueError as exc:
        message = str(exc)
        assert "reference KL is not implemented" in message
        assert "kl_coeff: 0.0" in message
    else:
        raise AssertionError("Expected nonzero kl_coeff to fail")


def test_validate_supported_grpo_config_rejects_nonpositive_policy_updates():
    for value in (0, -1):
        try:
            validate_supported_grpo_config({"policy_updates_per_rollout": value})
        except ValueError as exc:
            assert "policy_updates_per_rollout" in str(exc)
            assert "at least 1" in str(exc)
        else:
            raise AssertionError(f"Expected policy_updates_per_rollout={value} to fail")


def test_validate_supported_grpo_config_rejects_nonpositive_loop_sizes():
    for key in ("batch_size", "group_size", "max_steps"):
        try:
            validate_supported_grpo_config({key: 0})
        except ValueError as exc:
            assert key in str(exc)
            assert "at least 1" in str(exc)
        else:
            raise AssertionError(f"Expected {key}=0 to fail")


def test_ensure_policy_examples_available_fails_fast_on_empty_rollout_encoding():
    try:
        ensure_policy_examples_available([], rollout_count=4)
    except RuntimeError as exc:
        message = str(exc)
        assert "No policy examples could be encoded from the rollout" in message
        assert "4 completions" in message
    else:
        raise AssertionError("Expected empty policy examples to fail")


def test_ensure_policy_examples_available_returns_nonempty_examples():
    examples = [({"input_ids": [1], "labels": [1], "attention_mask": [1]}, 0.25)]

    assert ensure_policy_examples_available(examples, rollout_count=1) is examples
