from __future__ import annotations

from post_train_v2.verl.rewards.countdown_reward import compute_score


def test_verl_reward_adapter_returns_json_safe_diagnostics():
    result = compute_score(
        data_source="countdown",
        solution_str="<answer>1+1</answer>",
        ground_truth={"numbers": [1, 1], "target": 2},
        extra_info={"id": "x"},
        unused_verl_argument=True,
    )

    assert result["score"] == 1.2
    assert result["answer_correct"] is True
    assert result["format_ok"] is True
    assert result["expression"] == "1+1"
    assert result["value"] == "2/1"


def test_verl_reward_adapter_rejects_unknown_data_source():
    result = compute_score(
        data_source="other",
        solution_str="<answer>1+1</answer>",
        ground_truth={"numbers": [1, 1], "target": 2},
    )

    assert result["score"] == 0.0
    assert result["error"] == "unsupported_data_source"
