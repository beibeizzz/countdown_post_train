from __future__ import annotations

from post_train_v2.src.rewards.countdown import RewardResult, score_response


def test_countdown_reward_scores_format_and_correctness():
    assert score_response("<answer>1+1</answer>", [1, 1], 2).score == 1.2
    assert score_response("<answer>1+2</answer>", [1, 1], 2).score == 0.2
    assert score_response("1+1", [1, 1], 2).score == 0.0


def test_countdown_reward_diagnostics_are_exact():
    result = score_response("<answer>47+47</answer>", [47, 47], 94)

    assert isinstance(result, RewardResult)
    assert result.format_ok is True
    assert result.answer_correct is True
    assert result.error is None
    assert result.expression == "47+47"
    assert result.value == "94/1"


def test_countdown_reward_reports_semantic_error():
    result = score_response("<answer>1+2</answer>", [1, 1], 2)

    assert result.format_ok is True
    assert result.answer_correct is False
    assert result.error == "number_mismatch"
    assert result.expression == "1+2"
    assert result.value == "3/1"
