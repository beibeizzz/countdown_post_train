from post_train.src.countdown.prompts import build_dpo_forced_wrong_prompt, build_solution_prompt


def test_solution_prompt_contains_required_rules_without_exact_division():
    prompt = build_solution_prompt([1, 1, 1, 1], 4)

    assert "Using the numbers [1, 1, 1, 1]" in prompt
    assert "Use each number exactly once" in prompt
    assert "<answer>" in prompt
    assert "Division must be exact" not in prompt


def test_forced_wrong_prompt_requests_wrong_answer():
    prompt = build_dpo_forced_wrong_prompt([7, 3, 8, 2], 24, "short <answer> (7-3)*(8-2) </answer>")

    assert "wrong" in prompt.lower()
    assert "Use the same numbers exactly once" in prompt
    assert "complete <answer> expression </answer>" in prompt
    assert "different from" in prompt
    assert "Do not mention" in prompt
    assert "intentionally wrong" in prompt
    assert "if possible" not in prompt
    assert "when possible" not in prompt
