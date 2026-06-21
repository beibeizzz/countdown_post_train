"""Canonical prompts for V2 Countdown generation."""


def build_solution_prompt(numbers: list[int], target: int) -> str:
    return (
        f"Using the numbers {numbers}, create an equation that equals {target}.\n"
        "Use each number exactly once. Only use +, -, *, / and parentheses.\n"
        "Do not use any other numbers. Keep the response concise.\n"
        "Finally return <answer> equation </answer>."
    )


def build_dpo_forced_wrong_prompt(
    numbers: list[int],
    target: int,
    chosen_response: str,
) -> str:
    return (
        f"Using the numbers {numbers}, create a plausible but mathematically wrong "
        f"answer for {target}.\n"
        "Use the same numbers exactly once. Only use +, -, *, / and parentheses.\n"
        f"The expression must be parseable and evaluate to a value different from {target}.\n"
        "Return a complete <answer> expression </answer> block.\n"
        "Do not mention that the response is intentionally wrong.\n"
        "Imitate the style and brevity of this response:\n"
        f"{chosen_response}"
    )


def build_chat_messages(prompt: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": prompt}]
