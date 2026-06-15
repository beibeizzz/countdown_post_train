"""Deterministic Transformers generation capped at 256 new tokens."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from post_train_v2.src.evaluation.scoring import score_response


MAX_NEW_TOKENS = 256


def render_prompt(tokenizer, prompt: str) -> str:
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("prompt must be a nonempty string")
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def generation_kwargs(
    max_new_tokens: int,
    *,
    pad_token_id: int | None,
) -> dict[str, Any]:
    if (
        type(max_new_tokens) is not int
        or max_new_tokens <= 0
        or max_new_tokens > MAX_NEW_TOKENS
    ):
        raise ValueError("max_new_tokens must be an integer between 1 and 256")
    kwargs: dict[str, Any] = {
        "do_sample": False,
        "max_new_tokens": max_new_tokens,
    }
    if pad_token_id is not None:
        kwargs["pad_token_id"] = pad_token_id
    return kwargs


def is_truncated(
    generated_ids,
    *,
    max_new_tokens: int,
    eos_token_id: int | Sequence[int] | None,
) -> bool:
    tokens = [int(token) for token in generated_ids]
    if len(tokens) < max_new_tokens:
        return False
    if eos_token_id is None:
        return True
    eos_ids = (
        {eos_token_id}
        if type(eos_token_id) is int
        else {int(token) for token in eos_token_id}
    )
    return not any(token in eos_ids for token in tokens)


def generate_one(
    tokenizer,
    model,
    prompt: str,
    *,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> tuple[str, int, bool]:
    import torch

    rendered = render_prompt(tokenizer, prompt)
    inputs = tokenizer(rendered, return_tensors="pt")
    device = getattr(model, "device", None)
    if device is not None:
        if hasattr(inputs, "to"):
            inputs = inputs.to(device)
        else:
            inputs = {
                key: value.to(device) for key, value in inputs.items()
            }
    kwargs = generation_kwargs(
        max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **kwargs)
    input_tokens = int(inputs["input_ids"].shape[-1])
    generated_ids = output_ids[0][input_tokens:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    generated_tokens = int(generated_ids.shape[-1])
    truncated = is_truncated(
        generated_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
    )
    return text, generated_tokens, truncated


def evaluate_rows(
    rows: Sequence[Mapping[str, Any]],
    tokenizer,
    model,
    *,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for row in rows:
        text, token_count, truncated = generate_one(
            tokenizer,
            model,
            row["prompt"],
            max_new_tokens=max_new_tokens,
        )
        results.append(
            score_response(
                row,
                text,
                generated_tokens=token_count,
                truncated=truncated,
            )
        )
    return results

