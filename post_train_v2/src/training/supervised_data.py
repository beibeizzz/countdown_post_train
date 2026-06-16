"""Response-only supervised data encoding for Countdown SFT/RFT."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any

from post_train_v2.src.countdown.prompts import build_chat_messages

IGNORE_INDEX = -100


@dataclass(frozen=True)
class EncodedSupervisedExample:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    prompt_length: int
    supervised_text: str


def _tokenize(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    input_ids = encoded["input_ids"]
    if not isinstance(input_ids, list) or not all(
        isinstance(token_id, int) for token_id in input_ids
    ):
        raise ValueError("tokenizer must return a list of integer input_ids")
    return input_ids


def _render_prompt(tokenizer: Any, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        build_chat_messages(prompt),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _render_full(tokenizer: Any, prompt: str, response: str) -> str:
    return tokenizer.apply_chat_template(
        build_chat_messages(prompt) + [{"role": "assistant", "content": response}],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )


def _pad_id(tokenizer: Any) -> int:
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        return int(pad_token_id)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        raise ValueError("tokenizer must define pad_token_id or eos_token_id")
    return int(eos_token_id)


def encode_prompt_response(
    *,
    tokenizer: Any,
    prompt: str,
    response: str,
    max_seq_len: int,
) -> EncodedSupervisedExample:
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("prompt must be a nonempty string")
    if not isinstance(response, str) or not response:
        raise ValueError("response must be a nonempty string")
    if type(max_seq_len) is not int or max_seq_len <= 0:
        raise ValueError("max_seq_len must be a positive integer")

    prompt_text = _render_prompt(tokenizer, prompt)
    supervised_text = _render_full(tokenizer, prompt, response)
    prompt_ids = _tokenize(tokenizer, prompt_text)
    full_ids = _tokenize(tokenizer, supervised_text)
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("prompt chat template must be a prefix of full template")

    truncated_ids = full_ids[:max_seq_len]
    prompt_length = min(len(prompt_ids), len(truncated_ids))
    labels = [IGNORE_INDEX] * prompt_length + truncated_ids[prompt_length:]
    if not any(label != IGNORE_INDEX for label in labels):
        raise ValueError("truncation left no supervised response tokens")

    attention_mask = [1] * len(truncated_ids)
    pad_length = max_seq_len - len(truncated_ids)
    if pad_length > 0:
        truncated_ids = truncated_ids + [_pad_id(tokenizer)] * pad_length
        attention_mask = attention_mask + [0] * pad_length
        labels = labels + [IGNORE_INDEX] * pad_length

    return EncodedSupervisedExample(
        input_ids=truncated_ids,
        attention_mask=attention_mask,
        labels=labels,
        prompt_length=prompt_length,
        supervised_text=supervised_text,
    )


class SupervisedDataCollator:
    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        torch = import_module("torch")
        return {
            key: torch.tensor([feature[key] for feature in features])
            for key in ("input_ids", "attention_mask", "labels")
        }
