from __future__ import annotations

import pytest

from post_train_v2.src.training.supervised_data import (
    EncodedSupervisedExample,
    encode_prompt_response,
)


class FakeQwenTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __init__(self):
        self.template_calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append((messages, kwargs))
        rendered = ""
        for message in messages:
            rendered += f"<|{message['role']}|>{message['content']}"
        if kwargs.get("add_generation_prompt"):
            rendered += "<|assistant|>"
        return rendered

    def __call__(self, text, *, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": [ord(character) for character in text]}


def test_encode_prompt_response_masks_prompt_and_supervises_response():
    tokenizer = FakeQwenTokenizer()

    encoded = encode_prompt_response(
        tokenizer=tokenizer,
        prompt="question",
        response="reasoning\n<answer>1+1</answer>",
        max_seq_len=256,
    )

    assert isinstance(encoded, EncodedSupervisedExample)
    assert encoded.labels[: encoded.prompt_length] == [-100] * encoded.prompt_length
    assert any(label != -100 for label in encoded.labels)
    assert encoded.supervised_text.endswith("<answer>1+1</answer>")
    response_labels = [
        label for label in encoded.labels[encoded.prompt_length :] if label != -100
    ]
    assert response_labels == [
        ord(character) for character in "reasoning\n<answer>1+1</answer>"
    ]


def test_encode_prompt_response_disables_thinking_for_both_templates():
    tokenizer = FakeQwenTokenizer()

    encode_prompt_response(
        tokenizer=tokenizer,
        prompt="question",
        response="<answer>1+1</answer>",
        max_seq_len=128,
    )

    assert [call[1]["enable_thinking"] for call in tokenizer.template_calls] == [
        False,
        False,
    ]
    assert tokenizer.template_calls[0][1]["add_generation_prompt"] is True
    assert tokenizer.template_calls[1][1]["add_generation_prompt"] is False


def test_encode_prompt_response_right_pads_inputs_masks_and_labels():
    tokenizer = FakeQwenTokenizer()

    encoded = encode_prompt_response(
        tokenizer=tokenizer,
        prompt="q",
        response="<answer>2</answer>",
        max_seq_len=64,
    )

    assert len(encoded.input_ids) == 64
    assert len(encoded.attention_mask) == 64
    assert len(encoded.labels) == 64
    first_padding = encoded.attention_mask.index(0)
    assert encoded.input_ids[first_padding:] == [tokenizer.pad_token_id] * (
        64 - first_padding
    )
    assert encoded.labels[first_padding:] == [-100] * (64 - first_padding)


def test_encode_prompt_response_rejects_when_truncation_removes_response():
    tokenizer = FakeQwenTokenizer()
    prompt_only = tokenizer.apply_chat_template(
        [{"role": "user", "content": "question"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    with pytest.raises(ValueError, match="no supervised response tokens"):
        encode_prompt_response(
            tokenizer=tokenizer,
            prompt="question",
            response="<answer>1+1</answer>",
            max_seq_len=len(prompt_only),
        )
