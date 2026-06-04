import pytest

from post_train.scripts.dpo.train_dpo import format_dpo_record_for_trl, prepare_dpo_records


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=False):
        assert tokenize is False
        roles = ",".join(message["role"] for message in messages)
        contents = "|".join(message["content"] for message in messages)
        suffix = ":gen" if add_generation_prompt else ""
        thinking = ":think" if enable_thinking else ""
        return f"{roles}:{contents}{suffix}{thinking}"


class LegacyFakeTokenizer(FakeTokenizer):
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return super().apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )


def test_prepare_dpo_records_keeps_exact_required_keys_and_text():
    rows = [
        {
            "prompt": "make 10",
            "chosen": "full chosen answer",
            "rejected": "full rejected answer",
            "extra": "ignored",
        }
    ]

    assert prepare_dpo_records(rows) == [
        {
            "prompt": "make 10",
            "chosen": "full chosen answer",
            "rejected": "full rejected answer",
        }
    ]


def test_prepare_dpo_records_requires_string_fields():
    with pytest.raises(ValueError, match="row 1.*chosen"):
        prepare_dpo_records([{"prompt": "p", "chosen": None, "rejected": "r"}])


def test_format_dpo_record_for_trl_uses_chat_template_without_losing_responses():
    record = {"prompt": "p", "chosen": "c", "rejected": "r"}

    formatted = format_dpo_record_for_trl(record, FakeTokenizer(), enable_thinking=True)

    assert formatted == {
        "prompt": "user:p:gen:think",
        "chosen": "c",
        "rejected": "r",
    }


def test_format_dpo_record_for_trl_supports_legacy_chat_templates():
    record = {"prompt": "p", "chosen": "c", "rejected": "r"}

    formatted = format_dpo_record_for_trl(record, LegacyFakeTokenizer(), enable_thinking=True)

    assert formatted == {
        "prompt": "user:p:gen",
        "chosen": "c",
        "rejected": "r",
    }
