import pytest

from post_train.scripts.sft.train_full import (
    DataCollatorForCausalSFT,
    build_eval_wandb_metrics,
    build_training_arguments,
    encode_prompt_response,
    normalize_sft_config,
)


class FakeTokenizer:
    def __init__(self, prompt_ids=None, full_ids=None):
        self.prompt_ids = prompt_ids or [11, 12, 13]
        self.full_ids = full_ids or [11, 12, 13, 21, 22]
        self.rendered_prompt = "PROMPT"
        self.rendered_full = "FULL"
        self.enable_thinking_calls = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=False):
        self.enable_thinking_calls.append(enable_thinking)
        if len(messages) == 1:
            return self.rendered_prompt
        return self.rendered_full

    def __call__(self, text, add_special_tokens=False):
        if text == self.rendered_prompt:
            return {"input_ids": list(self.prompt_ids)}
        if text == self.rendered_full:
            return {"input_ids": list(self.full_ids)}
        raise AssertionError(f"unexpected text: {text}")


class LegacyFakeTokenizer(FakeTokenizer):
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        if len(messages) == 1:
            return self.rendered_prompt
        return self.rendered_full


def test_encode_masks_prompt_labels_and_keeps_response_trainable():
    encoded = encode_prompt_response(FakeTokenizer(), "prompt", "response", max_seq_len=10)

    assert encoded == {
        "input_ids": [11, 12, 13, 21, 22],
        "labels": [-100, -100, -100, 21, 22],
        "attention_mask": [1, 1, 1, 1, 1],
    }


@pytest.mark.parametrize("max_seq_len", [2, 3])
def test_encode_returns_none_when_truncated_to_prompt_only_region(max_seq_len):
    encoded = encode_prompt_response(FakeTokenizer(), "prompt", "response", max_seq_len=max_seq_len)

    assert encoded is None


def test_encode_truncating_after_partial_response_keeps_only_response_labels():
    encoded = encode_prompt_response(FakeTokenizer(), "prompt", "response", max_seq_len=4)

    assert encoded == {
        "input_ids": [11, 12, 13, 21],
        "labels": [-100, -100, -100, 21],
        "attention_mask": [1, 1, 1, 1],
    }


def test_encode_raises_when_prompt_template_is_not_full_template_prefix():
    tokenizer = FakeTokenizer(prompt_ids=[11, 99], full_ids=[11, 12, 21])

    with pytest.raises(ValueError, match="Prompt tokens must be a prefix"):
        encode_prompt_response(tokenizer, "prompt", "response", max_seq_len=10)


def test_encode_passes_enable_thinking_when_tokenizer_supports_it():
    tokenizer = FakeTokenizer()

    encode_prompt_response(tokenizer, "prompt", "response", max_seq_len=10, enable_thinking=True)

    assert tokenizer.enable_thinking_calls == [True, True]


def test_encode_omits_enable_thinking_for_legacy_tokenizers():
    encoded = encode_prompt_response(LegacyFakeTokenizer(), "prompt", "response", max_seq_len=10, enable_thinking=True)

    assert encoded["labels"] == [-100, -100, -100, 21, 22]


def test_data_collator_pads_values_and_returns_long_tensors():
    torch = pytest.importorskip("torch")
    collator = DataCollatorForCausalSFT(pad_token_id=0)

    batch = collator(
        [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [-100, 2, 3]},
            {"input_ids": [4], "attention_mask": [1], "labels": [4]},
        ]
    )

    assert batch["input_ids"].tolist() == [[1, 2, 3], [4, 0, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    assert batch["labels"].tolist() == [[-100, 2, 3], [4, -100, -100]]
    assert batch["input_ids"].shape == torch.Size([2, 3])
    assert batch["attention_mask"].shape == torch.Size([2, 3])
    assert batch["labels"].shape == torch.Size([2, 3])
    assert batch["input_ids"].dtype == torch.long
    assert batch["attention_mask"].dtype == torch.long
    assert batch["labels"].dtype == torch.long


def test_normalize_sft_config_maps_rft_train_section_to_full_sft_config():
    cfg = normalize_sft_config(
        {
            "accepted_output": "post_train/data/sft/rft_accepted.jsonl",
            "output_dir": "post_train/outputs/sft/rft",
            "enable_thinking": False,
            "train": {
                "max_seq_len": 256,
                "learning_rate": 1e-5,
                "warmup_ratio": 0.03,
                "scheduler": "cosine",
                "epochs": 2,
                "per_device_train_batch_size": 4,
                "gradient_accumulation_steps": 4,
                "bf16": True,
                "gradient_checkpointing": True,
            },
        }
    )

    assert cfg["model_path"] == "post_train/model/qwen/qwen3-0.6b"
    assert cfg["train_data"] == "post_train/data/sft/rft_accepted.jsonl"
    assert cfg["output_dir"] == "post_train/outputs/sft/rft"
    assert cfg["weight_decay"] == 0.0
    assert cfg["eval_every_steps"] == 100
    assert cfg["save_every_steps"] == 100
    assert cfg["epochs"] == 2


def test_normalize_sft_config_preserves_rft_wandb_train_fields():
    cfg = normalize_sft_config(
        {
            "accepted_output": "post_train/data/sft/rft_accepted.jsonl",
            "output_dir": "post_train/outputs/sft/rft",
            "train": {
                "max_seq_len": 256,
                "learning_rate": 1e-5,
                "warmup_ratio": 0.03,
                "scheduler": "cosine",
                "epochs": 2,
                "per_device_train_batch_size": 4,
                "gradient_accumulation_steps": 4,
                "bf16": True,
                "gradient_checkpointing": True,
                "report_to": "wandb",
                "wandb_project": "countdown-post-train",
                "run_name": "rft",
                "logging_steps": 5,
            },
        }
    )

    assert cfg["report_to"] == "wandb"
    assert cfg["wandb_project"] == "countdown-post-train"
    assert cfg["run_name"] == "rft"
    assert cfg["logging_steps"] == 5


def test_build_training_arguments_uses_wandb_config(monkeypatch, tmp_path):
    import sys

    captured = {}

    class FakeTrainingArguments:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        type("FakeTransformers", (), {"TrainingArguments": FakeTrainingArguments}),
    )
    monkeypatch.setattr("post_train.src.countdown.wandb_utils.current_timestamp_suffix", lambda: "20260604_171234")

    build_training_arguments(
        {
            "epochs": 1,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 1e-5,
            "weight_decay": 0.0,
            "warmup_ratio": 0.03,
            "scheduler": "cosine",
            "bf16": False,
            "gradient_checkpointing": False,
            "save_every_steps": 100,
            "report_to": "wandb",
            "run_name": "sft_full",
            "run_name_auto_suffix": True,
            "logging_steps": 7,
        },
        tmp_path,
        max_steps=2,
    )

    assert captured["report_to"] == ["wandb"]
    assert captured["run_name"] == "sft_full_20260604_171234"
    assert captured["logging_steps"] == 7


def test_build_eval_wandb_metrics_prefixes_numeric_values():
    assert build_eval_wandb_metrics({"accuracy": 0.4, "note": "skip", "truncated_count": 1}) == {
        "eval/accuracy": 0.4,
        "eval/truncated_count": 1,
    }
