from __future__ import annotations

from types import SimpleNamespace

from post_train_v2.src.training import model_loading, trainer_args


class FakeAutoModelForCausalLM:
    calls = []

    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        cls.calls.append((model_path, kwargs))
        return SimpleNamespace(config=SimpleNamespace(use_cache=True))


class FakeAutoTokenizer:
    calls = []

    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        cls.calls.append((model_path, kwargs))
        return SimpleNamespace(
            pad_token_id=None,
            eos_token_id=2,
            eos_token="<eos>",
            pad_token=None,
        )


class FakeTrainingArguments:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_model_loading_enforces_bf16_flash_attention_without_device_map(monkeypatch):
    fake_transformers = SimpleNamespace(
        AutoModelForCausalLM=FakeAutoModelForCausalLM,
        AutoTokenizer=FakeAutoTokenizer,
    )
    fake_torch = SimpleNamespace(bfloat16="bf16")

    def fake_import(name):
        return {"transformers": fake_transformers, "torch": fake_torch}[name]

    FakeAutoModelForCausalLM.calls.clear()
    FakeAutoTokenizer.calls.clear()
    monkeypatch.setattr(model_loading, "import_module", fake_import)

    model, tokenizer = model_loading.load_causal_lm_and_tokenizer(
        "models/qwen3-0.6b",
        gradient_checkpointing=True,
    )

    assert model.config.use_cache is False
    assert tokenizer.pad_token == "<eos>"
    assert FakeAutoTokenizer.calls == [
        ("models/qwen3-0.6b", {"trust_remote_code": True})
    ]
    assert FakeAutoModelForCausalLM.calls == [
        (
            "models/qwen3-0.6b",
            {
                "trust_remote_code": True,
                "attn_implementation": "flash_attention_2",
                "torch_dtype": "bf16",
            },
        )
    ]
    assert "device_map" not in FakeAutoModelForCausalLM.calls[0][1]


def test_training_arguments_use_conservative_shared_defaults(monkeypatch):
    fake_transformers = SimpleNamespace(TrainingArguments=FakeTrainingArguments)
    monkeypatch.setattr(
        trainer_args,
        "import_module",
        lambda name: fake_transformers,
    )

    args = trainer_args.build_training_arguments(
        {
            "learning_rate": 1e-5,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 2,
            "output_dir": "outputs/sft/full",
        }
    )

    assert args.kwargs["learning_rate"] == 1e-5
    assert args.kwargs["lr_scheduler_type"] == "cosine"
    assert args.kwargs["warmup_ratio"] == 0.03
    assert args.kwargs["bf16"] is True
    assert args.kwargs["gradient_checkpointing"] is True
    assert args.kwargs["logging_strategy"] == "steps"
    assert args.kwargs["logging_steps"] == 1
    assert args.kwargs["logging_first_step"] is True
    assert args.kwargs["save_strategy"] == "steps"
    assert args.kwargs["save_steps"] == 100
    assert args.kwargs["save_total_limit"] == 2
    assert "max_steps" not in args.kwargs


def test_training_arguments_expose_optional_max_steps(monkeypatch):
    fake_transformers = SimpleNamespace(TrainingArguments=FakeTrainingArguments)
    monkeypatch.setattr(
        trainer_args,
        "import_module",
        lambda name: fake_transformers,
    )

    args = trainer_args.build_training_arguments(
        {
            "learning_rate": 1e-5,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 2,
            "output_dir": "outputs/sft/full",
        },
        max_steps=2,
    )

    assert args.kwargs["max_steps"] == 2
