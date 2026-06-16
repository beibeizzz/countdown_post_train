from __future__ import annotations

from types import SimpleNamespace

from post_train_v2.src.training import dpo


class FakeDPOConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeDPOTrainer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.trained_with = None

    def train(self, *, resume_from_checkpoint=None):
        self.trained_with = resume_from_checkpoint

    def save_model(self, path):
        self.saved_path = path


def test_dpo_trainer_construction_uses_full_sft_model_and_implicit_ref(monkeypatch):
    events = []
    config = {
        "model_path": "post_train_v2/outputs/sft/full/best",
        "train_data": "post_train_v2/data/dpo/dpo_pairs.jsonl",
        "eval_data": "post_train_v2/data/processed/eval_50.jsonl",
        "output_dir": "post_train_v2/outputs/dpo",
        "learning_rate": 5e-7,
        "num_train_epochs": 1,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 2,
        "max_length": 256,
        "beta": 0.05,
        "eval_every_steps": 100,
        "max_new_tokens": 256,
        "export_kind": "full_model",
    }

    monkeypatch.setattr(dpo, "load_yaml", lambda path: config)
    monkeypatch.setattr(
        dpo,
        "read_jsonl_strict",
        lambda path, validator: [
            {
                "prompt": "prompt",
                "chosen": "chosen",
                "rejected": "rejected",
                "rejected_category": "wrong_value",
                "generation_route": "forced_wrong",
                "provenance": {},
            }
        ],
    )
    monkeypatch.setattr(
        dpo,
        "load_causal_lm_and_tokenizer",
        lambda model_path, gradient_checkpointing: events.append(
            ("load", str(model_path), gradient_checkpointing)
        )
        or ("model", "tokenizer"),
    )
    monkeypatch.setattr(dpo, "FixedEvaluationCallback", lambda **kwargs: "eval")
    monkeypatch.setattr(dpo, "export_dpo_outputs", lambda **kwargs: events.append(("export", kwargs)))
    fake_modules = {
        "trl": SimpleNamespace(DPOConfig=FakeDPOConfig, DPOTrainer=FakeDPOTrainer),
        "datasets": SimpleNamespace(
            Dataset=SimpleNamespace(from_list=lambda rows: ("dataset", rows))
        ),
    }
    monkeypatch.setattr(dpo, "import_module", lambda name: fake_modules[name])

    summary = dpo.run_dpo_training(
        "post_train_v2/configs/dpo/train.yaml",
        max_steps=2,
        resume_from_checkpoint="checkpoint-100",
    )

    trainer = summary["trainer"]
    assert events[0] == ("load", str(dpo.resolve_repo_path(config["model_path"])), True)
    assert trainer.trained_with == "checkpoint-100"
    assert trainer.kwargs["model"] == "model"
    assert "ref_model" not in trainer.kwargs
    assert trainer.kwargs["processing_class"] == "tokenizer"
    assert trainer.kwargs["train_dataset"][1][0] == {
        "prompt": "prompt",
        "chosen": "chosen",
        "rejected": "rejected",
    }
    args = trainer.kwargs["args"].kwargs
    assert args["beta"] == 0.05
    assert args["max_length"] == 256
    assert args["learning_rate"] == 5e-7
    assert args["num_train_epochs"] == 1
    assert args["per_device_train_batch_size"] == 4
    assert args["gradient_accumulation_steps"] == 2
    assert args["logging_steps"] == 1
    assert args["logging_first_step"] is True
    assert summary["global_batch_size"] == 8
