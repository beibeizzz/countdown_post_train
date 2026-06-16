from __future__ import annotations

from types import SimpleNamespace

from post_train_v2.scripts.sft.train_full import build_parser
from post_train_v2.src.training import supervised


def test_train_full_cli_accepts_config_max_steps_and_resume():
    args = build_parser().parse_args(
        [
            "--config",
            "post_train_v2/configs/sft/full_smoke.yaml",
            "--max-steps",
            "2",
            "--resume-from-checkpoint",
            "checkpoint-100",
        ]
    )

    assert args.config == "post_train_v2/configs/sft/full_smoke.yaml"
    assert args.max_steps == 2
    assert args.resume_from_checkpoint == "checkpoint-100"


def test_full_sft_runner_trains_with_resume_and_expected_global_batch(monkeypatch):
    events = []
    config = {
        "model_path": "post_train/model/qwen/qwen3-0.6b",
        "train_data": "train.jsonl",
        "eval_data": "eval.jsonl",
        "output_dir": "post_train_v2/outputs/sft/full",
        "learning_rate": 1e-5,
        "num_train_epochs": 3,
        "max_seq_len": 256,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 2,
        "eval_every_steps": 100,
        "max_new_tokens": 256,
    }

    class FakeTrainer:
        def __init__(self, **kwargs):
            events.append(("trainer", kwargs))

        def train(self, *, resume_from_checkpoint=None):
            events.append(("train", resume_from_checkpoint))

    monkeypatch.setattr(supervised, "load_yaml", lambda path: config)
    monkeypatch.setattr(
        supervised,
        "read_jsonl_strict",
        lambda path, validator: [
            {"prompt": "q", "response": "<answer>1+1</answer>"}
        ],
    )
    monkeypatch.setattr(
        supervised,
        "load_causal_lm_and_tokenizer",
        lambda model_path, gradient_checkpointing: ("model", "tokenizer"),
    )
    monkeypatch.setattr(
        supervised,
        "encode_prompt_response",
        lambda **kwargs: SimpleNamespace(
            input_ids=[1],
            attention_mask=[1],
            labels=[1],
        ),
    )
    monkeypatch.setattr(supervised, "build_training_arguments", lambda cfg, max_steps: "args")
    monkeypatch.setattr(supervised, "FixedEvaluationCallback", lambda **kwargs: "eval-callback")
    monkeypatch.setattr(supervised, "import_module", lambda name: SimpleNamespace(Trainer=FakeTrainer))
    monkeypatch.setenv("WORLD_SIZE", "2")

    summary = supervised.run_supervised_training(
        "config.yaml",
        max_steps=2,
        resume_from_checkpoint="checkpoint-100",
    )

    assert summary["global_batch_size"] == 16
    assert events[-1] == ("train", "checkpoint-100")
    trainer_kwargs = events[0][1]
    assert trainer_kwargs["model"] == "model"
    assert trainer_kwargs["args"] == "args"
    assert trainer_kwargs["train_dataset"][0] == {
        "input_ids": [1],
        "attention_mask": [1],
        "labels": [1],
    }
    assert trainer_kwargs["callbacks"] == ["eval-callback"]
