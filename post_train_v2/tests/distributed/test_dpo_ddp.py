from __future__ import annotations

from types import SimpleNamespace

from post_train_v2.src.training import dpo


def test_dpo_tracking_and_eval_are_rank_zero_owned(monkeypatch, tmp_path):
    events = []
    monkeypatch.setattr(dpo, "current_context", lambda: SimpleNamespace(rank=1, world_size=2))
    monkeypatch.setattr(
        dpo,
        "load_yaml",
        lambda path: {
            "model_path": "post_train_v2/outputs/sft/full/best",
            "train_data": "post_train_v2/data/dpo/dpo_pairs.jsonl",
            "eval_data": "post_train_v2/data/processed/eval_50.jsonl",
            "output_dir": str(tmp_path),
            "learning_rate": 5e-7,
            "num_train_epochs": 1,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 2,
            "max_length": 256,
            "beta": 0.05,
            "eval_every_steps": 100,
            "max_new_tokens": 256,
            "export_kind": "full_model",
            "tracking": {"enabled": True},
        },
    )
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
        lambda *args, **kwargs: ("model", "tokenizer"),
    )
    monkeypatch.setattr(dpo, "FixedEvaluationCallback", lambda **kwargs: events.append(("eval", kwargs)) or "eval")
    monkeypatch.setattr(dpo, "init_run", lambda config, rank, stage: events.append(("wandb", rank)) or None)
    monkeypatch.setattr(dpo, "finish_run", lambda run: events.append(("finish",)))
    monkeypatch.setattr(dpo, "export_dpo_outputs", lambda **kwargs: None)

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def train(self, *, resume_from_checkpoint=None):
            events.append(("train", resume_from_checkpoint))

    fake_modules = {
        "trl": SimpleNamespace(
            DPOConfig=lambda **kwargs: SimpleNamespace(kwargs=kwargs),
            DPOTrainer=FakeTrainer,
        ),
        "datasets": SimpleNamespace(
            Dataset=SimpleNamespace(from_list=lambda rows: rows)
        ),
    }
    monkeypatch.setattr(dpo, "import_module", lambda name: fake_modules[name])

    dpo.run_dpo_training("config.yaml", max_steps=2)

    assert ("wandb", 1) in events
    assert [event[0] for event in events].count("eval") == 1
