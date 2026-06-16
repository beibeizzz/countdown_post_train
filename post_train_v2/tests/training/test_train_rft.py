from __future__ import annotations

from post_train_v2.scripts.sft import train_rft


def test_train_rft_uses_base_model_rft_data_and_shared_runner(monkeypatch):
    calls = []
    config = {
        "model_path": "post_train/model/qwen/qwen3-0.6b",
        "train_data": "post_train_v2/data/rft/rft_accepted.jsonl",
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 2,
    }

    monkeypatch.setattr(train_rft, "load_yaml", lambda path: config)
    monkeypatch.setattr(
        train_rft,
        "run_supervised_training",
        lambda config_path, **kwargs: calls.append((config_path, kwargs)),
    )

    train_rft.main(
        [
            "--config",
            "post_train_v2/configs/sft/rft_train_smoke.yaml",
            "--max-steps",
            "2",
        ]
    )

    assert calls == [
        (
            "post_train_v2/configs/sft/rft_train_smoke.yaml",
            {
                "max_steps": 2,
                "resume_from_checkpoint": None,
            },
        )
    ]
