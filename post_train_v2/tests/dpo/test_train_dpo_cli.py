from __future__ import annotations

from post_train_v2.scripts.dpo import train_dpo


def test_train_dpo_cli_forwards_common_arguments(monkeypatch):
    calls = []
    monkeypatch.setattr(
        train_dpo,
        "run_dpo_training",
        lambda config, **kwargs: calls.append((config, kwargs)),
    )

    train_dpo.main(
        [
            "--config",
            "post_train_v2/configs/dpo/train_smoke.yaml",
            "--max-steps",
            "2",
            "--resume-from-checkpoint",
            "checkpoint-100",
        ]
    )

    assert calls == [
        (
            "post_train_v2/configs/dpo/train_smoke.yaml",
            {
                "max_steps": 2,
                "resume_from_checkpoint": "checkpoint-100",
            },
        )
    ]
