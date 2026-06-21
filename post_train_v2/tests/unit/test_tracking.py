from __future__ import annotations

import sys
from datetime import datetime, timezone
from types import SimpleNamespace

from post_train_v2.src.tracking.wandb import (
    finish_run,
    init_run,
    log_metrics,
    make_run_name,
)


def config(enabled: bool = True) -> dict:
    return {
        "enabled": enabled,
        "project": "countdown-v2",
        "entity": None,
        "group": "sft",
        "run_name": "full-sft",
        "mode": "online",
        "tags": ["phase1"],
    }


def test_make_run_name_has_timestamp_and_short_revision():
    assert make_run_name(
        "full-sft",
        datetime(2026, 6, 16, 1, 2, 3, tzinfo=timezone.utc),
        "abcdef123456",
    ) == "full-sft-20260616-010203-abcdef1"


def test_disabled_and_nonzero_rank_do_not_import_wandb(monkeypatch):
    monkeypatch.delitem(sys.modules, "wandb", raising=False)
    imported = []
    monkeypatch.setattr(
        "post_train_v2.src.tracking.wandb.import_module",
        lambda name: imported.append(name),
    )

    assert init_run(config(False), rank=0, stage="sft") is None
    assert init_run(config(True), rank=1, stage="sft") is None
    assert imported == []


def test_rank_zero_initializes_with_automatic_suffix(monkeypatch):
    calls = []
    run = object()
    fake_wandb = SimpleNamespace(
        init=lambda **kwargs: calls.append(kwargs) or run
    )
    monkeypatch.setattr(
        "post_train_v2.src.tracking.wandb.import_module",
        lambda name: fake_wandb,
    )

    result = init_run(
        config(),
        rank=0,
        stage="sft",
        now=datetime(2026, 6, 16, 1, 2, 3, tzinfo=timezone.utc),
        revision="1234567890",
    )

    assert result is run
    assert calls == [
        {
            "project": "countdown-v2",
            "entity": None,
            "group": "sft",
            "name": "full-sft-20260616-010203-1234567",
            "mode": "online",
            "tags": ["phase1"],
            "config": config(),
        }
    ]


def test_log_metrics_records_explicit_every_step_and_finish():
    events = []
    run = SimpleNamespace(
        log=lambda metrics, step: events.append(("log", metrics, step)),
        finish=lambda: events.append(("finish",)),
    )

    log_metrics(run, {"loss": 1.0, "reward": 0.2}, step=1)
    log_metrics(run, {"loss": 0.8, "reward": 0.4}, step=2)
    log_metrics(None, {"loss": 0.0}, step=3)
    finish_run(run)
    finish_run(None)

    assert events == [
        ("log", {"loss": 1.0, "reward": 0.2}, 1),
        ("log", {"loss": 0.8, "reward": 0.4}, 2),
        ("finish",),
    ]

