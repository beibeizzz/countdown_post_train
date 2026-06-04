import os
import sys
import types

from post_train.src.countdown.wandb_utils import (
    build_wandb_init_kwargs,
    configure_wandb_env,
    finish_wandb,
    formatted_run_name,
    init_wandb_if_enabled,
    is_wandb_enabled,
    log_wandb_metrics,
    prefixed_metrics,
    trainer_report_to,
    wandb_run_name,
)


def test_wandb_disabled_by_default():
    cfg = {}

    assert is_wandb_enabled(cfg) is False
    assert trainer_report_to(cfg) == []
    assert wandb_run_name(cfg) is None


def test_wandb_enabled_by_report_to_string():
    cfg = {"report_to": "wandb", "run_name": "sft_full"}

    assert is_wandb_enabled(cfg) is True
    assert trainer_report_to(cfg) == ["wandb"]
    assert wandb_run_name(cfg) == "sft_full"


def test_wandb_enabled_by_report_to_list():
    cfg = {"report_to": ["tensorboard", "wandb"]}

    assert is_wandb_enabled(cfg) is True
    assert trainer_report_to(cfg) == ["tensorboard", "wandb"]


def test_build_wandb_init_kwargs_omits_empty_optional_values():
    cfg = {
        "wandb_project": "countdown-post-train",
        "wandb_entity": None,
        "wandb_group": "",
        "wandb_tags": ["sft", "full"],
        "run_name": "sft_full",
        "learning_rate": 1e-5,
    }

    assert build_wandb_init_kwargs(cfg, default_name="fallback") == {
        "project": "countdown-post-train",
        "name": "sft_full",
        "tags": ["sft", "full"],
        "config": cfg,
    }


def test_formatted_run_name_adds_timestamp_suffix_when_enabled(monkeypatch):
    monkeypatch.setattr(
        "post_train.src.countdown.wandb_utils.current_timestamp_suffix",
        lambda: "20260604_171234",
    )

    assert formatted_run_name({"run_name": "sft_full", "run_name_auto_suffix": True}, "fallback") == (
        "sft_full_20260604_171234"
    )


def test_formatted_run_name_keeps_plain_name_by_default():
    assert formatted_run_name({"run_name": "sft_full"}, "fallback") == "sft_full"
    assert formatted_run_name({}, "fallback") == "fallback"


def test_prefixed_metrics_filters_non_numeric_values():
    metrics = {
        "accuracy": 0.5,
        "format_rate": 1.0,
        "note": "skip",
        "entropy": None,
        "truncated_count": 2,
        "flag": True,
    }

    assert prefixed_metrics("eval", metrics) == {
        "eval/accuracy": 0.5,
        "eval/format_rate": 1.0,
        "eval/truncated_count": 2,
    }


def test_configure_wandb_env_sets_values_only_when_enabled(monkeypatch):
    for key in ("WANDB_PROJECT", "WANDB_ENTITY", "WANDB_RUN_GROUP", "WANDB_TAGS"):
        monkeypatch.delenv(key, raising=False)

    configure_wandb_env({"report_to": None, "wandb_project": "disabled"})
    assert "WANDB_PROJECT" not in os.environ

    configure_wandb_env(
        {
            "report_to": "wandb",
            "wandb_project": "countdown-post-train",
            "wandb_entity": "team",
            "wandb_group": "sft",
            "wandb_tags": ["qwen", "sft"],
        }
    )

    assert os.environ["WANDB_PROJECT"] == "countdown-post-train"
    assert os.environ["WANDB_ENTITY"] == "team"
    assert os.environ["WANDB_RUN_GROUP"] == "sft"
    assert os.environ["WANDB_TAGS"] == "qwen,sft"


def test_init_log_finish_wandb_run_when_enabled(monkeypatch):
    events = []

    class FakeRun:
        def log(self, metrics, step=None):
            events.append(("log", metrics, step))

        def finish(self):
            events.append(("finish",))

    def fake_init(**kwargs):
        events.append(("init", kwargs))
        return FakeRun()

    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(init=fake_init))

    run = init_wandb_if_enabled({"report_to": "wandb", "run_name": "grpo"}, default_name="fallback")
    log_wandb_metrics(run, {"train/loss": 0.5}, step=3)
    finish_wandb(run)

    assert events == [
        (
            "init",
            {
                "project": "countdown-post-train",
                "name": "grpo",
                "config": {"report_to": "wandb", "run_name": "grpo"},
            },
        ),
        ("log", {"train/loss": 0.5}, 3),
        ("finish",),
    ]
