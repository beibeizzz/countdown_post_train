"""Rank-aware optional Weights & Biases utilities."""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from typing import Any

from post_train_v2.src.config.loading import REPO_ROOT


def make_run_name(base: str, now: datetime, git_revision: str) -> str:
    if not isinstance(base, str) or not base.strip():
        raise ValueError("run name base must be a nonempty string")
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("run name timestamp must be timezone-aware")
    if not isinstance(git_revision, str) or not git_revision.strip():
        raise ValueError("git revision must be a nonempty string")
    timestamp = now.astimezone(timezone.utc)
    return (
        f"{base.strip()}-{timestamp:%Y%m%d-%H%M%S}-"
        f"{git_revision.strip()[:7]}"
    )


def git_revision(repo_root: str | Path = REPO_ROOT) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    value = result.stdout.strip()
    return value or "unknown"


def init_run(
    config: Mapping[str, Any],
    *,
    rank: int,
    stage: str,
    now: datetime | None = None,
    revision: str | None = None,
):
    if not isinstance(config, Mapping):
        raise ValueError("tracking config must be a mapping")
    if type(rank) is not int or rank < 0:
        raise ValueError("rank must be a nonnegative exact integer")
    if not isinstance(stage, str) or not stage:
        raise ValueError("stage must be a nonempty string")
    enabled = config.get("enabled", False)
    if type(enabled) is not bool:
        raise ValueError("tracking enabled must be a boolean")
    if rank != 0 or not enabled:
        return None

    base_name = config.get("run_name") or stage
    run_name = make_run_name(
        base_name,
        now or datetime.now(timezone.utc),
        revision or git_revision(),
    )
    wandb = import_module("wandb")
    return wandb.init(
        project=config.get("project", "countdown-post-train-v2"),
        entity=config.get("entity"),
        group=config.get("group"),
        name=run_name,
        mode=config.get("mode", "online"),
        tags=config.get("tags"),
        config=dict(config),
    )


def log_metrics(
    run,
    metrics: Mapping[str, Any],
    *,
    step: int,
) -> None:
    if run is None:
        return
    if not isinstance(metrics, Mapping):
        raise ValueError("metrics must be a mapping")
    if type(step) is not int or step < 0:
        raise ValueError("step must be a nonnegative exact integer")
    run.log(dict(metrics), step=step)


def finish_run(run) -> None:
    if run is not None:
        run.finish()

