from __future__ import annotations

import os
from datetime import datetime
from typing import Any


def trainer_report_to(cfg: dict[str, Any]) -> list[str]:
    value = cfg.get("report_to")
    if value is None or value is False or value == "":
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    raise TypeError("report_to must be null, a string, or a list of strings")


def is_wandb_enabled(cfg: dict[str, Any]) -> bool:
    return "wandb" in trainer_report_to(cfg)


def wandb_run_name(cfg: dict[str, Any]) -> str | None:
    value = cfg.get("run_name")
    if value is None:
        return None
    run_name = str(value).strip()
    return run_name or None


def current_timestamp_suffix() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def formatted_run_name(cfg: dict[str, Any], default_name: str) -> str:
    base_name = wandb_run_name(cfg) or default_name
    if bool(cfg.get("run_name_auto_suffix", False)):
        return f"{base_name}_{current_timestamp_suffix()}"
    return base_name


def build_wandb_init_kwargs(cfg: dict[str, Any], default_name: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "project": str(cfg.get("wandb_project") or "countdown-post-train"),
        "name": formatted_run_name(cfg, default_name),
        "config": cfg,
    }
    entity = cfg.get("wandb_entity")
    if entity:
        kwargs["entity"] = str(entity)
    group = cfg.get("wandb_group")
    if group:
        kwargs["group"] = str(group)
    tags = cfg.get("wandb_tags") or []
    if tags:
        kwargs["tags"] = [str(tag) for tag in tags]
    return kwargs


def configure_wandb_env(cfg: dict[str, Any]) -> None:
    if not is_wandb_enabled(cfg):
        return
    project = cfg.get("wandb_project")
    if project:
        os.environ["WANDB_PROJECT"] = str(project)
    entity = cfg.get("wandb_entity")
    if entity:
        os.environ["WANDB_ENTITY"] = str(entity)
    group = cfg.get("wandb_group")
    if group:
        os.environ["WANDB_RUN_GROUP"] = str(group)
    tags = cfg.get("wandb_tags") or []
    if tags:
        os.environ["WANDB_TAGS"] = ",".join(str(tag) for tag in tags)


def init_wandb_if_enabled(cfg: dict[str, Any], default_name: str):
    if not is_wandb_enabled(cfg):
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("wandb logging is enabled, but the 'wandb' package is not installed") from exc
    return wandb.init(**build_wandb_init_kwargs(cfg, default_name=default_name))


def log_wandb_metrics(run, metrics: dict[str, Any], step: int | None = None) -> None:
    if run is None or not metrics:
        return
    if step is None:
        run.log(metrics)
    else:
        run.log(metrics, step=step)


def finish_wandb(run) -> None:
    if run is not None:
        run.finish()


def prefixed_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, float | int]:
    output: dict[str, float | int] = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            output[f"{prefix}/{key}"] = value
    return output
