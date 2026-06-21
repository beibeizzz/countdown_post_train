"""Shared Transformers TrainingArguments construction."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from typing import Any


def build_training_arguments(
    config: Mapping[str, Any],
    *,
    max_steps: int | None = None,
):
    transformers = import_module("transformers")
    kwargs: dict[str, Any] = {
        "output_dir": _require(config, "output_dir"),
        "learning_rate": _require(config, "learning_rate"),
        "num_train_epochs": _require(config, "num_train_epochs"),
        "per_device_train_batch_size": _require(
            config,
            "per_device_train_batch_size",
        ),
        "gradient_accumulation_steps": _require(
            config,
            "gradient_accumulation_steps",
        ),
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "bf16": True,
        "gradient_checkpointing": True,
        "logging_strategy": "steps",
        "logging_steps": 1,
        "logging_first_step": True,
        "save_strategy": "steps",
        "save_steps": 100,
        "save_total_limit": 2,
        "remove_unused_columns": False,
        "report_to": config.get("report_to", []),
    }
    if max_steps is not None:
        if type(max_steps) is not int or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer when supplied")
        kwargs["max_steps"] = max_steps
    return transformers.TrainingArguments(**kwargs)


def _require(config: Mapping[str, Any], key: str) -> Any:
    if key not in config:
        raise ValueError(f"training config missing key: {key}")
    return config[key]
