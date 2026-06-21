"""GRPO configuration loading and validation for stock verl."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from post_train_v2.src.config.loading import load_yaml


def load_grpo_config(path: str | Path) -> dict[str, Any]:
    config = load_yaml(path)
    validate_grpo_config(config)
    return config


def validate_grpo_config(config: dict[str, Any]) -> None:
    actor = config["actor_rollout_ref"]["actor"]
    rollout = config["actor_rollout_ref"]["rollout"]
    data = config["data"]
    trainer = config["trainer"]
    algorithm = config["algorithm"]
    if algorithm.get("adv_estimator") != "grpo":
        raise ValueError("adv_estimator must be grpo")
    if algorithm.get("use_kl_in_reward") is not False:
        raise ValueError("KL reward must be disabled")
    if actor.get("use_kl_loss") is not False:
        raise ValueError("KL loss must be disabled")
    if actor["ppo_mini_batch_size"] > data["train_batch_size"]:
        raise ValueError("mini batch cannot exceed train batch")
    if actor.get("ppo_micro_batch_size_per_gpu") is None:
        raise ValueError("per-GPU micro batch is required")
    if rollout.get("tensor_model_parallel_size") != 1:
        raise ValueError("TP must be 1 for two 40GB GPUs")
    if trainer.get("max_actor_ckpt_to_keep") is not None:
        raise ValueError("checkpoint pruning must be disabled during training")
    if data.get("validation_files") != ["post_train_v2/data/verl/validation.parquet"]:
        raise ValueError("validation data must be the fixed 50 Parquet")
    if "critic" in config:
        raise ValueError("critic is not used for GRPO")
    if "reward_model" in config:
        raise ValueError("learned reward model is not used")
    if data.get("apply_chat_template_kwargs", {}).get("enable_thinking") is not False:
        raise ValueError("thinking must be disabled")
    model = config["actor_rollout_ref"]["model"]
    if model.get("override_config", {}).get("attn_implementation") != "flash_attention_2":
        raise ValueError("actor must force Flash Attention 2")


def with_overrides(config: dict[str, Any], overrides: dict[tuple[str, ...], Any]) -> dict[str, Any]:
    result = deepcopy(config)
    for path, value in overrides.items():
        target = result
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
    validate_grpo_config(result)
    return result
