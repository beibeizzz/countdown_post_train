from __future__ import annotations

import pytest

from post_train_v2.verl.launch.configuration import (
    load_grpo_config,
    validate_grpo_config,
)


def test_grpo_config_contains_required_stock_verl_settings():
    config = load_grpo_config("post_train_v2/verl/configs/grpo.yaml")

    assert config["algorithm"]["adv_estimator"] == "grpo"
    assert config["algorithm"]["use_kl_in_reward"] is False
    model = config["actor_rollout_ref"]["model"]
    assert model["path"] == "post_train_v2/outputs/sft/full/best"
    assert model["enable_gradient_checkpointing"] is True
    assert model["override_config"]["attn_implementation"] == "flash_attention_2"
    actor = config["actor_rollout_ref"]["actor"]
    assert actor["strategy"] == "fsdp2"
    assert actor["use_kl_loss"] is False
    assert actor["ppo_mini_batch_size"] == 4
    assert actor["ppo_micro_batch_size_per_gpu"] == 2
    assert actor["ppo_epochs"] == 2
    rollout = config["actor_rollout_ref"]["rollout"]
    assert rollout["tensor_model_parallel_size"] == 1
    assert rollout["n"] == 4
    assert rollout["temperature"] == 1.0
    assert rollout["top_p"] == 0.95
    data = config["data"]
    assert data["train_batch_size"] == 4
    assert data["max_prompt_length"] == 256
    assert data["max_response_length"] == 256
    assert data["apply_chat_template_kwargs"]["enable_thinking"] is False
    trainer = config["trainer"]
    assert trainer["nnodes"] == 1
    assert trainer["n_gpus_per_node"] == 2
    assert trainer["save_freq"] == 100
    assert trainer["test_freq"] == 100
    assert trainer["log_val_generations"] == 50
    assert trainer["max_actor_ckpt_to_keep"] is None
    assert "critic" not in config
    assert "reward_model" not in config


@pytest.mark.parametrize(
    ("path", "value", "match"),
    (
        (("actor_rollout_ref", "actor", "ppo_mini_batch_size"), 8, "mini batch"),
        (("actor_rollout_ref", "actor", "use_kl_loss"), True, "KL"),
        (("actor_rollout_ref", "rollout", "tensor_model_parallel_size"), 2, "TP"),
        (("trainer", "max_actor_ckpt_to_keep"), 2, "checkpoint pruning"),
        (("data", "validation_files"), ["other.parquet"], "fixed 50"),
    ),
)
def test_grpo_config_validation_rejects_unsafe_values(path, value, match):
    config = load_grpo_config("post_train_v2/verl/configs/grpo.yaml")
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ValueError, match=match):
        validate_grpo_config(config)
