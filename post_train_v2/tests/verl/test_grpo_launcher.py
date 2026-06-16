from __future__ import annotations

from post_train_v2.verl.launch.train_grpo import build_verl_command


def test_grpo_launcher_builds_stock_verl_command_with_overrides():
    command = build_verl_command(
        config_path="post_train_v2/verl/configs/grpo_smoke.yaml",
        max_steps=1,
        resume_from_checkpoint="checkpoint",
        wandb_project="countdown",
        wandb_group="grpo",
        wandb_name="smoke",
        seed=123,
    )

    assert command[:3] == ["python", "-m", "verl.trainer.main_ppo"]
    joined = "\n".join(command)
    assert "actor_rollout_ref.model.path=post_train_v2/outputs/sft/full/best" in joined
    assert "trainer.n_gpus_per_node=2" in joined
    assert "custom_reward_function.path=post_train_v2/verl/rewards/countdown_reward.py" in joined
    assert "trainer.project_name=countdown" in joined
    assert "trainer.experiment_name=smoke" in joined
    assert "trainer.resume_from_path=checkpoint" in joined
    assert "trainer.total_training_steps=1" in joined
    assert "data.train_files=['post_train_v2/data/verl/train.parquet']" in joined
    assert "data.val_files=['post_train_v2/data/verl/validation.parquet']" in joined
