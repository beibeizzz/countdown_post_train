"""Stock verl GRPO launcher."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from post_train_v2.src.artifacts.atomic import publish_json
from post_train_v2.src.artifacts.hashing import sha256_canonical_json
from post_train_v2.verl.launch.configuration import load_grpo_config


def build_verl_command(
    *,
    config_path: str,
    max_steps: int | None = None,
    resume_from_checkpoint: str | None = None,
    wandb_project: str | None = None,
    wandb_group: str | None = None,
    wandb_name: str | None = None,
    seed: int | None = None,
) -> list[str]:
    config = load_grpo_config(config_path)
    command = ["python", "-m", "verl.trainer.main_ppo"]
    command.extend(
        [
            f"actor_rollout_ref.model.path={config['actor_rollout_ref']['model']['path']}",
            f"trainer.n_gpus_per_node={config['trainer']['n_gpus_per_node']}",
            f"data.train_files={config['data']['train_files']}",
            f"data.val_files={config['data']['validation_files']}",
            "custom_reward_function.path=post_train_v2/verl/rewards/countdown_reward.py",
        ]
    )
    if max_steps is not None:
        command.append(f"trainer.total_training_steps={max_steps}")
    if resume_from_checkpoint:
        command.append(f"trainer.resume_from_path={resume_from_checkpoint}")
    if wandb_project:
        command.append(f"trainer.project_name={wandb_project}")
    if wandb_group:
        command.append(f"trainer.wandb_group={wandb_group}")
    if wandb_name:
        command.append(f"trainer.experiment_name={wandb_name}")
    if seed is not None:
        command.append(f"trainer.seed={seed}")
    return command


def run_grpo(
    *,
    config_path: str,
    max_steps: int | None = None,
    resume_from_checkpoint: str | None = None,
) -> int:
    command = build_verl_command(
        config_path=config_path,
        max_steps=max_steps,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    config_hash = sha256_canonical_json({"command": command})
    output_dir = Path("post_train_v2/outputs/grpo/launch")
    publish_json(
        output_dir / "last_command.json",
        {"command": command, "config_sha256": config_hash},
    )
    executable_command = [sys.executable if item == "python" else item for item in command]
    return subprocess.run(executable_command, check=False).returncode
