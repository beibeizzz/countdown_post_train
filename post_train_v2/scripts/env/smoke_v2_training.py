from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml


STAGE_ORDER = (
    "build_source",
    "validation_split",
    "teacher_pool",
    "accepted_splits",
    "full_sft",
    "lora_sft",
    "rft_data",
    "rft_train",
    "dpo_data",
    "dpo_train",
    "grpo_convert",
    "grpo_train",
    "grpo_export",
)


@dataclass(frozen=True)
class SmokeCommand:
    stage: str
    command: list[str]
    env: dict[str, str]


def build_smoke_commands(*, through_stage: str, work_dir: str | Path) -> list[SmokeCommand]:
    if through_stage not in STAGE_ORDER:
        raise ValueError(f"unknown smoke stage: {through_stage}")
    work_dir = Path(work_dir)
    configs = work_dir / "configs"
    outputs = work_dir / "outputs"
    data = work_dir / "data"
    cuda_env = {"CUDA_VISIBLE_DEVICES": "0,1"}
    commands = [
        SmokeCommand(
            "build_source",
            [
                "python",
                "post_train_v2/scripts/data/build_source.py",
                "--config",
                str(configs / "build_source.yaml"),
                "--limit",
                "32",
            ],
            {},
        ),
        SmokeCommand(
            "validation_split",
            [
                "python",
                "post_train_v2/scripts/data/build_splits.py",
                "--config",
                str(configs / "build_splits.yaml"),
                "validation",
            ],
            {},
        ),
        SmokeCommand(
            "teacher_pool",
            [
                "python",
                "post_train_v2/scripts/generation/build_teacher_pool.py",
                "--config",
                str(configs / "teacher_rollout.yaml"),
            ],
            cuda_env,
        ),
        SmokeCommand(
            "accepted_splits",
            [
                "python",
                "post_train_v2/scripts/data/build_splits.py",
                "--config",
                str(configs / "build_splits.yaml"),
                "accepted",
            ],
            {},
        ),
        _torchrun_stage(
            "full_sft",
            "post_train_v2/scripts/sft/train_full.py",
            configs / "full_sft.yaml",
        ),
        _torchrun_stage(
            "lora_sft",
            "post_train_v2/scripts/sft/train_lora.py",
            configs / "lora_sft.yaml",
        ),
        SmokeCommand(
            "rft_data",
            [
                "python",
                "post_train_v2/scripts/sft/build_rft_data.py",
                "--config",
                str(configs / "rft_rollout.yaml"),
            ],
            cuda_env,
        ),
        _torchrun_stage(
            "rft_train",
            "post_train_v2/scripts/sft/train_rft.py",
            configs / "rft_train.yaml",
        ),
        SmokeCommand(
            "dpo_data",
            [
                "python",
                "post_train_v2/scripts/dpo/build_dpo_data.py",
                "--config",
                str(configs / "dpo_build.yaml"),
            ],
            cuda_env,
        ),
        _torchrun_stage(
            "dpo_train",
            "post_train_v2/scripts/dpo/train_dpo.py",
            configs / "dpo_train.yaml",
        ),
        SmokeCommand(
            "grpo_convert",
            [
                "python",
                "post_train_v2/scripts/grpo/convert_to_parquet.py",
                "--train-jsonl",
                str(data / "processed" / "grpo_train_4k.jsonl"),
                "--val-jsonl",
                str(data / "processed" / "eval_50.jsonl"),
                "--output-dir",
                str(data / "verl"),
            ],
            {},
        ),
        SmokeCommand(
            "grpo_train",
            [
                "python",
                "post_train_v2/scripts/grpo/train_grpo.py",
                "--config",
                str(configs / "grpo_smoke.yaml"),
                "--max-steps",
                "1",
            ],
            cuda_env,
        ),
        SmokeCommand(
            "grpo_export",
            [
                "python",
                "post_train_v2/scripts/grpo/export_grpo.py",
                "--run-dir",
                str(outputs / "grpo" / "smoke"),
            ],
            {},
        ),
    ]
    stop = STAGE_ORDER.index(through_stage) + 1
    return commands[:stop]


def prepare_smoke_workspace(work_dir: str | Path) -> None:
    work_dir = Path(work_dir)
    (work_dir / "configs").mkdir(parents=True, exist_ok=True)
    (work_dir / "data").mkdir(parents=True, exist_ok=True)
    (work_dir / "outputs").mkdir(parents=True, exist_ok=True)
    _write_placeholder_configs(work_dir)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    prepare_smoke_workspace(args.work_dir)
    for item in build_smoke_commands(
        through_stage=args.through_stage,
        work_dir=args.work_dir,
    ):
        env = os.environ.copy()
        env.update(item.env)
        result = subprocess.run(item.command, env=env, check=False)
        if result.returncode != 0:
            return result.returncode
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run isolated V2 GPU smoke stages.")
    parser.add_argument("--through-stage", default="grpo_export", choices=STAGE_ORDER)
    parser.add_argument("--work-dir", required=True)
    return parser


def _torchrun_stage(stage: str, script: str, config: Path) -> SmokeCommand:
    return SmokeCommand(
        stage,
        ["torchrun", "--standalone", "--nproc_per_node=2", script, "--config", str(config)],
        {"CUDA_VISIBLE_DEVICES": "0,1"},
    )


def _write_placeholder_configs(work_dir: Path) -> None:
    configs = work_dir / "configs"
    data = work_dir / "data"
    outputs = work_dir / "outputs"
    _write_yaml(
        configs / "build_source.yaml",
        {
            "raw_train": "post_train/datasets/raw_train.parquet",
            "raw_test": "post_train/datasets/raw_test.json",
            "output_dir": str(data / "processed"),
        },
    )
    _write_yaml(
        configs / "build_splits.yaml",
        {
            "input_dir": str(data / "processed"),
            "output_dir": str(data / "processed"),
            "validation_size": 8,
            "eval_size": 4,
            "sft_size": 8,
            "grpo_size": 4,
            "seed": 42,
        },
    )
    _write_yaml(
        configs / "teacher_rollout.yaml",
        {
            "input_path": str(data / "processed" / "train_candidates.jsonl"),
            "output_dir": str(data / "teacher"),
            "model_path": "post_train/model/qwen/qwen3-8b",
            "stop_after_accepted": 8,
            "batch_size": 4,
            "devices": [0, 1],
        },
    )
    for name, output in {
        "full_sft": outputs / "sft" / "full",
        "lora_sft": outputs / "sft" / "lora",
        "rft_train": outputs / "sft" / "rft",
        "dpo_train": outputs / "dpo",
    }.items():
        _write_yaml(
            configs / f"{name}.yaml",
            {
                "model_path": "post_train/model/qwen/qwen3-0.6b",
                "train_data": str(data / "processed" / "sft_train_8k.jsonl"),
                "eval_data": str(data / "processed" / "eval_50.jsonl"),
                "output_dir": str(output),
                "learning_rate": 1.0e-5,
                "num_train_epochs": 1,
                "max_seq_len": 256,
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "eval_every_steps": 1,
                "max_new_tokens": 256,
                "report_to": [],
                "export_kind": "full_model",
            },
        )
    _write_yaml(
        configs / "rft_rollout.yaml",
        {
            "input_path": str(data / "processed" / "sft_train_8k.jsonl"),
            "output_dir": str(data / "processed"),
            "model_path": str(outputs / "sft" / "full" / "best"),
            "rollouts_per_prompt": 2,
        },
    )
    _write_yaml(
        configs / "dpo_build.yaml",
        {
            "input_path": str(data / "processed" / "sft_train_8k.jsonl"),
            "output_dir": str(data / "processed"),
            "model_path": "post_train/model/qwen/qwen3-8b",
        },
    )
    _write_yaml(
        configs / "grpo_smoke.yaml",
        {
            "algorithm": {"adv_estimator": "grpo", "use_kl_in_reward": False},
            "actor_rollout_ref": {
                "model": {
                    "path": str(outputs / "sft" / "full" / "best"),
                    "override_config": {"attn_implementation": "flash_attention_2"},
                },
                "actor": {"strategy": "fsdp2", "use_kl_loss": False},
                "rollout": {"tensor_model_parallel_size": 1, "n": 4},
            },
            "data": {
                "train_files": [str(data / "verl" / "train.parquet")],
                "validation_files": [str(data / "verl" / "validation.parquet")],
                "train_batch_size": 4,
                "max_prompt_length": 256,
                "max_response_length": 256,
                "apply_chat_template_kwargs": {"enable_thinking": False},
            },
            "trainer": {
                "nnodes": 1,
                "n_gpus_per_node": 2,
                "save_freq": 100,
                "test_freq": 100,
                "log_val_generations": 50,
                "default_local_dir": str(outputs / "grpo" / "smoke" / "checkpoints"),
            },
        },
    )


def _write_yaml(path: Path, value: dict) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
