from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _find_repo_root(script_path: Path) -> Path:
    for parent in script_path.resolve().parents:
        if (parent / "post_train_v2").is_dir() and (parent / ".git").exists():
            return parent
    raise RuntimeError(f"could not locate repository root from {script_path}")


REPO_ROOT = _find_repo_root(Path(__file__))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from post_train_v2.src.config.loading import load_yaml
from post_train_v2.src.training.supervised import run_supervised_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train V2 RFT with DDP.")
    parser.add_argument(
        "--config",
        default="post_train_v2/configs/sft/rft_train.yaml",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--resume-from-checkpoint", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _validate_rft_config(load_yaml(args.config))
    run_supervised_training(
        args.config,
        max_steps=args.max_steps,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )


def _validate_rft_config(config: dict) -> None:
    if config.get("model_path") != "post_train/model/qwen/qwen3-0.6b":
        raise ValueError("RFT must use the Qwen3-0.6B base model")
    if "rft_accepted" not in str(config.get("train_data", "")):
        raise ValueError("RFT train_data must point to accepted RFT data")
    if config.get("per_device_train_batch_size") != 4:
        raise ValueError("RFT per-device micro batch must be 4")
    if config.get("gradient_accumulation_steps") != 2:
        raise ValueError("RFT gradient accumulation must be 2")


if __name__ == "__main__":
    main()
