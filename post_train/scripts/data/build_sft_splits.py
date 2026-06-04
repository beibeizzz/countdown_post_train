from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from post_train.src.countdown.config import load_yaml_config, resolve_path
from post_train.src.countdown.io import read_jsonl, write_jsonl, write_manifest
from post_train.src.countdown.sampling import stratified_sample


DEFAULT_CONFIG = "post_train/configs/data_build.yaml"
DEFAULT_ACCEPTED = "post_train/data/teacher_rollouts/teacher_accepted_20k.jsonl"
SFT_OUTPUT_DIR = "post_train/data/sft"
GRPO_OUTPUT_DIR = "post_train/data/grpo"
SFT_FILENAME = "sft_train_8k.jsonl"
GRPO_FILENAME = "grpo_train_4k.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SFT and GRPO train splits.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--accepted", default=DEFAULT_ACCEPTED)
    return parser.parse_args()


def require_pool_size(accepted_count: int, requested_size: int, split_name: str) -> None:
    if accepted_count < requested_size:
        raise ValueError(
            f"accepted pool has {accepted_count} rows, but {split_name} requires "
            f"{requested_size} rows"
        )


def main() -> None:
    args = parse_args()

    cfg_path = resolve_path(args.config, REPO_ROOT)
    cfg = load_yaml_config(cfg_path)
    accepted_path = resolve_path(args.accepted, REPO_ROOT)
    sft_output_dir = resolve_path(SFT_OUTPUT_DIR, REPO_ROOT)
    grpo_output_dir = resolve_path(GRPO_OUTPUT_DIR, REPO_ROOT)

    accepted_rows = read_jsonl(accepted_path)
    seed = int(cfg["seed"])
    sft_target = int(cfg["sft_train_target"])
    grpo_target = int(cfg["grpo_train_target"])

    require_pool_size(len(accepted_rows), sft_target, "SFT split")
    require_pool_size(len(accepted_rows), grpo_target, "GRPO split")

    sft_rows = stratified_sample(accepted_rows, size=sft_target, seed=seed + 10)
    grpo_rows = stratified_sample(accepted_rows, size=grpo_target, seed=seed + 20)

    write_jsonl(sft_output_dir / SFT_FILENAME, sft_rows)
    write_jsonl(grpo_output_dir / GRPO_FILENAME, grpo_rows)
    write_manifest(
        sft_output_dir / "manifest.json",
        {
            "name": "sft_and_grpo_splits",
            "num_accepted_pool": len(accepted_rows),
            "num_sft": len(sft_rows),
            "num_grpo": len(grpo_rows),
            "seed": seed,
        },
    )


if __name__ == "__main__":
    main()
