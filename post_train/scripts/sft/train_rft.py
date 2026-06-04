from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from post_train.scripts.sft.train_full import run_sft_training
from post_train.src.countdown.config import load_yaml_config, resolve_path


DEFAULT_CONFIG = "post_train/configs/rft.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RFT using the full SFT trainer and rft.train config.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--max-steps", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(resolve_path(args.config, REPO_ROOT))
    run_sft_training(cfg, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
