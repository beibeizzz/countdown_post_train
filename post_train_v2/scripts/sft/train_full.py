from __future__ import annotations

import argparse

from post_train_v2.src.training.supervised import run_supervised_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train V2 Full SFT with DDP.")
    parser.add_argument(
        "--config",
        default="post_train_v2/configs/sft/full.yaml",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--resume-from-checkpoint", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_supervised_training(
        args.config,
        max_steps=args.max_steps,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )


if __name__ == "__main__":
    main()
