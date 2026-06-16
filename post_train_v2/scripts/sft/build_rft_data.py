from __future__ import annotations

import argparse

from post_train_v2.src.generation.rft import run_rft_rollout


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build V2 RFT rollout data.")
    parser.add_argument(
        "--config",
        default="post_train_v2/configs/sft/rft_rollout.yaml",
    )
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_rft_rollout(args.config, limit=args.limit)


if __name__ == "__main__":
    main()
