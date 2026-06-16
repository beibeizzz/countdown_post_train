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

from post_train_v2.verl.launch.train_grpo import run_grpo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch stock verl GRPO.")
    parser.add_argument("--config", default="post_train_v2/verl/configs/grpo.yaml")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--resume-from-checkpoint", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_grpo(
        config_path=args.config,
        max_steps=args.max_steps,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )


if __name__ == "__main__":
    raise SystemExit(main())
