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

from post_train_v2.src.generation.dpo import run_build_dpo_data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build V2 DPO pair data.")
    parser.add_argument("--config", default="post_train_v2/configs/dpo/build.yaml")
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_build_dpo_data(args.config, limit=args.limit)


if __name__ == "__main__":
    main()
