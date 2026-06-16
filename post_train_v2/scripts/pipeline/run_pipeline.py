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

from post_train_v2.src.pipeline.runner import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the V2 post-training pipeline.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--from-stage", default=None)
    parser.add_argument("--through-stage", default=None)
    parser.add_argument("--rebuild-stage", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_pipeline(
        args.config,
        from_stage=args.from_stage,
        through_stage=args.through_stage,
        rebuild_stages=set(args.rebuild_stage),
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
