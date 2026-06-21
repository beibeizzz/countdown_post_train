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

from post_train_v2.src.data.splits import (
    run_accepted_splits,
    run_validation_splits,
)


DEFAULT_CONFIG = REPO_ROOT / "post_train_v2/configs/data/build_splits.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic V2 validation and training splits."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("mode", choices=("validation", "accepted"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "validation":
        run_validation_splits(args.config)
    else:
        run_accepted_splits(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
