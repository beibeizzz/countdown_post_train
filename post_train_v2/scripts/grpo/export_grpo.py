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

from post_train_v2.verl.export.merge_actor import export_grpo_actors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export best/final verl GRPO actors.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--selection-json", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--prune", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    export_grpo_actors(
        args.run_dir,
        selection_path=args.selection_json,
        output_dir=args.output_dir,
        prune=args.prune,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
