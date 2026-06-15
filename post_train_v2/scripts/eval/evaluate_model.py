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

from post_train_v2.src.evaluation.cli import run_evaluation


DEFAULT_CONFIG = REPO_ROOT / "post_train_v2/configs/common/eval.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a V2 Countdown model.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--base-model-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--limit", type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_evaluation(
        args.config,
        args.model_path,
        base_model_path=args.base_model_path,
        output_dir=args.output_dir,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

