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

from post_train_v2.src.training.lora import merge_lora_adapter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge a V2 LoRA adapter.")
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    merge_lora_adapter(
        base_model_path=args.base_model_path,
        adapter_path=args.adapter_path,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
