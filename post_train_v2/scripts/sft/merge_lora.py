from __future__ import annotations

import argparse

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
