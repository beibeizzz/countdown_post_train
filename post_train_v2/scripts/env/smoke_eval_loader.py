from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from post_train.scripts.eval.evaluate_model import load_model_and_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test the legacy full or LoRA evaluation loader with deterministic generation."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--base-model-path",
        default=None,
        help="Required when --model-path is a LoRA adapter directory.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=8)
    return parser.parse_args()


def validate_paths(model_path: Path, base_model_path: Path | None) -> None:
    if (model_path / "adapter_config.json").exists() and base_model_path is None:
        raise ValueError("--base-model-path is required when --model-path is a LoRA adapter")


def _normalize_device(value: Any) -> Any:
    if isinstance(value, int):
        return f"cuda:{value}"
    return value


def input_device(model) -> Any:
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict) and device_map:
        embedding_devices = [
            value
            for key, value in device_map.items()
            if "embed" in str(key).lower() and str(value) != "disk"
        ]
        candidates = embedding_devices or [
            value for value in device_map.values() if str(value) != "disk"
        ]
        if candidates:
            return _normalize_device(candidates[0])

    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise RuntimeError("Evaluation model has no parameters and no usable hf_device_map") from exc


def run_smoke(
    model_path: Path,
    base_model_path: Path | None,
    max_new_tokens: int,
) -> str:
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")
    validate_paths(model_path, base_model_path)

    tokenizer, model = load_model_and_tokenizer(
        model_path,
        base_model_path=base_model_path,
    )
    inputs = tokenizer(
        "Using the numbers 1, 1, 1, 1, create an equation that equals 4.",
        return_tensors="pt",
    )
    device = input_device(model)
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    prompt_length = inputs["input_ids"].shape[-1]
    generated_ids = output_ids[0][prompt_length:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    if not text:
        raise RuntimeError("Evaluation loader smoke produced an empty generation")

    print(f"OK eval_loader device={device} text={text!r}")
    return text


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    base_model_path = Path(args.base_model_path) if args.base_model_path else None
    run_smoke(
        model_path=model_path,
        base_model_path=base_model_path,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
