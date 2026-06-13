from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from post_train.src.countdown.config import load_yaml_config, resolve_path
from post_train.src.countdown.eval import aggregate_eval_rows, score_generation
from post_train.src.countdown.generation import apply_chat_template
from post_train.src.countdown.io import read_jsonl, write_json, write_jsonl


DEFAULT_CONFIG = "post_train/configs/eval.yaml"
DEFAULT_OUTPUT_DIR = "post_train/data/eval/default"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Countdown model.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--base-model-path",
        default=None,
        help="Base model path for PEFT/LoRA adapter checkpoints.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def load_model_and_tokenizer(model_path: Path, base_model_path: Path | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_config_path = model_path / "adapter_config.json"
    is_lora_adapter = adapter_config_path.exists()
    tokenizer_path = model_path

    if is_lora_adapter:
        resolved_base_model_path = base_model_path or _base_model_path_from_adapter_config(adapter_config_path)
        if resolved_base_model_path is None:
            raise ValueError(
                "LoRA adapter checkpoint detected. Pass --base-model-path or set "
                "base_model_name_or_path in adapter_config.json."
            )
        if not (model_path / "tokenizer_config.json").is_file():
            tokenizer_path = Path(resolved_base_model_path)
        from peft import PeftModel

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            resolved_base_model_path,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        )
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        )

    model.eval()

    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model


def _base_model_path_from_adapter_config(adapter_config_path: Path) -> str | None:
    payload = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    value = payload.get("base_model_name_or_path")
    if not isinstance(value, str) or not value.strip():
        return None
    return value


def generation_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    temperature = float(cfg.get("temperature", 0.0))
    kwargs: dict[str, Any] = {
        "max_new_tokens": int(cfg["max_new_tokens"]),
        "pad_token_id": None,
    }

    if temperature <= 0:
        kwargs["do_sample"] = False
    else:
        kwargs.update(
            {
                "do_sample": True,
                "temperature": temperature,
                "top_p": float(cfg.get("top_p", 1.0)),
            }
        )

    return kwargs


def is_truncated(generated_ids, max_new_tokens: int, eos_token_id) -> bool:
    generated_tokens = [int(token_id) for token_id in generated_ids]
    if len(generated_tokens) < max_new_tokens:
        return False

    if eos_token_id is None:
        return True

    if isinstance(eos_token_id, int):
        eos_token_ids = {eos_token_id}
    else:
        eos_token_ids = {int(token_id) for token_id in eos_token_id}

    return not any(token_id in eos_token_ids for token_id in generated_tokens)


def generate_one(tokenizer, model, prompt: str, cfg: dict[str, Any]) -> tuple[str, int, bool]:
    import torch

    rendered_prompt = apply_chat_template(
        tokenizer,
        prompt,
        enable_thinking=bool(cfg.get("enable_thinking", False)),
    )
    inputs = tokenizer(rendered_prompt, return_tensors="pt")
    device = getattr(model, "device", None)
    if device is not None:
        inputs = {key: value.to(device) for key, value in inputs.items()}

    kwargs = generation_kwargs(cfg)
    kwargs["pad_token_id"] = tokenizer.pad_token_id
    if kwargs["pad_token_id"] is None:
        kwargs.pop("pad_token_id")

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **kwargs)

    input_tokens = inputs["input_ids"].shape[-1]
    generated_ids = output_ids[0][input_tokens:]
    raw_generation = tokenizer.decode(generated_ids, skip_special_tokens=True)
    generated_tokens = int(generated_ids.shape[-1])
    truncated = is_truncated(generated_ids, int(cfg["max_new_tokens"]), tokenizer.eos_token_id)
    return raw_generation, generated_tokens, truncated


def evaluate_rows(
    rows: list[dict[str, Any]],
    tokenizer,
    model,
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    scored_rows: list[dict[str, Any]] = []

    for row in rows:
        raw_generation, generated_tokens, truncated = generate_one(tokenizer, model, row["prompt"], cfg)
        scored_rows.append(
            score_generation(
                row,
                raw_generation,
                generated_tokens=generated_tokens,
                truncated=truncated,
            )
        )

    return scored_rows


def main() -> None:
    args = parse_args()

    cfg_path = resolve_path(args.config, REPO_ROOT)
    cfg = load_yaml_config(cfg_path)
    eval_subset_path = resolve_path(cfg["eval_subset"], REPO_ROOT)
    model_path = resolve_path(args.model_path, REPO_ROOT)
    base_model_path = resolve_path(args.base_model_path, REPO_ROOT) if args.base_model_path else None
    output_dir = resolve_path(args.output_dir, REPO_ROOT)

    rows = read_jsonl(eval_subset_path)
    if args.limit is not None:
        rows = rows[: args.limit]

    tokenizer, model = load_model_and_tokenizer(model_path, base_model_path=base_model_path)
    scored_rows = evaluate_rows(rows, tokenizer, model, cfg)
    metrics = aggregate_eval_rows(scored_rows)

    write_jsonl(output_dir / "eval_samples.jsonl", scored_rows)
    write_json(output_dir / "eval_metrics.json", metrics)


if __name__ == "__main__":
    main()
