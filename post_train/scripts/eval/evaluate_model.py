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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Inference batch size for batched generate. Defaults to cfg['batch_size'] (eval.yaml).",
    )
    parser.add_argument(
        "--no-batch",
        action="store_true",
        help="Force serial evaluate_rows (batch=1) instead of batched inference.",
    )
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


def trim_generation_to_eos(generated_ids, eos_token_id) -> list[int]:
    """Trim a generated token-id sequence to the first EOS token (inclusive).

    model.generate with left-padding returns a fixed-length tail of
    max_new_tokens per row; tokens after the first EOS are padding/garbage, so
    reporting the raw shape as generated_tokens (always == max_new_tokens) is
    wrong. Slice at the first EOS so token counts and truncation reflect the
    real generation length. The EOS token is kept so is_truncated can still
    detect it (an EOS within the cap means not truncated). Returns the original
    list unchanged if eos_token_id is None.
    """
    if eos_token_id is None:
        return list(generated_ids)
    if isinstance(eos_token_id, int):
        eos_token_ids = {eos_token_id}
    else:
        eos_token_ids = {int(token_id) for token_id in eos_token_id}

    ids = [int(token_id) for token_id in generated_ids]
    for index, token_id in enumerate(ids):
        if token_id in eos_token_ids:
            return ids[: index + 1]
    return ids


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


def evaluate_rows_batched(
    rows: list[dict[str, Any]],
    tokenizer,
    model,
    cfg: dict[str, Any],
    batch_size: int = 32,
) -> list[dict[str, Any]]:
    """Batched counterpart of evaluate_rows.

    Generates completions for many prompts per forward pass instead of one at a
    time. Left-pads so the model attends to the real prompt on the right and the
    generated continuation starts at the correct position per row; each row is
    scored by the exact same score_generation() used in the serial path so the
    resulting metrics are identical to evaluate_rows (modulo numerical noise).
    """
    import torch

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if not rows:
        return []

    enable_thinking = bool(cfg.get("enable_thinking", False))
    max_new_tokens = int(cfg["max_new_tokens"])

    scored_rows: list[dict[str, Any]] = []

    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = eos_token_id

    # Generation models must be left-padded: right-padding puts pad tokens in the
    # middle of the prompt+continuation and corrupts the next-token prediction.
    # Flip the side locally and restore it on exit so the global tokenizer state
    # is unchanged for any other caller (e.g. training callbacks sharing it).
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    try:
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            rendered_prompts = [
                apply_chat_template(tokenizer, str(row["prompt"]), enable_thinking=enable_thinking)
                for row in batch
            ]
            encodings = tokenizer(rendered_prompts, return_tensors="pt", padding=True)
            device = getattr(model, "device", None)
            if device is not None:
                encodings = {key: value.to(device) for key, value in encodings.items()}

            kwargs = generation_kwargs(cfg)
            kwargs["pad_token_id"] = pad_token_id
            # generation_kwargs already sets max_new_tokens; don't pass it twice.
            kwargs.setdefault("max_new_tokens", max_new_tokens)

            with torch.inference_mode():
                output_ids = model.generate(**encodings, **kwargs)

            # Left-padding aligns every prompt's end to the same column, so the
            # generated tokens for every row begin right after the padded input
            # length -- slice uniformly with inlen rather than per-row prompt len.
            inlen = encodings["input_ids"].shape[1]
            for row, full_ids in zip(batch, output_ids, strict=True):
                # full_ids[inlen:] is a fixed max_new_tokens-length tail; tokens
                # past the first EOS are padding/garbage. Trim to EOS so the
                # token count and truncation flag reflect the real generation
                # length instead of always reporting max_new_tokens.
                generated_ids = trim_generation_to_eos(full_ids[inlen:], eos_token_id)
                raw_generation = tokenizer.decode(generated_ids, skip_special_tokens=True)
                generated_tokens = len(generated_ids)
                truncated = is_truncated(generated_ids, max_new_tokens, eos_token_id)
                scored_rows.append(
                    score_generation(
                        row,
                        raw_generation,
                        generated_tokens=generated_tokens,
                        truncated=truncated,
                    )
                )
    finally:
        tokenizer.padding_side = original_padding_side

    return scored_rows


def main() -> None:
    args = parse_args()

    cfg_path = resolve_path(args.config, REPO_ROOT)
    cfg = load_yaml_config(cfg_path)
    # Standalone CLI eval runs on the held-out TEST split (500 rows), not the
    # 50-row val_eval_50 subset used for in-training periodic eval. The val
    # subset is still consumed by training callbacks via evaluate_rows(), which
    # reads eval_subset themselves; this path only affects the CLI report.
    test_data_path = resolve_path(cfg["test_data"], REPO_ROOT)
    model_path = resolve_path(args.model_path, REPO_ROOT)
    base_model_path = resolve_path(args.base_model_path, REPO_ROOT) if args.base_model_path else None
    output_dir = resolve_path(args.output_dir, REPO_ROOT)

    rows = read_jsonl(test_data_path)
    if args.limit is not None:
        rows = rows[: args.limit]

    tokenizer, model = load_model_and_tokenizer(model_path, base_model_path=base_model_path)
    if args.no_batch:
        scored_rows = evaluate_rows(rows, tokenizer, model, cfg)
    else:
        batch_size = int(args.batch_size) if args.batch_size is not None else int(cfg.get("batch_size", 32))
        scored_rows = evaluate_rows_batched(rows, tokenizer, model, cfg, batch_size=batch_size)
    metrics = aggregate_eval_rows(scored_rows)

    write_jsonl(output_dir / "eval_samples.jsonl", scored_rows)
    write_json(output_dir / "eval_metrics.json", metrics)


if __name__ == "__main__":
    main()
