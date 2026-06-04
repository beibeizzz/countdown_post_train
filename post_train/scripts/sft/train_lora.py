from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from post_train.scripts.sft.train_full import (
    DEFAULT_EVAL_CONFIG,
    DataCollatorForCausalSFT,
    SFTDataset,
    build_eval_callback,
    build_trainer,
    build_training_arguments,
    load_model_and_tokenizer,
    tokenize_rows,
)
from post_train.src.countdown.config import load_yaml_config, resolve_path
from post_train.src.countdown.io import read_jsonl


DEFAULT_CONFIG = "post_train/configs/sft_lora.yaml"
AUTO_TARGET_MODULE_SUFFIXES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "up_proj",
    "down_proj",
    "gate_proj",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA SFT for Qwen3-0.6B on Countdown.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--max-steps", type=int, default=None)
    return parser.parse_args()


def resolve_lora_target_modules(config_value: Any, model) -> list[str]:
    if config_value == "auto":
        module_names = [name for name, _module in model.named_modules()]
        resolved = [
            suffix
            for suffix in AUTO_TARGET_MODULE_SUFFIXES
            if any(name == suffix or name.endswith(f".{suffix}") for name in module_names)
        ]
        if not resolved:
            raise ValueError(
                "No LoRA target modules found. Expected one of these module suffixes in model.named_modules(): "
                f"{', '.join(AUTO_TARGET_MODULE_SUFFIXES)}"
            )
        return resolved

    if isinstance(config_value, str):
        resolved = [module.strip() for module in config_value.split(",") if module.strip()]
    elif isinstance(config_value, list):
        resolved = [str(module).strip() for module in config_value if str(module).strip()]
    else:
        raise TypeError("lora_target_modules must be 'auto', a comma-separated string, or a list")

    if not resolved:
        raise ValueError("lora_target_modules must resolve to at least one module")
    return resolved


def trainable_parameter_summary(model) -> tuple[int, int, float]:
    trainable = 0
    total = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    percent = 100.0 * trainable / total if total else 0.0
    return trainable, total, percent


def apply_lora(model, cfg: dict[str, Any]):
    from peft import LoraConfig, get_peft_model

    target_modules = resolve_lora_target_modules(cfg["lora_target_modules"], model)
    lora_config = LoraConfig(
        r=int(cfg["lora_r"]),
        lora_alpha=int(cfg["lora_alpha"]),
        lora_dropout=float(cfg["lora_dropout"]),
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_config)


def main() -> None:
    args = parse_args()

    cfg_path = resolve_path(args.config, REPO_ROOT)
    cfg = load_yaml_config(cfg_path)
    model_path = resolve_path(cfg["model_path"], REPO_ROOT)
    train_data_path = resolve_path(cfg["train_data"], REPO_ROOT)
    output_dir = resolve_path(cfg["output_dir"], REPO_ROOT)

    gradient_checkpointing = bool(cfg.get("gradient_checkpointing", False))
    model, tokenizer = load_model_and_tokenizer(model_path, gradient_checkpointing)
    model = apply_lora(model, cfg)
    if gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    trainable, total, percent = trainable_parameter_summary(model)
    print(f"Trainable parameters: {trainable:,} / {total:,} ({percent:.4f}%)")

    rows = read_jsonl(train_data_path)
    examples = tokenize_rows(
        tokenizer,
        rows,
        int(cfg["max_seq_len"]),
        enable_thinking=bool(cfg.get("enable_thinking", False)),
    )
    train_dataset = SFTDataset(examples)
    collator = DataCollatorForCausalSFT(pad_token_id=int(tokenizer.pad_token_id))

    training_args = build_training_arguments(cfg, output_dir, args.max_steps)

    callbacks = []
    eval_every_steps = int(cfg.get("eval_every_steps", 0))
    eval_cfg_path = resolve_path(DEFAULT_EVAL_CONFIG, REPO_ROOT)
    if eval_every_steps > 0 and eval_cfg_path.exists():
        eval_cfg = load_yaml_config(eval_cfg_path)
        if "val_data" in cfg:
            eval_cfg["val_data"] = cfg["val_data"]
        callbacks.append(build_eval_callback(output_dir, eval_every_steps, eval_cfg))

    trainer = build_trainer(model, training_args, train_dataset, collator, tokenizer, callbacks)
    trainer.train()

    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(final_dir)


if __name__ == "__main__":
    main()
