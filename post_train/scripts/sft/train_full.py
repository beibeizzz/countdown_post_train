from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from post_train.src.countdown.config import load_yaml_config, resolve_path
from post_train.src.countdown.generation import _supports_enable_thinking
from post_train.src.countdown.io import read_jsonl, write_json, write_jsonl
from post_train.src.countdown.wandb_utils import (
    configure_wandb_env,
    formatted_run_name,
    is_wandb_enabled,
    prefixed_metrics,
    trainer_report_to,
)


DEFAULT_CONFIG = "post_train/configs/sft_full.yaml"
DEFAULT_EVAL_CONFIG = "post_train/configs/eval.yaml"
DEFAULT_RFT_TARGET_MODEL = "post_train/model/qwen/qwen3-0.6b"
DEFAULT_VAL_DATA = "post_train/data/processed/val_200.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-parameter SFT for Qwen3-0.6B on Countdown.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--max-steps", type=int, default=None)
    return parser.parse_args()


def normalize_sft_config(raw_cfg: dict[str, Any]) -> dict[str, Any]:
    if "train" not in raw_cfg:
        return dict(raw_cfg)

    train_cfg = dict(raw_cfg["train"])
    merged = {
        "model_path": raw_cfg.get("target_model_path", DEFAULT_RFT_TARGET_MODEL),
        "train_data": raw_cfg["accepted_output"],
        "val_data": raw_cfg.get("val_data", DEFAULT_VAL_DATA),
        "output_dir": raw_cfg["output_dir"],
        "weight_decay": train_cfg.get("weight_decay", 0.0),
        "eval_every_steps": train_cfg.get("eval_every_steps", 100),
        "save_every_steps": train_cfg.get("save_every_steps", 100),
        "enable_thinking": raw_cfg.get("enable_thinking", False),
    }
    merged.update(train_cfg)
    return merged


def build_eval_wandb_metrics(metrics: dict[str, Any]) -> dict[str, float | int]:
    return prefixed_metrics("eval", metrics)


def apply_chat_template_compat(tokenizer, messages: list[dict[str, str]], enable_thinking: bool, **kwargs) -> str:
    if _supports_enable_thinking(tokenizer.apply_chat_template):
        return tokenizer.apply_chat_template(messages, enable_thinking=enable_thinking, **kwargs)
    return tokenizer.apply_chat_template(messages, **kwargs)


def encode_prompt_response(
    tokenizer,
    prompt: str,
    response: str,
    max_seq_len: int,
    enable_thinking: bool = False,
) -> dict | None:
    prompt_text = apply_chat_template_compat(
        tokenizer,
        [{"role": "user", "content": prompt}],
        enable_thinking=enable_thinking,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = apply_chat_template_compat(
        tokenizer,
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
        enable_thinking=enable_thinking,
        tokenize=False,
        add_generation_prompt=False,
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Prompt tokens must be a prefix of full prompt-response tokens")
    if len(full_ids) > max_seq_len:
        full_ids = full_ids[:max_seq_len]
    labels = list(full_ids)
    prompt_len = min(len(prompt_ids), len(labels))
    labels[:prompt_len] = [-100] * prompt_len
    if all(label == -100 for label in labels):
        return None
    return {"input_ids": full_ids, "labels": labels, "attention_mask": [1] * len(full_ids)}


class SFTDataset:
    def __init__(self, examples: list[dict[str, list[int]]]) -> None:
        if not examples:
            raise ValueError("SFT dataset is empty after tokenization")
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.examples[index]


class DataCollatorForCausalSFT:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        max_len = max(len(feature["input_ids"]) for feature in features)
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        labels: list[list[int]] = []

        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append(feature["attention_mask"] + [0] * pad_len)
            labels.append(feature["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def tokenize_rows(
    tokenizer,
    rows: list[dict[str, Any]],
    max_seq_len: int,
    enable_thinking: bool = False,
) -> list[dict[str, list[int]]]:
    examples: list[dict[str, list[int]]] = []
    for row in rows:
        encoded = encode_prompt_response(
            tokenizer,
            row["prompt"],
            row["response"],
            max_seq_len,
            enable_thinking=enable_thinking,
        )
        if encoded is not None:
            examples.append(encoded)
    return examples


def build_eval_callback(
    output_dir: Path,
    eval_every_steps: int,
    eval_cfg: dict[str, Any],
    wandb_enabled: bool = False,
):
    from transformers import TrainerCallback

    from post_train.scripts.eval.evaluate_model import evaluate_rows
    from post_train.src.countdown.eval import aggregate_eval_rows

    eval_subset_path = resolve_path(eval_cfg["eval_subset"], REPO_ROOT)
    eval_rows = read_jsonl(eval_subset_path)

    class CountdownEvalCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if eval_every_steps <= 0 or state.global_step <= 0:
                return control
            if state.global_step % eval_every_steps != 0:
                return control

            model = kwargs["model"]
            tokenizer = kwargs.get("processing_class") or kwargs.get("tokenizer")
            if tokenizer is None:
                return control
            was_training = model.training
            model.eval()

            scored_rows = evaluate_rows(eval_rows, tokenizer, model, eval_cfg)
            metrics = aggregate_eval_rows(scored_rows)

            step_dir = output_dir / "eval" / f"step_{state.global_step}"
            write_jsonl(step_dir / "eval_samples.jsonl", scored_rows)
            write_json(step_dir / "eval_metrics.json", metrics)
            if wandb_enabled:
                try:
                    import wandb
                except ImportError:
                    wandb = None
                if wandb is not None and getattr(wandb, "run", None) is not None:
                    wandb.log(build_eval_wandb_metrics(metrics), step=int(state.global_step))

            if was_training:
                model.train()
            return control

    return CountdownEvalCallback()


def load_model_and_tokenizer(model_path: Path, gradient_checkpointing: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )
    if gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

    return model, tokenizer


def build_training_arguments(cfg: dict[str, Any], output_dir: Path, max_steps: int | None):
    from transformers import TrainingArguments

    configure_wandb_env(cfg)
    return TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=False,
        num_train_epochs=float(cfg["epochs"]),
        max_steps=max_steps if max_steps is not None else -1,
        per_device_train_batch_size=int(cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        learning_rate=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
        warmup_ratio=float(cfg["warmup_ratio"]),
        lr_scheduler_type=str(cfg.get("scheduler", "cosine")),
        bf16=bool(cfg.get("bf16", False)),
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", False)),
        save_strategy="steps",
        save_steps=int(cfg["save_every_steps"]),
        logging_steps=int(cfg.get("logging_steps", 10)),
        report_to=trainer_report_to(cfg),
        run_name=formatted_run_name(cfg, default_name=output_dir.name),
        remove_unused_columns=False,
    )


def build_trainer(model, training_args, train_dataset, collator, tokenizer, callbacks):
    from transformers import Trainer

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "data_collator": collator,
        "callbacks": callbacks,
    }
    trainer_signature = inspect.signature(Trainer)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    return Trainer(**trainer_kwargs)


def run_sft_training(cfg: dict[str, Any], max_steps: int | None = None) -> None:
    cfg = normalize_sft_config(cfg)
    model_path = resolve_path(cfg["model_path"], REPO_ROOT)
    train_data_path = resolve_path(cfg["train_data"], REPO_ROOT)
    output_dir = resolve_path(cfg["output_dir"], REPO_ROOT)

    gradient_checkpointing = bool(cfg.get("gradient_checkpointing", False))
    model, tokenizer = load_model_and_tokenizer(model_path, gradient_checkpointing)

    rows = read_jsonl(train_data_path)
    examples = tokenize_rows(
        tokenizer,
        rows,
        int(cfg["max_seq_len"]),
        enable_thinking=bool(cfg.get("enable_thinking", False)),
    )
    train_dataset = SFTDataset(examples)
    collator = DataCollatorForCausalSFT(pad_token_id=int(tokenizer.pad_token_id))

    training_args = build_training_arguments(cfg, output_dir, max_steps)

    callbacks = []
    eval_every_steps = int(cfg.get("eval_every_steps", 0))
    eval_cfg_path = resolve_path(DEFAULT_EVAL_CONFIG, REPO_ROOT)
    if eval_every_steps > 0 and eval_cfg_path.exists():
        eval_cfg = load_yaml_config(eval_cfg_path)
        if "val_data" in cfg:
            eval_cfg["val_data"] = cfg["val_data"]
        callbacks.append(
            build_eval_callback(
                output_dir,
                eval_every_steps,
                eval_cfg,
                wandb_enabled=is_wandb_enabled(cfg),
            )
        )

    trainer = build_trainer(model, training_args, train_dataset, collator, tokenizer, callbacks)
    trainer.train()

    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(final_dir)


def main() -> None:
    args = parse_args()

    cfg_path = resolve_path(args.config, REPO_ROOT)
    cfg = load_yaml_config(cfg_path)
    run_sft_training(cfg, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
