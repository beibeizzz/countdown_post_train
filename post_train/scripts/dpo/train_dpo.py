from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from post_train.scripts.sft.train_full import (
    DEFAULT_EVAL_CONFIG,
    apply_chat_template_compat,
    build_eval_callback,
    load_model_and_tokenizer,
)
from post_train.src.countdown.config import load_yaml_config, resolve_path
from post_train.src.countdown.io import read_jsonl
from post_train.src.countdown.wandb_utils import configure_wandb_env, formatted_run_name, trainer_report_to


DEFAULT_CONFIG = "post_train/configs/dpo_train.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DPO training for the Countdown SFT model.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--max-steps", type=int, default=None)
    return parser.parse_args()


def prepare_dpo_records(rows: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        record: dict[str, str] = {}
        for key in ("prompt", "chosen", "rejected"):
            value = row.get(key)
            if not isinstance(value, str):
                raise ValueError(f"DPO row {index} field {key!r} must be a string")
            record[key] = value
        records.append(record)
    if not records:
        raise ValueError("DPO dataset is empty")
    return records


def format_dpo_record_for_trl(
    record: dict[str, str],
    tokenizer,
    enable_thinking: bool = False,
) -> dict[str, str]:
    prompt = apply_chat_template_compat(
        tokenizer,
        [{"role": "user", "content": record["prompt"]}],
        enable_thinking=enable_thinking,
        tokenize=False,
        add_generation_prompt=True,
    )
    return {
        "prompt": prompt,
        "chosen": record["chosen"],
        "rejected": record["rejected"],
    }


def format_dpo_records_for_trl(
    records: Iterable[dict[str, str]],
    tokenizer,
    enable_thinking: bool = False,
) -> list[dict[str, str]]:
    return [
        format_dpo_record_for_trl(record, tokenizer, enable_thinking=enable_thinking)
        for record in records
    ]


def _signature_accepts(signature: inspect.Signature, name: str) -> bool:
    if name in signature.parameters:
        return True
    return any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())


def _filter_kwargs(callable_obj, kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(callable_obj)
    return {key: value for key, value in kwargs.items() if _signature_accepts(signature, key)}


def build_dpo_training_arguments(cfg: dict[str, Any], output_dir: Path, max_steps: int | None):
    try:
        from trl import DPOConfig

        arguments_class = DPOConfig
    except ImportError:
        from transformers import TrainingArguments

        arguments_class = TrainingArguments

    configure_wandb_env(cfg)
    kwargs = {
        "output_dir": str(output_dir),
        "overwrite_output_dir": False,
        "num_train_epochs": float(cfg["epochs"]),
        "max_steps": max_steps if max_steps is not None else -1,
        "per_device_train_batch_size": int(cfg["per_device_train_batch_size"]),
        "gradient_accumulation_steps": int(cfg["gradient_accumulation_steps"]),
        "learning_rate": float(cfg["learning_rate"]),
        "weight_decay": float(cfg["weight_decay"]),
        "warmup_ratio": float(cfg["warmup_ratio"]),
        "lr_scheduler_type": str(cfg.get("scheduler", "cosine")),
        "bf16": bool(cfg.get("bf16", False)),
        "gradient_checkpointing": bool(cfg.get("gradient_checkpointing", False)),
        "save_strategy": "steps",
        "save_steps": int(cfg["save_every_steps"]),
        "eval_steps": int(cfg["eval_every_steps"]),
        "logging_steps": int(cfg.get("logging_steps", 10)),
        "report_to": trainer_report_to(cfg),
        "run_name": formatted_run_name(cfg, default_name=output_dir.name),
        "remove_unused_columns": False,
        "beta": float(cfg["beta"]),
        "max_length": int(cfg["max_seq_len"]),
        "max_prompt_length": int(cfg["max_seq_len"]),
    }
    return arguments_class(**_filter_kwargs(arguments_class, kwargs))


def build_preference_dataset(records: list[dict[str, str]]):
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise ImportError("DPO training requires the 'datasets' package in the target environment") from exc

    return Dataset.from_list(records)


def build_dpo_trainer(model, training_args, train_dataset, tokenizer, callbacks, cfg: dict[str, Any]):
    try:
        from trl import DPOTrainer
    except ImportError as exc:
        raise ImportError("DPO training requires TRL with DPOTrainer available in the target environment") from exc

    trainer_signature = inspect.signature(DPOTrainer)
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "callbacks": callbacks,
    }

    if _signature_accepts(trainer_signature, "processing_class"):
        trainer_kwargs["processing_class"] = tokenizer
    elif _signature_accepts(trainer_signature, "tokenizer"):
        trainer_kwargs["tokenizer"] = tokenizer

    for key, value in {
        "beta": float(cfg["beta"]),
        "max_length": int(cfg["max_seq_len"]),
        "max_prompt_length": int(cfg["max_seq_len"]),
    }.items():
        if _signature_accepts(trainer_signature, key):
            trainer_kwargs[key] = value

    return DPOTrainer(**trainer_kwargs)


def main() -> None:
    args = parse_args()

    cfg_path = resolve_path(args.config, REPO_ROOT)
    cfg = load_yaml_config(cfg_path)
    model_path = resolve_path(cfg["model_path"], REPO_ROOT)
    train_data_path = resolve_path(cfg["train_data"], REPO_ROOT)
    output_dir = resolve_path(cfg["output_dir"], REPO_ROOT)

    gradient_checkpointing = bool(cfg.get("gradient_checkpointing", False))
    model, tokenizer = load_model_and_tokenizer(model_path, gradient_checkpointing)

    rows = read_jsonl(train_data_path)
    records = prepare_dpo_records(rows)
    formatted_records = format_dpo_records_for_trl(
        records,
        tokenizer,
        enable_thinking=bool(cfg.get("enable_thinking", False)),
    )
    train_dataset = build_preference_dataset(formatted_records)

    training_args = build_dpo_training_arguments(cfg, output_dir, args.max_steps)

    callbacks = []
    eval_every_steps = int(cfg.get("eval_every_steps", 0))
    eval_cfg_path = resolve_path(DEFAULT_EVAL_CONFIG, REPO_ROOT)
    if eval_every_steps > 0 and eval_cfg_path.exists():
        eval_cfg = load_yaml_config(eval_cfg_path)
        eval_cfg["eval_subset"] = "post_train/data/processed/val_eval_50.jsonl"
        if "val_data" in cfg:
            eval_cfg["val_data"] = cfg["val_data"]
        callbacks.append(build_eval_callback(output_dir, eval_every_steps, eval_cfg))

    trainer = build_dpo_trainer(model, training_args, train_dataset, tokenizer, callbacks, cfg)
    trainer.train()

    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(final_dir)


if __name__ == "__main__":
    main()
