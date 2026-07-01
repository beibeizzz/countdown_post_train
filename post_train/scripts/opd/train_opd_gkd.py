from __future__ import annotations

import argparse
import inspect
import math
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from post_train.scripts.sft.train_full import DEFAULT_EVAL_CONFIG, build_eval_callback, load_model_and_tokenizer
from post_train.src.countdown.config import load_yaml_config, resolve_path
from post_train.src.countdown.io import read_jsonl
from post_train.src.countdown.wandb_utils import (
    configure_wandb_env,
    formatted_run_name,
    is_wandb_enabled,
    trainer_report_to,
)


DEFAULT_CONFIG = "post_train/configs/opd_gkd.yaml"
DEFAULT_TRAIN_DATA = "post_train/data/grpo/grpo_train_4k.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="On-policy distillation with TRL GKDTrainer.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--max-steps", type=int, default=None)
    return parser.parse_args()


def _signature_accepts(signature: inspect.Signature, name: str) -> bool:
    if name in signature.parameters:
        return True
    return any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())


def _filter_kwargs(callable_obj, kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(callable_obj)
    return {key: value for key, value in kwargs.items() if _signature_accepts(signature, key)}


def prepare_opd_gkd_records(
    rows: Iterable[dict[str, Any]],
    response_field: str = "response",
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        prompt = row.get("prompt")
        response = row.get(response_field)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"OPD GKD row {index} field 'prompt' must be a non-empty string")
        if not isinstance(response, str) or not response.strip():
            raise ValueError(f"OPD GKD row {index} field {response_field!r} must be a non-empty string")

        record = dict(row)
        record["prompt"] = prompt
        record["response"] = response
        record["messages"] = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        records.append(record)

    if not records:
        raise ValueError("OPD GKD dataset is empty")
    return records


def build_opd_gkd_dataset(rows: Iterable[dict[str, Any]], response_field: str = "response"):
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise ImportError("OPD GKD training requires the 'datasets' package") from exc

    return Dataset.from_list(prepare_opd_gkd_records(rows, response_field=response_field))


def build_teacher_model_init_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "trust_remote_code": bool(cfg.get("trust_remote_code", True)),
    }
    attn_implementation = cfg.get("teacher_attn_implementation", "flash_attention_2")
    if attn_implementation:
        kwargs["attn_implementation"] = str(attn_implementation)
    teacher_dtype = cfg.get("teacher_dtype", "bfloat16" if bool(cfg.get("bf16", False)) else "auto")
    if teacher_dtype:
        # TRL GKDTrainer converts this string to torch.<dtype> before loading the teacher.
        kwargs["dtype"] = str(teacher_dtype)
    return kwargs


def build_gkd_training_args(
    cfg: dict[str, Any],
    output_dir: str | Path,
    max_steps: int,
    config_cls=None,
):
    if config_cls is None:
        try:
            from trl.experimental.gkd import GKDConfig
        except ImportError as exc:
            raise ImportError("OPD GKD training requires TRL with experimental.gkd.GKDConfig") from exc
        config_cls = GKDConfig

    configure_wandb_env(cfg)
    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "overwrite_output_dir": False,
        "max_steps": int(max_steps),
        "num_train_epochs": float(cfg.get("epochs", 1.0)),
        "per_device_train_batch_size": int(cfg["per_device_train_batch_size"]),
        "gradient_accumulation_steps": int(cfg.get("gradient_accumulation_steps", 1)),
        "learning_rate": float(cfg["learning_rate"]),
        "weight_decay": float(cfg.get("weight_decay", 0.0)),
        "lr_scheduler_type": str(cfg.get("scheduler", "cosine")),
        "bf16": bool(cfg.get("bf16", False)),
        "gradient_checkpointing": bool(cfg.get("gradient_checkpointing", False)),
        "save_strategy": "steps",
        "save_steps": int(cfg["save_every_steps"]),
        "logging_steps": int(cfg.get("logging_steps", 10)),
        "report_to": trainer_report_to(cfg),
        "run_name": formatted_run_name(cfg, default_name=Path(output_dir).name),
        "remove_unused_columns": False,
        "max_length": int(cfg["max_seq_len"]),
        "max_new_tokens": int(cfg["max_new_tokens"]),
        "temperature": float(cfg.get("temperature", 0.9)),
        "lmbda": float(cfg.get("lmbda", 1.0)),
        "beta": float(cfg.get("beta", 0.0)),
        "seq_kd": bool(cfg.get("seq_kd", False)),
        "teacher_model_name_or_path": str(cfg["teacher_model_path"]),
        "teacher_model_init_kwargs": build_teacher_model_init_kwargs(cfg),
        "dataset_kwargs": {"skip_prepare_dataset": True},
    }
    warmup_ratio = float(cfg.get("warmup_ratio", 0.0))
    if _signature_accepts(inspect.signature(config_cls), "warmup_steps"):
        kwargs["warmup_steps"] = int(math.ceil(int(max_steps) * warmup_ratio))
    else:
        kwargs["warmup_ratio"] = warmup_ratio
    return config_cls(**_filter_kwargs(config_cls, kwargs))


def build_gkd_trainer(
    model,
    tokenizer,
    teacher_model_path: str | Path,
    training_args,
    train_dataset,
    callbacks,
    trainer_cls=None,
):
    if trainer_cls is None:
        try:
            from trl.experimental.gkd import GKDTrainer
        except ImportError as exc:
            raise ImportError("OPD GKD training requires TRL with experimental.gkd.GKDTrainer") from exc
        trainer_cls = GKDTrainer

    kwargs = {
        "model": model,
        "teacher_model": str(teacher_model_path),
        "args": training_args,
        "train_dataset": train_dataset,
        "processing_class": tokenizer,
        "callbacks": callbacks,
    }
    return trainer_cls(**_filter_kwargs(trainer_cls, kwargs))


def run_opd_gkd_training(cfg: dict[str, Any], max_steps: int | None = None) -> None:
    student_model_path = resolve_path(cfg["student_model_path"], REPO_ROOT)
    teacher_model_path = resolve_path(cfg["teacher_model_path"], REPO_ROOT)
    train_data_path = resolve_path(cfg.get("train_data", DEFAULT_TRAIN_DATA), REPO_ROOT)
    output_dir = resolve_path(cfg["output_dir"], REPO_ROOT)
    effective_max_steps = int(max_steps if max_steps is not None else cfg["max_steps"])

    model, tokenizer = load_model_and_tokenizer(
        student_model_path,
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", False)),
    )
    rows = read_jsonl(train_data_path)
    train_dataset = build_opd_gkd_dataset(rows, response_field=str(cfg.get("response_field", "response")))
    training_args = build_gkd_training_args(cfg, output_dir, effective_max_steps)

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

    trainer = build_gkd_trainer(
        model,
        tokenizer,
        teacher_model_path,
        training_args,
        train_dataset,
        callbacks,
    )
    trainer.train()

    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(final_dir)


def main() -> None:
    args = parse_args()
    cfg_path = resolve_path(args.config, REPO_ROOT)
    cfg = load_yaml_config(cfg_path)
    run_opd_gkd_training(cfg, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
