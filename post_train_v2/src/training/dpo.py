"""TRL DPO training core for V2."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

from post_train_v2.src.config.loading import load_yaml, resolve_repo_path
from post_train_v2.src.data.schema import validate_dpo_record, validate_normalized_source
from post_train_v2.src.data.splits import read_jsonl_strict
from post_train_v2.src.distributed.runtime import current_context
from post_train_v2.src.training.fixed_eval import FixedEvaluationCallback
from post_train_v2.src.training.model_loading import load_causal_lm_and_tokenizer
from post_train_v2.src.training.model_selection import (
    export_best_checkpoint,
    export_final_model,
)
from post_train_v2.src.tracking.wandb import finish_run, init_run


def run_dpo_training(
    config_path: str | Path,
    *,
    max_steps: int | None = None,
    resume_from_checkpoint: str | None = None,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    context = current_context()
    wandb_run = init_run(
        config.get("tracking", {"enabled": False}),
        rank=context.rank,
        stage="dpo",
    )
    model, tokenizer = load_causal_lm_and_tokenizer(
        resolve_repo_path(config["model_path"]),
        gradient_checkpointing=True,
    )
    rows = read_jsonl_strict(resolve_repo_path(config["train_data"]), validate_dpo_record)
    eval_rows = read_jsonl_strict(
        resolve_repo_path(config["eval_data"]),
        validate_normalized_source,
    )
    dataset = _dataset_from_pairs(rows)
    args = build_dpo_config(config, max_steps=max_steps)
    output_dir = resolve_repo_path(config["output_dir"])
    eval_callback = FixedEvaluationCallback(
        eval_rows=eval_rows,
        tokenizer=tokenizer,
        output_dir=output_dir / "eval",
        eval_every_steps=int(config["eval_every_steps"]),
        max_new_tokens=int(config["max_new_tokens"]),
    )
    trl = import_module("trl")
    trainer = trl.DPOTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[eval_callback],
    )
    if hasattr(trainer, "accelerator"):
        eval_callback.accelerator = trainer.accelerator
    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        export_dpo_outputs(
            trainer=trainer,
            tokenizer=tokenizer,
            output_dir=output_dir,
            export_kind=str(config.get("export_kind", "full_model")),
        )
    finally:
        finish_run(wandb_run)
    return {
        "trainer": trainer,
        "rank": context.rank,
        "world_size": context.world_size,
        "global_batch_size": (
            int(config["per_device_train_batch_size"])
            * int(config["gradient_accumulation_steps"])
            * context.world_size
        ),
    }


def build_dpo_config(config: dict[str, Any], *, max_steps: int | None = None):
    trl = import_module("trl")
    kwargs = {
        "output_dir": config["output_dir"],
        "learning_rate": config["learning_rate"],
        "num_train_epochs": config["num_train_epochs"],
        "per_device_train_batch_size": config["per_device_train_batch_size"],
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        "beta": config["beta"],
        "max_length": config["max_length"],
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "bf16": True,
        "gradient_checkpointing": True,
        "logging_strategy": "steps",
        "logging_steps": 1,
        "logging_first_step": True,
        "save_strategy": "steps",
        "save_steps": 100,
        "save_total_limit": 2,
        "remove_unused_columns": False,
        "report_to": config.get("report_to", []),
    }
    if max_steps is not None:
        kwargs["max_steps"] = max_steps
    return trl.DPOConfig(**kwargs)


def _dataset_from_pairs(rows: list[dict[str, Any]]):
    datasets = import_module("datasets")
    return datasets.Dataset.from_list(
        [
            {
                "prompt": row["prompt"],
                "chosen": row["chosen"],
                "rejected": row["rejected"],
            }
            for row in rows
        ]
    )


def export_dpo_outputs(
    *,
    trainer,
    tokenizer,
    output_dir: Path,
    export_kind: str,
) -> None:
    direct_load_check = _direct_load_check if export_kind == "full_model" else None
    export_final_model(
        trainer=trainer,
        tokenizer=tokenizer,
        output_dir=output_dir,
        export_kind=export_kind,
        direct_load_check=direct_load_check,
    )
    export_best_checkpoint(
        tokenizer=tokenizer,
        output_dir=output_dir,
        export_kind=export_kind,
        direct_load_check=direct_load_check,
    )


def _direct_load_check(path: Path) -> None:
    load_causal_lm_and_tokenizer(path, gradient_checkpointing=False)
