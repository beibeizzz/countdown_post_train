"""Shared supervised training runner for Full SFT and RFT."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib import import_module
from pathlib import Path
from typing import Any

from post_train_v2.src.config.loading import load_yaml, resolve_repo_path
from post_train_v2.src.data.schema import validate_normalized_source, validate_sft_record
from post_train_v2.src.data.splits import read_jsonl_strict
from post_train_v2.src.distributed.runtime import current_context
from post_train_v2.src.training.fixed_eval import FixedEvaluationCallback
from post_train_v2.src.training.model_loading import load_causal_lm_and_tokenizer
from post_train_v2.src.training.supervised_data import (
    SupervisedDataCollator,
    encode_prompt_response,
)
from post_train_v2.src.training.trainer_args import build_training_arguments


class SupervisedDataset:
    def __init__(self, rows: Sequence[Mapping[str, list[int]]]) -> None:
        self._rows = [dict(row) for row in rows]

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return dict(self._rows[index])


def global_batch_size(config: Mapping[str, Any], world_size: int) -> int:
    return (
        int(config["per_device_train_batch_size"])
        * int(config["gradient_accumulation_steps"])
        * int(world_size)
    )


def run_supervised_training(
    config_path: str | Path,
    *,
    max_steps: int | None = None,
    resume_from_checkpoint: str | None = None,
    model_adapter=None,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    context = current_context()
    model, tokenizer = load_causal_lm_and_tokenizer(
        resolve_repo_path(config["model_path"]),
        gradient_checkpointing=True,
    )
    if model_adapter is not None:
        model = model_adapter(model, config)

    train_rows = read_jsonl_strict(
        resolve_repo_path(config["train_data"]),
        validate_sft_record,
    )
    train_dataset = SupervisedDataset(
        _encode_rows(train_rows, tokenizer, int(config["max_seq_len"]))
    )
    eval_rows = read_jsonl_strict(
        resolve_repo_path(config["eval_data"]),
        validate_normalized_source,
    )
    output_dir = resolve_repo_path(config["output_dir"])
    args = build_training_arguments(config, max_steps=max_steps)
    transformers = import_module("transformers")
    trainer = transformers.Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=SupervisedDataCollator(),
        tokenizer=tokenizer,
        callbacks=[
            FixedEvaluationCallback(
                eval_rows=eval_rows,
                tokenizer=tokenizer,
                output_dir=output_dir / "eval",
                eval_every_steps=int(config["eval_every_steps"]),
                max_new_tokens=int(config["max_new_tokens"]),
            )
        ],
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    return {
        "rank": context.rank,
        "world_size": context.world_size,
        "global_batch_size": global_batch_size(config, context.world_size),
        "output_dir": str(output_dir),
    }


def _encode_rows(rows, tokenizer, max_seq_len: int) -> list[dict[str, list[int]]]:
    encoded_rows = []
    for row in rows:
        encoded = encode_prompt_response(
            tokenizer=tokenizer,
            prompt=row["prompt"],
            response=row["response"],
            max_seq_len=max_seq_len,
        )
        encoded_rows.append(
            {
                "input_ids": encoded.input_ids,
                "attention_mask": encoded.attention_mask,
                "labels": encoded.labels,
            }
        )
    return encoded_rows
