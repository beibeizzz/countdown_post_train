from __future__ import annotations

import argparse
import inspect
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from post_train.scripts.grpo.train_grpo import (
    METRICS_FILENAME,
    grpo_metric_summary,
    length_penalty_value,
)
from post_train.scripts.sft.train_full import (
    DEFAULT_EVAL_CONFIG,
    build_eval_callback,
    load_model_and_tokenizer,
)
from post_train.src.countdown.config import load_yaml_config, resolve_path
from post_train.src.countdown.io import read_jsonl
from post_train.src.countdown.validation import extract_answer_text, validate_countdown_response
from post_train.src.countdown.wandb_utils import (
    configure_wandb_env,
    formatted_run_name,
    is_wandb_enabled,
    prefixed_metrics,
    trainer_report_to,
)


DEFAULT_CONFIG = "post_train/configs/grpo_trl.yaml"
DEFAULT_TRAIN_DATA = "post_train/data/grpo/grpo_train_4k.jsonl"

_REWARD_DIAGNOSTICS: list[dict[str, Any]] = []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TRL GRPO training for Countdown.")
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


def _first_supported_name(callable_obj, names: Iterable[str]) -> str | None:
    signature = inspect.signature(callable_obj)
    for name in names:
        if _signature_accepts(signature, name):
            return name
    return None


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
        handle.write("\n")


def prepare_grpo_records(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        prompt = row.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"GRPO row {index} field 'prompt' must be a non-empty string")
        if not isinstance(row.get("numbers"), list):
            raise ValueError(f"GRPO row {index} field 'numbers' must be a list")
        if "target" not in row:
            raise ValueError(f"GRPO row {index} field 'target' is required")

        record = {key: value for key, value in row.items() if key != "prompt"}
        record["prompt"] = [{"role": "user", "content": prompt}]
        record["source_prompt"] = prompt
        records.append(record)

    if not records:
        raise ValueError("GRPO dataset is empty")
    return records


def build_grpo_dataset(rows: Iterable[dict[str, Any]]):
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise ImportError("TRL GRPO training requires the 'datasets' package") from exc

    return Dataset.from_list(prepare_grpo_records(rows))


def _completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    if isinstance(completion, (list, tuple)):
        parts: list[str] = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part).strip()
    return str(completion)


def _batched_column(value: Any, expected_len: int, name: str) -> list[Any]:
    if isinstance(value, list) and len(value) == expected_len:
        return value
    if expected_len == 1:
        return [value]
    raise ValueError(f"Reward column {name!r} must have {expected_len} values")


def consume_reward_diagnostics() -> list[dict[str, Any]]:
    diagnostics = list(_REWARD_DIAGNOSTICS)
    _REWARD_DIAGNOSTICS.clear()
    return diagnostics


def countdown_reward_func(
    completions,
    numbers,
    target,
    completion_ids=None,
    *,
    format_reward: float = 0.5,
    answer_reward: float = 1.0,
    max_completion_length: int = 1024,
    max_new_tokens: int | None = None,
    length_penalty_start: int = 800,
    length_penalty_max: float = -0.5,
    log_metric=None,
    log_extra=None,
    **kwargs,
) -> list[float]:
    completion_texts = [_completion_to_text(completion) for completion in completions]
    rollout_count = len(completion_texts)
    numbers_batch = _batched_column(numbers, rollout_count, "numbers")
    target_batch = _batched_column(target, rollout_count, "target")
    completion_id_batch = (
        completion_ids
        if isinstance(completion_ids, list) and len(completion_ids) == rollout_count
        else [None] * rollout_count
    )

    cap = int(max_new_tokens if max_new_tokens is not None else max_completion_length)
    rewards: list[float] = []
    diagnostics: list[dict[str, Any]] = []

    for text, row_numbers, row_target, token_ids in zip(
        completion_texts,
        numbers_batch,
        target_batch,
        completion_id_batch,
        strict=True,
    ):
        token_count = len(token_ids) if token_ids is not None else None
        has_format = extract_answer_text(text) is not None
        result = validate_countdown_response(text, list(row_numbers), int(row_target))
        reward = (float(format_reward) if has_format else 0.0) + (
            float(answer_reward) if result.ok else 0.0
        )
        reward += length_penalty_value(
            token_count,
            cap,
            int(length_penalty_start),
            float(length_penalty_max),
        )
        truncated = token_count is not None and token_count >= cap

        rewards.append(reward)
        diagnostics.append(
            {
                "reward": reward,
                "format_ok": has_format,
                "correct": result.ok,
                "token_count": token_count,
                "truncated": truncated,
                "completion": text,
                "error": result.error,
            }
        )

    _REWARD_DIAGNOSTICS.extend(diagnostics)

    if callable(log_metric) and diagnostics:
        row = build_legacy_metric_row(step=None, diagnostics=diagnostics, group_size=rollout_count)
        for key in ("reward_mean", "accuracy", "format_rate", "truncated_rate"):
            log_metric(f"countdown/{key}", row[key])
    if callable(log_extra) and diagnostics:
        # TRL >=1.6 signature: log_extra(column: str, values: list) -- one call
        # per column, values batched across the whole rollout group. Earlier
        # versions accepted a single diagnostic dict per call.
        import inspect as _inspect

        try:
            _extra_params = _inspect.signature(log_extra).parameters
        except (TypeError, ValueError):
            _extra_params = {}
        if "column" in _extra_params and "values" in _extra_params:
            for column in ("reward", "format_ok", "correct", "token_count", "truncated", "error"):
                log_extra(column, [d.get(column) for d in diagnostics])
        else:
            for diagnostic in diagnostics:
                log_extra(diagnostic)

    return rewards


def build_countdown_reward_func(cfg: dict[str, Any]):
    def reward_func(prompts=None, completions=None, *, numbers=None, target=None, completion_ids=None, **kwargs):
        if completions is None:
            raise ValueError("TRL reward call must provide completions")
        if numbers is None or target is None:
            raise ValueError("Countdown reward requires numbers and target dataset columns")
        return countdown_reward_func(
            completions=completions,
            numbers=numbers,
            target=target,
            completion_ids=completion_ids,
            format_reward=float(cfg.get("format_reward", 0.5)),
            answer_reward=float(cfg.get("answer_reward", 1.0)),
            max_completion_length=int(cfg.get("max_new_tokens", 1024)),
            length_penalty_start=int(cfg.get("length_penalty_start", 800)),
            length_penalty_max=float(cfg.get("length_penalty_max", -0.5)),
            **kwargs,
        )

    reward_func.__name__ = "countdown_reward_func"
    return reward_func


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 12) if values else 0.0


def build_legacy_metric_row(
    step: int | None,
    diagnostics: list[dict[str, Any]],
    group_size: int,
    logs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    logs = logs or {}
    rewards = [float(row.get("reward", 0.0)) for row in diagnostics]
    rollout_count = len(diagnostics)
    token_counts = [
        int(row["token_count"])
        for row in diagnostics
        if row.get("token_count") is not None
    ]
    summary = grpo_metric_summary(rewards, group_size) if rewards else grpo_metric_summary([], group_size)

    row: dict[str, Any] = {
        "step": step,
        "reward_mean": _mean(rewards),
        "mean_reward": _mean(rewards),
        **summary,
        "accuracy": _mean([1.0 if item.get("correct") else 0.0 for item in diagnostics]),
        "format_rate": _mean([1.0 if item.get("format_ok") else 0.0 for item in diagnostics]),
        "truncated_rate": _mean([1.0 if item.get("truncated") else 0.0 for item in diagnostics]),
        "truncated_count": sum(1 for item in diagnostics if item.get("truncated")),
        "avg_gen_tokens": _mean([float(value) for value in token_counts]),
        "max_gen_tokens": max(token_counts) if token_counts else 0,
        "rollout_count": rollout_count,
    }
    for key in ("loss", "learning_rate", "grad_norm", "kl", "entropy", "reward", "completion_length"):
        if key in logs and isinstance(logs[key], (int, float)):
            row[key] = float(logs[key])
    return row


def build_trl_grpo_wandb_metrics(metric: dict[str, Any]) -> dict[str, float | int]:
    return prefixed_metrics("train", {key: value for key, value in metric.items() if key != "step"})


def build_trl_metrics_callback(
    output_dir: Path,
    group_size: int,
    wandb_enabled: bool = False,
):
    from transformers import TrainerCallback

    class CountdownGRPOMetricsCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if hasattr(state, "is_world_process_zero") and not state.is_world_process_zero:
                return control
            diagnostics = consume_reward_diagnostics()
            metric = build_legacy_metric_row(
                step=int(state.global_step),
                diagnostics=diagnostics,
                group_size=group_size,
                logs=logs or {},
            )
            if not diagnostics and not (logs or {}):
                return control

            _append_jsonl(output_dir / METRICS_FILENAME, metric)
            if wandb_enabled:
                try:
                    import wandb
                except ImportError:
                    wandb = None
                if wandb is not None and getattr(wandb, "run", None) is not None:
                    wandb.log(build_trl_grpo_wandb_metrics(metric), step=int(state.global_step))
            return control

    return CountdownGRPOMetricsCallback()


def _add_standard_clip_arg(config_cls, kwargs: dict[str, Any], cfg: dict[str, Any]) -> None:
    if cfg.get("clip_eps") is None:
        return
    name = _first_supported_name(config_cls, ("epsilon", "clip_range", "cliprange", "clip_eps"))
    if name is None:
        raise ValueError(
            "clip_eps is configured, but the installed TRL GRPOConfig does not expose "
            "a known policy-ratio clipping parameter (expected one of: epsilon, "
            "clip_range, cliprange, clip_eps)."
        )
    kwargs[name] = float(cfg["clip_eps"])


def _add_optional_alias(
    config_cls,
    kwargs: dict[str, Any],
    names: tuple[str, ...],
    value: Any,
) -> None:
    if value is None:
        return
    name = _first_supported_name(config_cls, names)
    if name is not None:
        kwargs[name] = value


def build_trl_grpo_config(
    cfg: dict[str, Any],
    output_dir: str | Path,
    max_steps: int,
    config_cls=None,
):
    if config_cls is None:
        try:
            from trl import GRPOConfig
        except ImportError as exc:
            raise ImportError("TRL GRPO training requires the 'trl' package with GRPOConfig") from exc
        config_cls = GRPOConfig

    configure_wandb_env(cfg)
    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "overwrite_output_dir": False,
        "max_steps": int(max_steps),
        "num_train_epochs": float(cfg.get("epochs", 1.0)),
        "per_device_train_batch_size": int(cfg["batch_size"]),
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
        "max_prompt_length": int(cfg["max_prompt_len"]),
        "max_completion_length": int(cfg["max_new_tokens"]),
        "num_generations": int(cfg["group_size"]),
        "generation_batch_size": int(
            cfg.get("generation_batch_size", int(cfg["batch_size"]) * int(cfg["group_size"]))
        ),
        "temperature": float(cfg.get("temperature", 1.0)),
        "top_p": float(cfg.get("top_p", 1.0)),
        "use_vllm": bool(cfg.get("use_vllm", True)),
        "vllm_mode": str(cfg.get("vllm_mode", "colocate")),
        "vllm_gpu_memory_utilization": float(cfg["rollout_gpu_memory_utilization"])
        if cfg.get("rollout_gpu_memory_utilization") is not None
        else None,
        "vllm_tensor_parallel_size": int(cfg.get("vllm_tensor_parallel_size", cfg.get("tensor_parallel_size", 1))),
        "chat_template_kwargs": {"enable_thinking": bool(cfg.get("enable_thinking", False))},
        "beta": float(cfg.get("kl_coeff", 0.0)),
    }
    warmup_ratio = float(cfg.get("warmup_ratio", 0.0))
    if _signature_accepts(inspect.signature(config_cls), "warmup_steps"):
        kwargs["warmup_steps"] = int(math.ceil(int(max_steps) * warmup_ratio))
    else:
        kwargs["warmup_ratio"] = warmup_ratio
    _add_optional_alias(
        config_cls,
        kwargs,
        ("vllm_max_model_length", "vllm_max_model_len"),
        int(cfg["rollout_max_model_len"]) if cfg.get("rollout_max_model_len") is not None else None,
    )
    _add_standard_clip_arg(config_cls, kwargs, cfg)
    return config_cls(**_filter_kwargs(config_cls, kwargs))


def build_trl_grpo_trainer(model, tokenizer, training_args, train_dataset, callbacks, cfg: dict[str, Any]):
    try:
        from trl import GRPOTrainer
    except ImportError as exc:
        raise ImportError("TRL GRPO training requires TRL with GRPOTrainer available") from exc

    reward_func = build_countdown_reward_func(cfg)
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "reward_funcs": reward_func,
        "args": training_args,
        "train_dataset": train_dataset,
        "callbacks": callbacks,
    }
    trainer_signature = inspect.signature(GRPOTrainer)
    if _signature_accepts(trainer_signature, "processing_class"):
        trainer_kwargs["processing_class"] = tokenizer
    elif _signature_accepts(trainer_signature, "tokenizer"):
        trainer_kwargs["tokenizer"] = tokenizer
    return GRPOTrainer(**trainer_kwargs)


def run_grpo_trl_training(cfg: dict[str, Any], max_steps: int | None = None) -> None:
    model_path = resolve_path(cfg["model_path"], REPO_ROOT)
    train_data_path = resolve_path(cfg.get("train_data", DEFAULT_TRAIN_DATA), REPO_ROOT)
    output_dir = resolve_path(cfg["output_dir"], REPO_ROOT)
    effective_max_steps = int(max_steps if max_steps is not None else cfg["max_steps"])

    model, tokenizer = load_model_and_tokenizer(
        model_path,
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", False)),
    )
    rows = read_jsonl(train_data_path)
    train_dataset = build_grpo_dataset(rows)
    training_args = build_trl_grpo_config(cfg, output_dir, effective_max_steps)

    callbacks = [
        build_trl_metrics_callback(
            output_dir,
            group_size=int(cfg["group_size"]),
            wandb_enabled=is_wandb_enabled(cfg),
        )
    ]
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

    trainer = build_trl_grpo_trainer(model, tokenizer, training_args, train_dataset, callbacks, cfg)
    trainer.train()

    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(final_dir)


def main() -> None:
    args = parse_args()
    cfg_path = resolve_path(args.config, REPO_ROOT)
    cfg = load_yaml_config(cfg_path)
    run_grpo_trl_training(cfg, max_steps=args.max_steps)


if __name__ == "__main__":
    main()


