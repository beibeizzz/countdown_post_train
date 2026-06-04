from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from itertools import cycle
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from post_train.scripts.sft.train_full import apply_chat_template_compat, load_model_and_tokenizer
from post_train.src.countdown.config import load_yaml_config, resolve_path
from post_train.src.countdown.generation import GenerationConfig, VLLMGenerator
from post_train.src.countdown.io import read_jsonl, write_json, write_jsonl
from post_train.src.countdown.validation import extract_answer_text, validate_countdown_response
from post_train.src.countdown.wandb_utils import (
    finish_wandb,
    init_wandb_if_enabled,
    log_wandb_metrics,
    prefixed_metrics,
)


DEFAULT_CONFIG = "post_train/configs/grpo.yaml"
DEFAULT_EVAL_CONFIG = "post_train/configs/eval.yaml"
METRICS_FILENAME = "metrics.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal GRPO training for Countdown.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--max-steps", type=int, default=None)
    return parser.parse_args()


def compute_rewards(
    rows: list[dict],
    completions: list[str],
    format_reward: float,
    answer_reward: float,
) -> list[dict]:
    if len(rows) != len(completions):
        raise ValueError("rows and completions must have the same length")

    rewarded_rows: list[dict] = []
    for row, completion in zip(rows, completions, strict=True):
        result = validate_countdown_response(completion, row["numbers"], int(row["target"]))
        has_format = extract_answer_text(completion) is not None
        reward = (format_reward if has_format else 0.0) + (answer_reward if result.ok else 0.0)
        rewarded_rows.append(
            {
                **row,
                "completion": completion,
                "reward": reward,
                "format_ok": has_format,
                "correct": result.ok,
            }
        )
    return rewarded_rows


def grpo_metric_summary(rewards: list[float], group_size: int) -> dict:
    if group_size < 1:
        raise ValueError("group_size must be at least 1")
    if not rewards:
        return {
            "reward_std": 0.0,
            "group_reward_std": 0.0,
            "frac_reward_zero_std": 0.0,
        }

    full_groups = [
        rewards[index : index + group_size]
        for index in range(0, len(rewards), group_size)
        if len(rewards[index : index + group_size]) == group_size
    ]
    group_stds = [statistics.pstdev(group) if len(group) > 1 else 0.0 for group in full_groups]

    return {
        "reward_std": statistics.pstdev(rewards) if len(rewards) > 1 else 0.0,
        "group_reward_std": sum(group_stds) / len(group_stds) if group_stds else 0.0,
        "frac_reward_zero_std": (
            sum(1 for std in group_stds if std == 0.0) / len(group_stds) if group_stds else 0.0
        ),
    }


def build_metric_row(
    loss: float,
    rewarded_rows: list[dict[str, Any]],
    group_size: int,
    approx_kl: float,
    entropy: float | None,
    learning_rate: float,
) -> dict[str, Any]:
    rewards = [float(row["reward"]) for row in rewarded_rows]
    token_counts = [int(row.get("token_count") or 0) for row in rewarded_rows]
    summary = grpo_metric_summary(rewards, group_size)
    rollout_count = len(rewarded_rows)

    return {
        "loss": float(loss),
        "mean_reward": sum(rewards) / rollout_count if rollout_count else 0.0,
        **summary,
        "accuracy": sum(1 for row in rewarded_rows if row.get("correct")) / rollout_count
        if rollout_count
        else 0.0,
        "format_rate": sum(1 for row in rewarded_rows if row.get("format_ok")) / rollout_count
        if rollout_count
        else 0.0,
        "approx_kl": float(approx_kl),
        "entropy": entropy,
        "avg_gen_tokens": sum(token_counts) / rollout_count if rollout_count else 0.0,
        "max_gen_tokens": max(token_counts) if token_counts else 0,
        "truncated_count": sum(1 for row in rewarded_rows if row.get("truncated")),
        "rollout_count": rollout_count,
        "learning_rate": float(learning_rate),
    }


def validate_supported_grpo_config(cfg: dict[str, Any], effective_max_steps: int | None = None) -> None:
    kl_coeff = float(cfg.get("kl_coeff", 0.0))
    if kl_coeff != 0.0:
        raise ValueError(
            "Nonzero kl_coeff is not supported: reference KL is not implemented in this "
            "minimal GRPO script, and this project expects kl_coeff: 0.0"
        )

    for key in ("policy_updates_per_rollout", "batch_size", "group_size"):
        if key in cfg and int(cfg[key]) < 1:
            raise ValueError(f"{key} must be at least 1")

    max_steps = effective_max_steps if effective_max_steps is not None else cfg.get("max_steps")
    if max_steps is not None and int(max_steps) < 1:
        raise ValueError("max_steps must be at least 1")


def ensure_policy_examples_available(
    examples_with_advantages: list[tuple[dict[str, list[int]], float]],
    rollout_count: int,
) -> list[tuple[dict[str, list[int]], float]]:
    if not examples_with_advantages:
        raise RuntimeError(
            "No policy examples could be encoded from the rollout; "
            f"all {rollout_count} completions were empty or invalid for tokenization."
        )
    return examples_with_advantages


def group_relative_advantages(rewards: list[float], group_size: int, clip_eps: float) -> list[float]:
    if group_size < 1:
        raise ValueError("group_size must be at least 1")
    advantages: list[float] = []
    for index in range(0, len(rewards), group_size):
        group = rewards[index : index + group_size]
        mean_reward = sum(group) / len(group)
        std_reward = statistics.pstdev(group) if len(group) > 1 else 0.0
        for reward in group:
            advantage = 0.0 if std_reward == 0.0 else (reward - mean_reward) / std_reward
            advantages.append(max(-clip_eps, min(clip_eps, advantage)))
    return advantages


def batched(items: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def build_generation_config(cfg: dict[str, Any]) -> GenerationConfig:
    return GenerationConfig(
        max_new_tokens=int(cfg["max_new_tokens"]),
        temperature=float(cfg["temperature"]),
        top_p=float(cfg["top_p"]),
        enable_thinking=bool(cfg.get("enable_thinking", False)),
    )


def rollout_batch(
    generator: VLLMGenerator,
    source_rows: list[dict[str, Any]],
    generation_config: GenerationConfig,
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    group_size = int(cfg["group_size"])
    rollout_rows = [row for row in source_rows for _ in range(group_size)]
    metadata_rows = generator.generate_with_metadata(
        [str(row["prompt"]) for row in rollout_rows],
        generation_config,
    )
    completions = [str(metadata.get("text", "")).strip() for metadata in metadata_rows]
    rewarded_rows = compute_rewards(
        rollout_rows,
        completions,
        format_reward=float(cfg["format_reward"]),
        answer_reward=float(cfg["answer_reward"]),
    )

    for rewarded_row, metadata in zip(rewarded_rows, metadata_rows, strict=True):
        finish_reason = metadata.get("finish_reason")
        rewarded_row["token_count"] = metadata.get("token_count")
        rewarded_row["finish_reason"] = finish_reason
        rewarded_row["truncated"] = finish_reason == "length"
    return rewarded_rows


def encode_policy_example(
    tokenizer,
    prompt: str,
    completion: str,
    max_prompt_len: int,
    max_new_tokens: int,
    enable_thinking: bool,
) -> dict[str, list[int]] | None:
    prompt_text = apply_chat_template_compat(
        tokenizer,
        [{"role": "user", "content": prompt}],
        enable_thinking=enable_thinking,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"][-max_prompt_len:]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"][:max_new_tokens]
    if not completion_ids:
        return None

    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def collate_policy_examples(examples: list[dict[str, list[int]]], pad_token_id: int):
    import torch

    max_len = max(len(example["input_ids"]) for example in examples)
    input_ids: list[list[int]] = []
    attention_mask: list[list[int]] = []
    labels: list[list[int]] = []
    for example in examples:
        pad_len = max_len - len(example["input_ids"])
        input_ids.append(example["input_ids"] + [pad_token_id] * pad_len)
        attention_mask.append(example["attention_mask"] + [0] * pad_len)
        labels.append(example["labels"] + [-100] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def sequence_policy_loss(model, batch: dict[str, Any], advantages, compute_entropy: bool):
    import torch
    import torch.nn.functional as F

    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = outputs.logits[:, :-1, :]
    labels = batch["labels"][:, 1:]
    response_mask = labels.ne(-100)
    safe_labels = labels.masked_fill(~response_mask, 0)

    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    response_token_counts = response_mask.sum(dim=1).clamp_min(1)
    sequence_log_probs = (token_log_probs * response_mask).sum(dim=1) / response_token_counts

    advantage_tensor = torch.tensor(advantages, dtype=sequence_log_probs.dtype, device=sequence_log_probs.device)
    loss = -(sequence_log_probs * advantage_tensor).mean()

    entropy = None
    if compute_entropy:
        probs = log_probs.exp()
        token_entropy = -(probs * log_probs).sum(dim=-1)
        entropy = ((token_entropy * response_mask).sum() / response_mask.sum().clamp_min(1)).detach()
    return loss, entropy


def save_checkpoint(model, tokenizer, output_dir: Path, step: int) -> Path:
    checkpoint_dir = output_dir / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    return checkpoint_dir


def sync_rollout_model(
    generator: VLLMGenerator,
    model,
    tokenizer,
    output_dir: Path,
    step: int,
    tensor_parallel_size: int = 1,
) -> VLLMGenerator:
    checkpoint_dir = save_checkpoint(model, tokenizer, output_dir / "rollout_sync", step)
    note_path = checkpoint_dir / "README.txt"
    note_path.write_text(
        "This checkpoint was written for rollout synchronization. "
        "Live vLLM weight sync is environment-dependent; this script attempts a reload "
        "from the saved checkpoint path when practical.\n",
        encoding="utf-8",
    )
    try:
        reloaded_generator = VLLMGenerator(str(checkpoint_dir), tensor_parallel_size=tensor_parallel_size)
    except Exception:
        return generator
    return reloaded_generator


def write_metric(output_dir: Path, metric: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / METRICS_FILENAME).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metric, sort_keys=True) + "\n")


def build_grpo_wandb_metrics(metric: dict[str, Any]) -> dict[str, float | int]:
    return prefixed_metrics(
        "train",
        {key: value for key, value in metric.items() if key != "step"},
    )


def build_grpo_eval_wandb_metrics(metrics: dict[str, Any]) -> dict[str, float | int]:
    return prefixed_metrics("eval", metrics)


def run_fixed_eval(
    model,
    tokenizer,
    output_dir: Path,
    step: int,
    eval_rows: list[dict[str, Any]],
    eval_cfg: dict[str, Any],
    wandb_run=None,
) -> None:
    from post_train.scripts.eval.evaluate_model import evaluate_rows
    from post_train.src.countdown.eval import aggregate_eval_rows

    was_training = model.training
    model.eval()
    scored_rows = evaluate_rows(eval_rows, tokenizer, model, eval_cfg)
    metrics = aggregate_eval_rows(scored_rows)
    step_dir = output_dir / "eval" / f"step_{step}"
    write_jsonl(step_dir / "eval_samples.jsonl", scored_rows)
    write_json(step_dir / "eval_metrics.json", metrics)
    log_wandb_metrics(wandb_run, build_grpo_eval_wandb_metrics(metrics), step=step)
    if was_training:
        model.train()


def build_optimizer_and_scheduler(model, cfg: dict[str, Any], max_steps: int):
    from torch.optim import AdamW
    from transformers import get_cosine_schedule_with_warmup

    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    warmup_steps = int(math.ceil(max_steps * float(cfg["warmup_ratio"])))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )
    return optimizer, scheduler


def train_grpo(cfg: dict[str, Any], max_steps: int, model_path: Path, output_dir: Path) -> None:
    import torch

    validate_supported_grpo_config(cfg, effective_max_steps=max_steps)
    output_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = init_wandb_if_enabled(cfg, default_name="grpo")
    model, tokenizer = load_model_and_tokenizer(
        model_path,
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", False)),
    )
    if bool(cfg.get("bf16", False)):
        model = model.to(dtype=torch.bfloat16)
    if torch.cuda.is_available():
        model = model.cuda()
    model.train()

    rollout_generator = VLLMGenerator(
        str(model_path),
        tensor_parallel_size=int(cfg.get("tensor_parallel_size", 1)),
    )
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg, max_steps)
    generation_config = build_generation_config(cfg)
    rows = read_jsonl(resolve_path(cfg["train_data"], REPO_ROOT))
    if not rows:
        raise ValueError("GRPO train_data is empty")

    row_iter = cycle(rows)
    global_step = 0
    batch_size = int(cfg["batch_size"])
    group_size = int(cfg["group_size"])
    updates_per_rollout = int(cfg["policy_updates_per_rollout"])
    sync_every_steps = int(cfg["sync_every_steps"])
    save_every_steps = int(cfg["save_every_steps"])
    enable_thinking = bool(cfg.get("enable_thinking", False))
    compute_entropy = bool(cfg.get("compute_entropy", False))
    eval_every_steps = int(cfg.get("eval_every_steps", 0))
    eval_rows: list[dict[str, Any]] = []
    eval_cfg: dict[str, Any] | None = None
    if eval_every_steps > 0:
        eval_cfg_path = resolve_path(DEFAULT_EVAL_CONFIG, REPO_ROOT)
        if eval_cfg_path.exists():
            eval_cfg = load_yaml_config(eval_cfg_path)
            eval_rows = read_jsonl(resolve_path(eval_cfg["eval_subset"], REPO_ROOT))

    try:
        while global_step < max_steps:
            source_rows = [next(row_iter) for _ in range(batch_size)]
            rewarded_rows = rollout_batch(rollout_generator, source_rows, generation_config, cfg)
            rewards = [float(row["reward"]) for row in rewarded_rows]
            advantages = group_relative_advantages(rewards, group_size, float(cfg["clip_eps"]))

            examples_with_advantages = []
            for row, advantage in zip(rewarded_rows, advantages, strict=True):
                encoded = encode_policy_example(
                    tokenizer,
                    str(row["prompt"]),
                    str(row["completion"]),
                    max_prompt_len=int(cfg["max_prompt_len"]),
                    max_new_tokens=int(cfg["max_new_tokens"]),
                    enable_thinking=enable_thinking,
                )
                if encoded is not None:
                    examples_with_advantages.append((encoded, advantage))
            examples_with_advantages = ensure_policy_examples_available(
                examples_with_advantages,
                rollout_count=len(rewarded_rows),
            )

            loss_value = 0.0
            entropy_value = None if compute_entropy else 0.0
            for _ in range(updates_per_rollout):
                if global_step >= max_steps:
                    break
                optimizer.zero_grad(set_to_none=True)
                batch = collate_policy_examples(
                    [example for example, _ in examples_with_advantages],
                    pad_token_id=int(tokenizer.pad_token_id),
                )
                batch = {key: value.to(model.device) for key, value in batch.items()}
                batch_advantages = [advantage for _, advantage in examples_with_advantages]
                loss, entropy = sequence_policy_loss(model, batch, batch_advantages, compute_entropy)
                loss.backward()
                optimizer.step()
                scheduler.step()

                global_step += 1
                loss_value = float(loss.detach().float().cpu().item())
                if entropy is not None:
                    entropy_value = float(entropy.float().cpu().item())

                metric = build_metric_row(
                    loss=loss_value,
                    rewarded_rows=rewarded_rows,
                    group_size=group_size,
                    approx_kl=0.0,
                    entropy=entropy_value,
                    learning_rate=scheduler.get_last_lr()[0],
                )
                metric["step"] = global_step
                write_metric(output_dir, metric)
                log_wandb_metrics(
                    wandb_run,
                    build_grpo_wandb_metrics(metric),
                    step=global_step,
                )

                if save_every_steps > 0 and global_step % save_every_steps == 0:
                    save_checkpoint(model, tokenizer, output_dir, global_step)
                if eval_cfg is not None and eval_every_steps > 0 and global_step % eval_every_steps == 0:
                    run_fixed_eval(
                        model,
                        tokenizer,
                        output_dir,
                        global_step,
                        eval_rows,
                        eval_cfg,
                        wandb_run=wandb_run,
                    )
                if sync_every_steps > 0 and global_step % sync_every_steps == 0:
                    rollout_generator = sync_rollout_model(
                        rollout_generator,
                        model,
                        tokenizer,
                        output_dir,
                        global_step,
                        tensor_parallel_size=int(cfg.get("tensor_parallel_size", 1)),
                    )

        final_dir = output_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
    finally:
        finish_wandb(wandb_run)


def main() -> None:
    args = parse_args()
    cfg_path = resolve_path(args.config, REPO_ROOT)
    cfg = load_yaml_config(cfg_path)
    model_path = resolve_path(cfg["model_path"], REPO_ROOT)
    output_dir = resolve_path(cfg["output_dir"], REPO_ROOT)
    max_steps = args.max_steps if args.max_steps is not None else int(cfg["max_steps"])
    validate_supported_grpo_config(cfg, effective_max_steps=max_steps)
    train_grpo(cfg, max_steps=max_steps, model_path=model_path, output_dir=output_dir)


if __name__ == "__main__":
    main()
