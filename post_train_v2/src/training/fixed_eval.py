"""Fixed-set evaluation callback for supervised DDP stages."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from post_train_v2.src.artifacts.atomic import publish_json, publish_jsonl
from post_train_v2.src.distributed.runtime import main_rank_section
from post_train_v2.src.evaluation.generation import evaluate_rows
from post_train_v2.src.evaluation.scoring import aggregate_rows


class FixedEvaluationCallback:
    def __init__(
        self,
        *,
        eval_rows: Sequence[Mapping[str, Any]],
        tokenizer,
        output_dir: str | Path,
        eval_every_steps: int = 100,
        max_new_tokens: int = 256,
        accelerator=None,
    ) -> None:
        if type(eval_every_steps) is not int or eval_every_steps <= 0:
            raise ValueError("eval_every_steps must be a positive integer")
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 256:
            raise ValueError("max_new_tokens must be between 1 and 256")
        self.eval_rows = list(eval_rows)
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.eval_every_steps = eval_every_steps
        self.max_new_tokens = max_new_tokens
        self.accelerator = accelerator

    def on_step_end(self, args, state, control, **kwargs):
        step = int(getattr(state, "global_step", 0) or 0)
        max_steps = int(getattr(state, "max_steps", 0) or 0)
        if not _should_evaluate(step, max_steps, self.eval_every_steps):
            return control

        model = kwargs.get("model")
        accelerator = kwargs.get("accelerator", self.accelerator)

        def evaluate_once() -> dict[str, Any]:
            if model is None:
                raise ValueError("FixedEvaluationCallback requires model")
            return self._evaluate(step, model, accelerator)

        main_rank_section(evaluate_once)
        return control

    def _evaluate(self, step: int, model, accelerator) -> dict[str, Any]:
        evaluated_model = (
            accelerator.unwrap_model(model)
            if accelerator is not None and hasattr(accelerator, "unwrap_model")
            else model
        )
        was_training = bool(getattr(evaluated_model, "training", False))
        if hasattr(evaluated_model, "eval"):
            evaluated_model.eval()
        try:
            samples = evaluate_rows(
                self.eval_rows,
                self.tokenizer,
                evaluated_model,
                max_new_tokens=self.max_new_tokens,
            )
            metrics = aggregate_rows(samples)
            self._publish(step, samples, metrics)
            return {"step": step, **metrics}
        finally:
            if was_training and hasattr(evaluated_model, "train"):
                evaluated_model.train()

    def _publish(
        self,
        step: int,
        samples: Sequence[Mapping[str, Any]],
        metrics: Mapping[str, Any],
    ) -> None:
        step_dir = self.output_dir / f"step_{step}"
        publish_jsonl(step_dir / "samples.jsonl", samples)
        publish_json(step_dir / "metrics.json", dict(metrics))
        ledger_path = self.output_dir / "ledger.jsonl"
        rows = _read_jsonl(ledger_path)
        rows.append(
            {
                "step": step,
                "metrics": dict(metrics),
                "samples": f"step_{step}/samples.jsonl",
                "metrics_file": f"step_{step}/metrics.json",
            }
        )
        publish_jsonl(ledger_path, rows)


def _should_evaluate(step: int, max_steps: int, every: int) -> bool:
    if step <= 0:
        return False
    if step % every == 0:
        return True
    return max_steps > 0 and step == max_steps


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows
