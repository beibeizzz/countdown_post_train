"""Model selection and export helpers for supervised stages."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Any

from post_train_v2.src.artifacts.atomic import publish_json


@dataclass(frozen=True)
class EvaluationResult:
    step: int
    metrics: Mapping[str, Any]


def select_best(results: Sequence[EvaluationResult]) -> EvaluationResult:
    if not results:
        raise ValueError("at least one evaluation result is required")
    return min(
        results,
        key=lambda item: (
            -float(item.metrics.get("accuracy", 0.0)),
            -float(item.metrics.get("format_rate", 0.0)),
            item.step,
        ),
    )


def publish_model_export(
    *,
    model,
    tokenizer,
    output_dir: str | Path,
    export_name: str,
    export_kind: str,
    direct_load_check: Callable[[Path], None] | None = None,
) -> dict[str, Any]:
    if export_kind not in {"full_model", "lora_adapter"}:
        raise ValueError("export_kind must be full_model or lora_adapter")
    export_dir = Path(output_dir) / export_name
    export_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(export_dir)
    tokenizer.save_pretrained(export_dir)

    direct_loadable = export_kind == "full_model"
    if direct_loadable:
        if direct_load_check is None:
            raise ValueError("full model export requires direct_load_check")
        direct_load_check(export_dir)

    manifest = {
        "export_name": export_name,
        "export_kind": export_kind,
        "direct_loadable": direct_loadable,
    }
    publish_json(export_dir / "export_manifest.json", manifest)
    return manifest


def export_final_model(
    *,
    trainer,
    tokenizer,
    output_dir: str | Path,
    export_kind: str,
    direct_load_check: Callable[[Path], None] | None = None,
) -> dict[str, Any]:
    export_dir = Path(output_dir) / "final"
    export_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(trainer, "save_model"):
        trainer.save_model(export_dir)
    else:
        trainer.model.save_pretrained(export_dir)
    tokenizer.save_pretrained(export_dir)
    return _publish_export_manifest(
        export_dir,
        export_name="final",
        export_kind=export_kind,
        direct_load_check=direct_load_check,
    )


def export_best_checkpoint(
    *,
    tokenizer,
    output_dir: str | Path,
    export_kind: str,
    direct_load_check: Callable[[Path], None] | None = None,
) -> dict[str, Any] | None:
    output_dir = Path(output_dir)
    results = _read_eval_ledger(output_dir / "eval" / "ledger.jsonl")
    if not results:
        return None
    checkpoint = output_dir / f"checkpoint-{select_best(results).step}"
    if not checkpoint.exists():
        return None
    export_dir = output_dir / "best"
    if export_dir.exists():
        shutil.rmtree(export_dir)
    shutil.copytree(checkpoint, export_dir)
    tokenizer.save_pretrained(export_dir)
    return _publish_export_manifest(
        export_dir,
        export_name="best",
        export_kind=export_kind,
        direct_load_check=direct_load_check,
    )


def _read_eval_ledger(path: Path) -> list[EvaluationResult]:
    if not path.exists():
        return []
    results: list[EvaluationResult] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        results.append(
            EvaluationResult(
                step=int(row["step"]),
                metrics=dict(row["metrics"]),
            )
        )
    return results


def _publish_export_manifest(
    export_dir: Path,
    *,
    export_name: str,
    export_kind: str,
    direct_load_check: Callable[[Path], None] | None,
) -> dict[str, Any]:
    if export_kind not in {"full_model", "lora_adapter"}:
        raise ValueError("export_kind must be full_model or lora_adapter")
    direct_loadable = export_kind == "full_model"
    if direct_loadable:
        if direct_load_check is None:
            raise ValueError("full model export requires direct_load_check")
        direct_load_check(export_dir)
    manifest = {
        "export_name": export_name,
        "export_kind": export_kind,
        "direct_loadable": direct_loadable,
    }
    publish_json(export_dir / "export_manifest.json", manifest)
    return manifest
