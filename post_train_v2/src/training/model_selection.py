"""Model selection and export helpers for supervised stages."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
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
