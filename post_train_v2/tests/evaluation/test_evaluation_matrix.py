from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from post_train_v2.scripts.eval.evaluate_matrix import evaluate_matrix, main


def test_evaluation_matrix_runs_expected_models_on_requested_datasets(
    tmp_path: Path,
) -> None:
    config_path = _write_matrix_config(tmp_path)
    calls: list[dict] = []

    def evaluator(config_path, model_path, *, base_model_path=None, output_dir=None, limit=None):
        calls.append(
            {
                "config_path": Path(config_path),
                "model_path": model_path,
                "base_model_path": base_model_path,
                "output_dir": Path(output_dir),
                "limit": limit,
            }
        )
        metrics_path = Path(output_dir) / "metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps({"accuracy": 0.5, "format_rate": 1.0}),
            encoding="utf-8",
        )
        return object()

    summary = evaluate_matrix(config_path, evaluator=evaluator)

    assert [model["name"] for model in summary["models"].values()] == [
        "base",
        "full_sft_best",
        "lora_best",
        "rft_best",
        "dpo_best",
        "grpo_best",
        "teacher",
    ]
    assert {(call["model_path"], call["output_dir"].name) for call in calls} == {
        ("models/base", "val_200"),
        ("models/base", "solved_test"),
        ("outputs/sft/full/best", "val_200"),
        ("outputs/sft/full/best", "solved_test"),
        ("outputs/sft/lora/best", "val_200"),
        ("outputs/sft/lora/best", "solved_test"),
        ("outputs/sft/rft/best", "val_200"),
        ("outputs/sft/rft/best", "solved_test"),
        ("outputs/dpo/best", "val_200"),
        ("outputs/dpo/best", "solved_test"),
        ("outputs/grpo/run/export/best", "val_200"),
        ("outputs/grpo/run/export/best", "solved_test"),
        ("models/teacher", "val_200"),
        ("models/teacher", "solved_test"),
    }
    lora_calls = [call for call in calls if call["model_path"] == "outputs/sft/lora/best"]
    assert {call["base_model_path"] for call in lora_calls} == {"models/base"}
    assert summary["ranking"] == {
        "primary": "test_accuracy",
        "secondary": "test_format_rate",
    }
    assert (tmp_path / "matrix" / "summary.json").is_file()


def test_evaluation_matrix_records_failures_and_cli_returns_nonzero(
    tmp_path: Path,
) -> None:
    config_path = _write_matrix_config(tmp_path, include_teacher=False)

    def evaluator(config_path, model_path, *, base_model_path=None, output_dir=None, limit=None):
        if model_path == "outputs/dpo/best":
            raise RuntimeError("load failed")
        metrics_path = Path(output_dir) / "metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps({"accuracy": 1.0, "format_rate": 1.0}),
            encoding="utf-8",
        )
        return object()

    summary = evaluate_matrix(config_path, evaluator=evaluator)

    failed = [item for item in summary["results"] if item["status"] == "failed"]
    assert failed
    assert failed[0]["model"] == "dpo_best"
    assert "load failed" in failed[0]["error"]
    assert main(["--config", str(config_path)], evaluator=evaluator) == 1


def _write_matrix_config(
    tmp_path: Path,
    *,
    include_teacher: bool = True,
) -> Path:
    models = [
        {"name": "base", "path": "models/base"},
        {"name": "full_sft_best", "path": "outputs/sft/full/best"},
        {
            "name": "lora_best",
            "path": "outputs/sft/lora/best",
            "base_model_path": "models/base",
        },
        {"name": "rft_best", "path": "outputs/sft/rft/best"},
        {"name": "dpo_best", "path": "outputs/dpo/best"},
        {"name": "grpo_best", "path": "outputs/grpo/run/export/best"},
    ]
    if include_teacher:
        models.append({"name": "teacher", "path": "models/teacher", "optional": True})
    config = {
        "output_dir": str(tmp_path / "matrix"),
        "max_new_tokens": 256,
        "datasets": [
            {
                "name": "val_200",
                "eval_data": "data/val_200.jsonl",
                "eval_manifest": "data/validation_manifest.json",
                "ranking_key": "validation",
            },
            {
                "name": "solved_test",
                "eval_data": "data/test_solved.jsonl",
                "eval_manifest": "data/source_manifest.json",
                "ranking_key": "test",
            },
        ],
        "models": models,
        "ranking": {"primary": "test_accuracy", "secondary": "test_format_rate"},
    }
    path = tmp_path / "final_matrix.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path
