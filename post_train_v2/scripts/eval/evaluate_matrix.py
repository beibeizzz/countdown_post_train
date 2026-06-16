from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml


def _find_repo_root(script_path: Path) -> Path:
    for parent in script_path.resolve().parents:
        if (parent / "post_train_v2").is_dir() and (parent / ".git").exists():
            return parent
    raise RuntimeError(f"could not locate repository root from {script_path}")


REPO_ROOT = _find_repo_root(Path(__file__))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from post_train_v2.src.artifacts.atomic import publish_json
from post_train_v2.src.evaluation.cli import run_evaluation

Evaluator = Callable[..., object]


def evaluate_matrix(
    config_path: str | Path,
    *,
    evaluator: Evaluator = run_evaluation,
) -> dict[str, Any]:
    config = _load_config(Path(config_path))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    models = _ordered_mapping(config["models"])
    datasets = _ordered_mapping(config["datasets"])
    results: list[dict[str, Any]] = []

    for model in config["models"]:
        for dataset in config["datasets"]:
            results.append(
                _evaluate_one(
                    model=model,
                    dataset=dataset,
                    matrix_output_dir=output_dir,
                    max_new_tokens=int(config["max_new_tokens"]),
                    evaluator=evaluator,
                )
            )

    summary = {
        "models": models,
        "datasets": datasets,
        "results": results,
        "ranking": dict(config["ranking"]),
    }
    publish_json(output_dir / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the final V2 model matrix.")
    parser.add_argument("--config", required=True)
    return parser


def main(argv: Sequence[str] | None = None, *, evaluator: Evaluator = run_evaluation) -> int:
    args = build_parser().parse_args(argv)
    summary = evaluate_matrix(args.config, evaluator=evaluator)
    return 1 if any(item["status"] == "failed" for item in summary["results"]) else 0


def _evaluate_one(
    *,
    model: Mapping[str, Any],
    dataset: Mapping[str, Any],
    matrix_output_dir: Path,
    max_new_tokens: int,
    evaluator: Evaluator,
) -> dict[str, Any]:
    model_name = str(model["name"])
    dataset_name = str(dataset["name"])
    output_dir = matrix_output_dir / model_name / dataset_name
    eval_config = {
        "eval_data": dataset["eval_data"],
        "eval_manifest": dataset["eval_manifest"],
        "output_dir": str(output_dir),
        "max_new_tokens": max_new_tokens,
        "enable_thinking": False,
        "do_sample": False,
    }
    config_path = output_dir / "eval_config.yaml"
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(eval_config, sort_keys=False), encoding="utf-8")

    result = {
        "model": model_name,
        "dataset": dataset_name,
        "status": "complete",
        "output_dir": str(output_dir),
        "metrics": {},
    }
    try:
        evaluator(
            config_path,
            model["path"],
            base_model_path=model.get("base_model_path"),
            output_dir=output_dir,
            limit=model.get("limit"),
        )
        metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
        if not isinstance(metrics, dict):
            raise ValueError("metrics.json must contain an object")
        result["metrics"] = metrics
    except Exception as error:
        result["status"] = "failed"
        result["error"] = str(error)
    return result


def _load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("evaluation matrix config must be a mapping")
    for key in ("output_dir", "max_new_tokens", "datasets", "models", "ranking"):
        if key not in value:
            raise ValueError(f"evaluation matrix config missing {key}")
    if not isinstance(value["datasets"], list) or not value["datasets"]:
        raise ValueError("evaluation matrix datasets must be a non-empty list")
    if not isinstance(value["models"], list) or not value["models"]:
        raise ValueError("evaluation matrix models must be a non-empty list")
    return value


def _ordered_mapping(items: list[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(item["name"]): dict(item) for item in items}


if __name__ == "__main__":
    raise SystemExit(main())
