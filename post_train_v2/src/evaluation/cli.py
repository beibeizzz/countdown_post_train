"""Evaluation orchestration shared by the command-line entrypoint."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from post_train_v2.src.artifacts.atomic import publish_json, publish_jsonl
from post_train_v2.src.artifacts.hashing import (
    sha256_canonical_json,
    sha256_file,
)
from post_train_v2.src.artifacts.manifest import (
    ArtifactFile,
    ManifestV2,
    ParentArtifact,
    load_manifest,
    publish_manifest,
)
from post_train_v2.src.artifacts.locking import exclusive_output_lock
from post_train_v2.src.config.loading import load_yaml, require_keys, resolve_repo_path
from post_train_v2.src.data.schema import validate_normalized_source
from post_train_v2.src.data.splits import read_jsonl_strict
from post_train_v2.src.evaluation.generation import evaluate_rows
from post_train_v2.src.evaluation.model_loading import (
    adapter_base_model_path,
    load_model_and_tokenizer,
)
from post_train_v2.src.evaluation.scoring import aggregate_rows


CONFIG_KEYS = {
    "eval_data",
    "eval_manifest",
    "output_dir",
    "max_new_tokens",
    "enable_thinking",
    "do_sample",
}
SOURCE_SCHEMA = {
    "id": "string",
    "source_index": "integer",
    "numbers": "array[integer]",
    "target": "integer",
    "gold_expr": "string",
    "prompt": "string",
    "bucket": "object",
}
SAMPLES_SCHEMA = {
    "id": "string",
    "prompt": "string",
    "raw_generation": "string",
    "extracted_expr": "string|null",
    "format_ok": "boolean",
    "valid_expression": "boolean",
    "correct": "boolean",
    "error": "string|null",
    "value": "string|null",
    "generated_tokens": "integer",
    "truncated": "boolean",
}
METRICS_SCHEMA = {"metrics": "object"}


def _model_path_identity(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    if not root.exists():
        return {"identifier": str(path)}
    files = [root] if root.is_file() else sorted(
        item for item in root.rglob("*") if item.is_file()
    )
    payload = []
    for file_path in files:
        relative = file_path.name if root.is_file() else file_path.relative_to(root).as_posix()
        payload.append(
            {
                "path": relative,
                "size": file_path.stat().st_size,
                "sha256": sha256_file(file_path),
            }
        )
    return {"files": payload}


def fingerprint_model_path(path: str | Path) -> str:
    return sha256_canonical_json(_model_path_identity(path))


def fingerprint_evaluation_model(
    model_path: str | Path,
    base_model_path: str | Path | None = None,
) -> str:
    model_root = Path(model_path)
    adapter_config = model_root / "adapter_config.json"
    resolved_base = base_model_path
    if adapter_config.is_file() and resolved_base is None:
        configured_base = adapter_base_model_path(adapter_config)
        if configured_base is not None:
            resolved_base = _resolve_base_model_reference(configured_base)
    payload: dict[str, Any] = {
        "model": _model_path_identity(model_root),
    }
    if adapter_config.is_file():
        if resolved_base is None:
            raise ValueError(
                "LoRA adapter fingerprint requires a base model path"
            )
        payload["base_model"] = _model_path_identity(resolved_base)
    return sha256_canonical_json(payload)


def _resolve_base_model_reference(value: str | Path) -> str | Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate
    local_candidate = resolve_repo_path(candidate)
    return local_candidate if local_candidate.exists() else str(value)


def _manifest_file(manifest: ManifestV2, path: Path) -> ArtifactFile:
    matches = [item for item in manifest.files if item.relative_path == path.name]
    if len(matches) != 1:
        raise ValueError(f"eval manifest file entry not found for {path.name}")
    item = matches[0]
    if item.sha256 != sha256_file(path):
        raise ValueError("eval data hash does not match manifest")
    if item.byte_size != path.stat().st_size:
        raise ValueError("eval data size does not match manifest")
    if item.field_schema != SOURCE_SCHEMA:
        raise ValueError("eval data schema does not match canonical source schema")
    return item


def _output_file(
    output_dir: Path,
    filename: str,
    *,
    row_count: int,
    schema: dict[str, str],
) -> ArtifactFile:
    path = output_dir / filename
    return ArtifactFile(
        filename,
        sha256_file(path),
        path.stat().st_size,
        row_count,
        schema,
    )


def run_evaluation(
    config_path: str | Path,
    model_path: str | Path,
    *,
    base_model_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    limit: int | None = None,
) -> ManifestV2:
    config = load_yaml(config_path)
    require_keys(config, *sorted(CONFIG_KEYS))
    if set(config) != CONFIG_KEYS:
        raise ValueError("evaluation config has unexpected keys")
    if config["enable_thinking"] is not False:
        raise ValueError("evaluation enable_thinking must be false")
    if config["do_sample"] is not False:
        raise ValueError("evaluation do_sample must be false")
    max_new_tokens = config["max_new_tokens"]
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 256:
        raise ValueError("evaluation max_new_tokens must be between 1 and 256")
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("limit must be a positive exact integer or None")

    eval_path = resolve_repo_path(config["eval_data"])
    input_manifest_path = resolve_repo_path(config["eval_manifest"])
    destination = resolve_repo_path(output_dir or config["output_dir"])
    logical_output = (
        str(output_dir) if output_dir is not None else config["output_dir"]
    )
    resolved_model_path = resolve_repo_path(model_path)
    resolved_base = (
        _resolve_base_model_reference(base_model_path)
        if base_model_path is not None
        else None
    )
    with exclusive_output_lock(
        destination,
        lock_name=".evaluate_model.lock",
        metadata={
            "config_path": str(resolve_repo_path(config_path)),
            "output_dir": str(destination),
            "model_path": str(resolved_model_path),
        },
    ):
        return _run_evaluation_locked(
            config,
            resolved_model_path,
            resolved_base,
            destination,
            logical_output,
            limit,
        )


def _run_evaluation_locked(
    config: Mapping[str, Any],
    resolved_model_path: Path,
    resolved_base: str | Path | None,
    destination: Path,
    logical_output: str,
    limit: int | None,
) -> ManifestV2:
    max_new_tokens = config["max_new_tokens"]
    eval_path = resolve_repo_path(config["eval_data"])
    input_manifest_path = resolve_repo_path(config["eval_manifest"])
    input_hashes = {
        eval_path: sha256_file(eval_path),
        input_manifest_path: sha256_file(input_manifest_path),
    }
    input_manifest = load_manifest(input_manifest_path)
    if input_manifest.stage != "build_validation_splits":
        raise ValueError("evaluation parent stage must be build_validation_splits")
    if input_manifest.stage_metadata.get("completed") is not True:
        raise ValueError("evaluation parent manifest must have completed=true")
    eval_file = _manifest_file(input_manifest, eval_path)
    rows = read_jsonl_strict(eval_path, validate_normalized_source)
    if eval_file.row_count != len(rows):
        raise ValueError("eval data row count does not match manifest")
    if limit is not None:
        rows = rows[:limit]

    model_fingerprint = fingerprint_evaluation_model(
        resolved_model_path,
        resolved_base,
    )
    tokenizer, model = load_model_and_tokenizer(
        resolved_model_path,
        base_model_path=resolved_base,
    )
    scored = evaluate_rows(
        rows,
        tokenizer,
        model,
        max_new_tokens=max_new_tokens,
    )
    metrics = aggregate_rows(scored)
    if fingerprint_evaluation_model(
        resolved_model_path,
        resolved_base,
    ) != model_fingerprint:
        raise ValueError("evaluation model changed during run")

    for path, digest in input_hashes.items():
        if sha256_file(path) != digest:
            raise ValueError(f"evaluation input changed during run: {path}")
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "manifest.json"
    manifest_path.unlink(missing_ok=True)
    publish_jsonl(destination / "samples.jsonl", scored)
    publish_json(destination / "metrics.json", metrics)

    logical_config = dict(config)
    logical_config["output_dir"] = logical_output
    if limit is not None:
        logical_config["limit"] = limit
    manifest = ManifestV2.build(
        artifact_type="evaluation",
        stage="evaluate_model",
        files=[
            _output_file(
                destination,
                "samples.jsonl",
                row_count=len(scored),
                schema=SAMPLES_SCHEMA,
            ),
            _output_file(
                destination,
                "metrics.json",
                row_count=1,
                schema=METRICS_SCHEMA,
            ),
        ],
        parents=[
            ParentArtifact(
                input_manifest.artifact_id,
                input_hashes[input_manifest_path],
            )
        ],
        config=logical_config,
        stage_metadata={
            "completed": True,
            "evaluated_count": len(scored),
            "input_count": eval_file.row_count,
            "limit": limit,
        },
        model_path=str(resolved_model_path),
        model_fingerprint=model_fingerprint,
        global_seed=input_manifest.global_seed,
        seed_derivation_version=input_manifest.seed_derivation_version,
    )
    for path, digest in input_hashes.items():
        if sha256_file(path) != digest:
            raise ValueError(f"evaluation input changed during run: {path}")
    if fingerprint_evaluation_model(
        resolved_model_path,
        resolved_base,
    ) != model_fingerprint:
        raise ValueError("evaluation model changed during run")
    publish_manifest(manifest_path, manifest)
    return manifest
