"""Countdown JSONL to verl Parquet conversion."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

from post_train_v2.src.artifacts.atomic import publish_json
from post_train_v2.src.artifacts.hashing import sha256_file
from post_train_v2.src.artifacts.manifest import ArtifactFile, ManifestV2, publish_manifest
from post_train_v2.src.config.loading import resolve_repo_path
from post_train_v2.src.data.schema import validate_normalized_source, validate_verl_record
from post_train_v2.src.data.splits import read_jsonl_strict

VERL_PARQUET_SCHEMA = {
    "data_source": "string",
    "prompt": "array[object]",
    "ability": "string",
    "reward_model": "object",
    "extra_info": "object",
}


def source_to_verl_record(source: dict[str, Any]) -> dict[str, Any]:
    source = validate_normalized_source(source)
    record = {
        "data_source": "countdown",
        "prompt": [{"role": "user", "content": source["prompt"]}],
        "ability": "arithmetic",
        "reward_model": {
            "style": "rule",
            "ground_truth": {
                "numbers": source["numbers"],
                "target": source["target"],
            },
        },
        "extra_info": {
            "id": source["id"],
            "source_index": source["source_index"],
            "bucket": source["bucket"],
            "gold_expr": source["gold_expr"],
        },
    }
    return validate_verl_record(record)


def validate_unique_verl_ids(rows: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    for row in rows:
        row_id = row["id"]
        if row_id in seen:
            raise ValueError(f"duplicate id: {row_id}")
        seen.add(row_id)


def convert_source_rows(
    rows: list[dict[str, Any]],
    *,
    expected_count: int | None,
) -> list[dict[str, Any]]:
    validate_unique_verl_ids(rows)
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(
            f"row-count mismatch: expected {expected_count}, got {len(rows)}"
        )
    return [source_to_verl_record(row) for row in rows]


def convert_jsonl_to_parquet(
    *,
    input_jsonl: str | Path,
    output_parquet: str | Path,
    expected_count: int | None = None,
) -> ManifestV2:
    rows = read_jsonl_strict(resolve_repo_path(input_jsonl), validate_normalized_source)
    records = convert_source_rows(rows, expected_count=expected_count)
    output_path = resolve_repo_path(output_parquet)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = _records_to_arrow_table(records)
    pyarrow_parquet = import_module("pyarrow.parquet")
    pyarrow_parquet.write_table(table, output_path)
    reloaded = pyarrow_parquet.read_table(output_path)
    if reloaded.num_rows != len(records):
        raise ValueError("Parquet row-count mismatch after write")
    manifest = ManifestV2.build(
        artifact_type="verl_parquet",
        stage="jsonl_to_verl_parquet",
        files=(
            ArtifactFile(
                relative_path=output_path.name,
                sha256=sha256_file(output_path),
                byte_size=output_path.stat().st_size,
                row_count=len(records),
                field_schema=VERL_PARQUET_SCHEMA,
            ),
        ),
        parents=(),
        config={
            "input_jsonl": str(input_jsonl),
            "output_parquet": str(output_parquet),
            "expected_count": expected_count,
        },
        stage_metadata={"row_count": len(records)},
    )
    publish_manifest(output_path.with_suffix(".manifest.json"), manifest)
    publish_json(output_path.with_suffix(".schema.json"), VERL_PARQUET_SCHEMA)
    return manifest


def _records_to_arrow_table(records: list[dict[str, Any]]):
    pyarrow = import_module("pyarrow")
    return pyarrow.Table.from_pylist(records)
