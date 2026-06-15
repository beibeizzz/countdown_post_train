from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from post_train_v2.src.artifacts.atomic import publish_jsonl
from post_train_v2.src.artifacts.hashing import sha256_file
from post_train_v2.src.artifacts.manifest import (
    ArtifactFile,
    ManifestV2,
    ParentArtifact,
    load_manifest,
    publish_manifest,
)
from post_train_v2.src.countdown.bucketing import assign_bucket
from post_train_v2.src.countdown.prompts import build_solution_prompt
from post_train_v2.src.data import splits as splits_module
from post_train_v2.src.data.schema import validate_sft_record
from post_train_v2.src.data.splits import (
    derive_stage_seed,
    run_accepted_splits,
    run_validation_splits,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "post_train_v2/scripts/data/build_splits.py"
SOURCE_SCHEMA = {
    "id": "string",
    "source_index": "integer",
    "numbers": "array[integer]",
    "target": "integer",
    "gold_expr": "string",
    "prompt": "string",
    "bucket": "object",
}
SFT_SCHEMA = {
    **SOURCE_SCHEMA,
    "response": "string",
    "validation": "object",
    "provenance": "object",
}


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def source_row(index: int) -> dict:
    numbers = [1, 2, index + 3]
    target = sum(numbers)
    expression = f"{numbers[0]}+{numbers[1]}+{numbers[2]}"
    return {
        "id": f"train-{index:06d}",
        "source_index": index,
        "numbers": numbers,
        "target": target,
        "gold_expr": expression,
        "prompt": build_solution_prompt(numbers, target),
        "bucket": assign_bucket(numbers, expression),
    }


def sft_row(index: int) -> dict:
    source = source_row(index)
    expression = source["gold_expr"]
    response = f"<answer>{expression}</answer>"
    return validate_sft_record(
        {
            **source,
            "response": response,
            "validation": {
                "ok": True,
                "value": f"{source['target']}/1",
                "used_numbers": source["numbers"],
                "expression": expression,
                "error": None,
            },
            "provenance": {"teacher": "fixture"},
        }
    )


def artifact_file(path: Path, count: int, schema: dict) -> ArtifactFile:
    return ArtifactFile(
        relative_path=path.name,
        sha256=sha256_file(path),
        byte_size=path.stat().st_size,
        row_count=count,
        field_schema=schema,
    )


def write_source_fixture(
    tmp_path: Path,
    *,
    count: int = 12,
    rows: list[dict] | None = None,
    stage_metadata: dict | None = None,
) -> tuple[Path, Path, Path]:
    input_dir = tmp_path / "source"
    output_dir = tmp_path / "output"
    source_path = input_dir / "solvable_train.jsonl"
    manifest_path = input_dir / "manifest.json"
    selected = rows if rows is not None else [source_row(i) for i in range(count)]
    publish_jsonl(source_path, selected)
    manifest = ManifestV2.build(
        artifact_type="dataset",
        stage="build_source",
        files=[artifact_file(source_path, len(selected), SOURCE_SCHEMA)],
        parents=[],
        config={"seed": 1},
        global_seed=1,
        stage_metadata=stage_metadata
        or {
            "completed": True,
            "counts": {
                "train_input": len(selected),
                "solvable_train": len(selected),
                "unsolved_train": 0,
            },
            "limit": None,
        },
        git_revision="fixture",
        created_at="2026-06-15T00:00:00Z",
    )
    publish_manifest(manifest_path, manifest)
    return source_path, manifest_path, output_dir


def refresh_source_manifest_for_raw_file(
    source_path: Path,
    manifest_path: Path,
    *,
    row_count: int,
) -> None:
    manifest = ManifestV2.build(
        artifact_type="dataset",
        stage="build_source",
        files=[artifact_file(source_path, row_count, SOURCE_SCHEMA)],
        parents=[],
        config={"seed": 1},
        global_seed=1,
        stage_metadata={
            "completed": True,
            "counts": {
                "train_input": row_count,
                "solvable_train": row_count,
                "unsolved_train": 0,
            },
            "limit": None,
        },
        git_revision="fixture",
        created_at="2026-06-15T00:00:00Z",
    )
    publish_manifest(manifest_path, manifest)


def rebuild_manifest(
    path: Path,
    *,
    stage_metadata: dict | None = None,
    files: list[ArtifactFile] | None = None,
    parents: list[ParentArtifact] | None = None,
) -> ManifestV2:
    original = load_manifest(path)
    rebuilt = ManifestV2.build(
        artifact_type=original.artifact_type,
        stage=original.stage,
        files=files if files is not None else original.files,
        parents=parents if parents is not None else original.parents,
        config=original.config,
        global_seed=original.global_seed,
        seed_derivation_version=original.seed_derivation_version,
        git_revision=original.git_revision,
        runtime_versions=original.runtime_versions,
        stage_metadata=(
            stage_metadata
            if stage_metadata is not None
            else original.stage_metadata
        ),
        created_at=original.created_at,
    )
    publish_manifest(path, rebuilt)
    return rebuilt


def reparent_teacher_to_validation(config_path: Path, output_dir: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    teacher_manifest_path = Path(config["teacher_manifest"])
    validation_manifest_path = output_dir / "validation_manifest.json"
    validation_manifest = load_manifest(validation_manifest_path)
    rebuild_manifest(
        teacher_manifest_path,
        parents=[
            ParentArtifact(
                validation_manifest.artifact_id,
                sha256_file(validation_manifest_path),
            )
        ],
    )


def write_validation_config(
    tmp_path: Path,
    source_path: Path,
    source_manifest: Path,
    output_dir: Path,
    *,
    val_size: int = 6,
    eval_size: int = 2,
    seed: int = 42,
) -> Path:
    config_path = tmp_path / "configs" / "build_splits.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "seed": seed,
        "source_data": str(source_path),
        "source_manifest": str(source_manifest),
        "teacher_accepted": str(tmp_path / "teacher/teacher_accepted_20k.jsonl"),
        "teacher_manifest": str(tmp_path / "teacher/manifest.json"),
        "output_dir": str(output_dir),
        "val_size": val_size,
        "eval_size": eval_size,
        "sft_size": 5,
        "grpo_size": 3,
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def write_teacher_fixture(
    tmp_path: Path,
    *,
    rows: list[dict],
    completed: bool = True,
    stage: str = "teacher_accepted_pool",
    parents: list[ParentArtifact] | None = None,
    stage_metadata: dict | None = None,
) -> tuple[Path, Path]:
    teacher_dir = tmp_path / "teacher"
    accepted_path = teacher_dir / "teacher_accepted_20k.jsonl"
    manifest_path = teacher_dir / "manifest.json"
    publish_jsonl(accepted_path, rows)
    manifest = ManifestV2.build(
        artifact_type="dataset",
        stage=stage,
        files=[artifact_file(accepted_path, len(rows), SFT_SCHEMA)],
        parents=parents or [],
        config={"seed": 9},
        global_seed=9,
        stage_metadata=stage_metadata
        or {
            "completed": completed,
            "accepted_count": len(rows),
            "target_accepted_count": len(rows),
        },
        git_revision="fixture",
        created_at="2026-06-15T00:00:00Z",
    )
    publish_manifest(manifest_path, manifest)
    return accepted_path, manifest_path


def prepare_validation(tmp_path: Path, *, count: int = 12):
    source_path, source_manifest, output_dir = write_source_fixture(
        tmp_path, count=count
    )
    config_path = write_validation_config(
        tmp_path, source_path, source_manifest, output_dir
    )
    manifest = run_validation_splits(config_path)
    return config_path, output_dir, manifest


def prepare_accepted(tmp_path: Path):
    config_path, output_dir, validation_manifest = prepare_validation(
        tmp_path, count=14
    )
    val_ids = set(validation_manifest.stage_metadata["selected_ids"]["validation"])
    teacher_rows = [
        sft_row(i)
        for i in range(30, 42)
        if f"train-{i:06d}" not in val_ids
    ]
    accepted_path, teacher_manifest = write_teacher_fixture(
        tmp_path,
        rows=teacher_rows,
        parents=[
            ParentArtifact(
                validation_manifest.artifact_id,
                sha256_file(output_dir / "validation_manifest.json"),
            )
        ],
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["teacher_accepted"] = str(accepted_path)
    config["teacher_manifest"] = str(teacher_manifest)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path, output_dir, teacher_rows


def test_validation_is_deterministic_excludes_val_and_restores_source_order(
    tmp_path: Path,
):
    source_path, source_manifest, output_dir = write_source_fixture(tmp_path)
    config_path = write_validation_config(
        tmp_path, source_path, source_manifest, output_dir
    )

    first = run_validation_splits(config_path)
    first_bytes = {
        name: (output_dir / name).read_bytes()
        for name in (
            "val_200.jsonl",
            "eval_50.jsonl",
            "train_candidates.jsonl",
            "validation_manifest.json",
        )
    }
    second = run_validation_splits(config_path)

    val_rows = read_jsonl(output_dir / "val_200.jsonl")
    eval_rows = read_jsonl(output_dir / "eval_50.jsonl")
    train_rows = read_jsonl(output_dir / "train_candidates.jsonl")
    val_ids = {row["id"] for row in val_rows}
    assert {row["id"] for row in eval_rows} <= val_ids
    assert val_ids.isdisjoint(row["id"] for row in train_rows)
    assert [row["source_index"] for row in val_rows] == sorted(
        row["source_index"] for row in val_rows
    )
    assert [row["source_index"] for row in eval_rows] == sorted(
        row["source_index"] for row in eval_rows
    )
    assert [row["source_index"] for row in train_rows] == sorted(
        row["source_index"] for row in train_rows
    )
    assert first.artifact_id == second.artifact_id
    assert first_bytes == {
        name: (output_dir / name).read_bytes() for name in first_bytes
    }
    assert first.stage_metadata["selected_ids"]["validation"] != [
        row["id"] for row in val_rows
    ]
    assert first.stage_metadata["source_order_ids"]["validation"] == [
        row["id"] for row in val_rows
    ]


def test_validation_manifest_records_parent_files_counts_and_independent_seeds(
    tmp_path: Path,
):
    source_path, source_manifest_path, output_dir = write_source_fixture(tmp_path)
    config_path = write_validation_config(
        tmp_path, source_path, source_manifest_path, output_dir
    )

    manifest = run_validation_splits(config_path)
    source_manifest = load_manifest(source_manifest_path)

    assert manifest == load_manifest(output_dir / "validation_manifest.json")
    assert manifest.stage == "build_validation_splits"
    assert manifest.parents == (
        ParentArtifact(source_manifest.artifact_id, sha256_file(source_manifest_path)),
    )
    assert manifest.stage_metadata["completed"] is True
    assert manifest.stage_metadata["counts"] == {
        "source": 12,
        "validation": 6,
        "evaluation": 2,
        "train_candidates": 6,
    }
    seeds = manifest.stage_metadata["derived_seeds"]
    assert seeds == {
        "validation": derive_stage_seed(42, "validation"),
        "evaluation": derive_stage_seed(42, "evaluation"),
    }
    assert seeds["validation"] != seeds["evaluation"]
    assert [item.relative_path for item in manifest.files] == [
        "val_200.jsonl",
        "eval_50.jsonl",
        "train_candidates.jsonl",
    ]
    for item in manifest.files:
        path = output_dir / item.relative_path
        assert item.sha256 == sha256_file(path)
        assert item.byte_size == path.stat().st_size
        assert item.field_schema == SOURCE_SCHEMA


@pytest.mark.parametrize(
    ("stage_metadata", "match"),
    [
        (
            {
                "completed": True,
                "counts": {
                    "train_input": 12,
                    "solvable_train": 12,
                    "unsolved_train": 0,
                },
                "limit": 12,
            },
            "limit",
        ),
        (
            {
                "completed": True,
                "counts": {
                    "train_input": 12,
                    "solvable_train": 12,
                    "unsolved_train": 0,
                },
            },
            "limit",
        ),
        ({"completed": True, "limit": None}, "counts"),
        (
            {
                "completed": True,
                "counts": {
                    "train_input": 12,
                    "solvable_train": 12,
                },
                "limit": None,
            },
            "unsolved_train",
        ),
        (
            {
                "completed": True,
                "counts": {
                    "train_input": True,
                    "solvable_train": 12,
                    "unsolved_train": 0,
                },
                "limit": None,
            },
            "train_input",
        ),
        (
            {
                "completed": True,
                "counts": {
                    "train_input": 13,
                    "solvable_train": 12,
                    "unsolved_train": 0,
                },
                "limit": None,
            },
            "train_input",
        ),
        (
            {
                "completed": True,
                "counts": {
                    "train_input": 12,
                    "solvable_train": 11,
                    "unsolved_train": 1,
                },
                "limit": None,
            },
            "solvable_train",
        ),
    ],
)
def test_validation_requires_complete_production_source_manifest(
    tmp_path: Path, stage_metadata: dict, match: str
):
    source_path, source_manifest, output_dir = write_source_fixture(
        tmp_path, stage_metadata=stage_metadata
    )
    config_path = write_validation_config(
        tmp_path, source_path, source_manifest, output_dir
    )

    with pytest.raises(ValueError, match=match):
        run_validation_splits(config_path)

    assert not (output_dir / "validation_manifest.json").exists()


def test_accepted_samples_independently_with_separate_seeds_and_can_overlap(
    tmp_path: Path,
):
    config_path, output_dir, _ = prepare_accepted(tmp_path)

    first = run_accepted_splits(config_path)
    first_bytes = {
        name: (output_dir / name).read_bytes()
        for name in (
            "sft_train_8k.jsonl",
            "grpo_train_4k.jsonl",
            "accepted_splits_manifest.json",
        )
    }
    second = run_accepted_splits(config_path)

    sft_rows = read_jsonl(output_dir / "sft_train_8k.jsonl")
    grpo_rows = read_jsonl(output_dir / "grpo_train_4k.jsonl")
    assert len(sft_rows) == 5
    assert len(grpo_rows) == 3
    assert [row["source_index"] for row in sft_rows] == sorted(
        row["source_index"] for row in sft_rows
    )
    assert [row["source_index"] for row in grpo_rows] == sorted(
        row["source_index"] for row in grpo_rows
    )
    seeds = first.stage_metadata["derived_seeds"]
    assert seeds == {
        "sft": derive_stage_seed(42, "sft"),
        "grpo": derive_stage_seed(42, "grpo"),
    }
    assert seeds["sft"] != seeds["grpo"]
    assert first.stage_metadata["sampling"] == "independent_with_overlap_allowed"
    assert first.artifact_id == second.artifact_id
    assert first_bytes == {
        name: (output_dir / name).read_bytes() for name in first_bytes
    }


@pytest.mark.parametrize(
    ("metadata", "match"),
    [
        ({"accepted_count": 12, "target_accepted_count": 12}, "completed"),
        (
            {
                "completed": False,
                "accepted_count": 12,
                "target_accepted_count": 12,
            },
            "completed",
        ),
        (
            {
                "completed": 1,
                "accepted_count": 12,
                "target_accepted_count": 12,
            },
            "completed",
        ),
        (
            {
                "completed": True,
                "target_accepted_count": 12,
            },
            "accepted_count",
        ),
        (
            {
                "completed": True,
                "accepted_count": True,
                "target_accepted_count": 12,
            },
            "accepted_count",
        ),
        (
            {
                "completed": True,
                "accepted_count": 11,
                "target_accepted_count": 11,
            },
            "accepted_count",
        ),
        (
            {
                "completed": True,
                "accepted_count": 12,
                "target_accepted_count": True,
            },
            "target_accepted_count",
        ),
        (
            {
                "completed": True,
                "accepted_count": 12,
                "target_accepted_count": 13,
            },
            "target_accepted_count",
        ),
        (
            {
                "completed": True,
                "counts": {"accepted": 12},
            },
            "accepted_count",
        ),
    ],
)
def test_accepted_requires_strict_teacher_completion_contract(
    tmp_path: Path, metadata: dict, match: str
):
    config_path, output_dir, _ = prepare_accepted(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    rebuild_manifest(
        Path(config["teacher_manifest"]),
        stage_metadata=metadata,
    )

    with pytest.raises(ValueError, match=match):
        run_accepted_splits(config_path)

    assert not (output_dir / "accepted_splits_manifest.json").exists()


def test_accepted_allows_completed_pool_above_target_count(tmp_path: Path):
    config_path, output_dir, teacher_rows = prepare_accepted(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    rebuild_manifest(
        Path(config["teacher_manifest"]),
        stage_metadata={
            "completed": True,
            "accepted_count": len(teacher_rows),
            "target_accepted_count": len(teacher_rows) - 1,
        },
    )

    manifest = run_accepted_splits(config_path)

    assert manifest.stage_metadata["counts"]["accepted_pool"] == len(teacher_rows)
    assert (output_dir / "accepted_splits_manifest.json").exists()


@pytest.mark.parametrize(
    "mismatch", ["hash", "file_count", "selected_count", "metadata_count"]
)
def test_accepted_validates_validation_manifest_against_val_artifact_and_config(
    tmp_path: Path, mismatch: str
):
    config_path, output_dir, _ = prepare_accepted(tmp_path)
    validation_manifest_path = output_dir / "validation_manifest.json"
    val_path = output_dir / "val_200.jsonl"
    validation_manifest = load_manifest(validation_manifest_path)

    if mismatch == "hash":
        val_path.write_bytes(val_path.read_bytes() + b"\n")
    elif mismatch == "file_count":
        files = [
            ArtifactFile(
                relative_path=item.relative_path,
                sha256=item.sha256,
                byte_size=item.byte_size,
                row_count=(
                    item.row_count + 1
                    if item.relative_path == "val_200.jsonl"
                    else item.row_count
                ),
                field_schema=item.field_schema,
            )
            for item in validation_manifest.files
        ]
        rebuild_manifest(validation_manifest_path, files=files)
        reparent_teacher_to_validation(config_path, output_dir)
    elif mismatch == "selected_count":
        metadata = dict(validation_manifest.stage_metadata)
        metadata["selected_ids"] = dict(metadata["selected_ids"])
        metadata["selected_ids"]["validation"] = metadata["selected_ids"][
            "validation"
        ][:-1]
        rebuild_manifest(validation_manifest_path, stage_metadata=metadata)
        reparent_teacher_to_validation(config_path, output_dir)
    else:
        metadata = dict(validation_manifest.stage_metadata)
        metadata["counts"] = dict(metadata["counts"])
        metadata["counts"]["validation"] += 1
        rebuild_manifest(validation_manifest_path, stage_metadata=metadata)
        reparent_teacher_to_validation(config_path, output_dir)

    with pytest.raises(ValueError, match="validation|val_200|count|hash"):
        run_accepted_splits(config_path)

    assert not (output_dir / "accepted_splits_manifest.json").exists()


@pytest.mark.parametrize(
    ("field", "size", "split_name"),
    [
        ("sft_size", 99, "sft_train_8k"),
        ("grpo_size", 99, "grpo_train_4k"),
    ],
)
def test_accepted_shortfall_names_requested_split(
    tmp_path: Path, field: str, size: int, split_name: str
):
    config_path, output_dir, _ = prepare_accepted(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config[field] = size
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=split_name):
        run_accepted_splits(config_path)

    assert not (output_dir / "accepted_splits_manifest.json").exists()


@pytest.mark.parametrize(
    ("completed", "stage", "match"),
    [
        (False, "teacher_accepted_pool", "completed"),
        (True, "wrong_stage", "teacher_accepted_pool"),
    ],
)
def test_accepted_rejects_partial_or_wrong_stage_teacher_before_writing(
    tmp_path: Path, completed: bool, stage: str, match: str
):
    config_path, output_dir, validation_manifest = prepare_validation(tmp_path)
    val_ids = set(validation_manifest.stage_metadata["selected_ids"]["validation"])
    rows = [sft_row(i) for i in range(30, 40) if source_row(i)["id"] not in val_ids]
    accepted_path, teacher_manifest = write_teacher_fixture(
        tmp_path,
        rows=rows,
        completed=completed,
        stage=stage,
        parents=[
            ParentArtifact(
                validation_manifest.artifact_id,
                sha256_file(output_dir / "validation_manifest.json"),
            )
        ],
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["teacher_accepted"] = str(accepted_path)
    config["teacher_manifest"] = str(teacher_manifest)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        run_accepted_splits(config_path)

    assert not (output_dir / "sft_train_8k.jsonl").exists()
    assert not (output_dir / "grpo_train_4k.jsonl").exists()
    assert not (output_dir / "accepted_splits_manifest.json").exists()


def test_accepted_rejects_validation_ids(tmp_path: Path):
    config_path, output_dir, validation_manifest = prepare_validation(tmp_path)
    val_id = validation_manifest.stage_metadata["selected_ids"]["validation"][0]
    val_index = int(val_id.removeprefix("train-"))
    accepted_path, teacher_manifest = write_teacher_fixture(
        tmp_path,
        rows=[sft_row(val_index), *[sft_row(i) for i in range(30, 38)]],
        parents=[
            ParentArtifact(
                validation_manifest.artifact_id,
                sha256_file(output_dir / "validation_manifest.json"),
            )
        ],
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["teacher_accepted"] = str(accepted_path)
    config["teacher_manifest"] = str(teacher_manifest)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=f"validation id.*{val_id}"):
        run_accepted_splits(config_path)

    assert not (output_dir / "accepted_splits_manifest.json").exists()


@pytest.mark.parametrize("mode", ["validation", "accepted"])
@pytest.mark.parametrize("mismatch", ["hash", "count"])
def test_input_manifest_file_hash_and_count_mismatch_are_rejected(
    tmp_path: Path, mode: str, mismatch: str
):
    if mode == "validation":
        source_path, manifest_path, output_dir = write_source_fixture(tmp_path)
        config_path = write_validation_config(
            tmp_path, source_path, manifest_path, output_dir
        )
        data_path = source_path
    else:
        config_path, output_dir, _ = prepare_accepted(tmp_path)
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        data_path = Path(config["teacher_accepted"])
        manifest_path = Path(config["teacher_manifest"])

    manifest = load_manifest(manifest_path)
    if mismatch == "hash":
        data_path.write_bytes(data_path.read_bytes() + b"\n")
    else:
        changed_file = ArtifactFile(
            relative_path=manifest.files[0].relative_path,
            sha256=manifest.files[0].sha256,
            byte_size=manifest.files[0].byte_size,
            row_count=manifest.files[0].row_count + 1,
            field_schema=manifest.files[0].field_schema,
        )
        changed = ManifestV2.build(
            artifact_type=manifest.artifact_type,
            stage=manifest.stage,
            files=[changed_file],
            parents=manifest.parents,
            config=manifest.config,
            global_seed=manifest.global_seed,
            seed_derivation_version=manifest.seed_derivation_version,
            git_revision=manifest.git_revision,
            runtime_versions=manifest.runtime_versions,
            stage_metadata=manifest.stage_metadata,
            created_at=manifest.created_at,
        )
        publish_manifest(manifest_path, changed)

    runner = run_validation_splits if mode == "validation" else run_accepted_splits
    with pytest.raises(ValueError, match=mismatch):
        runner(config_path)


def test_validation_rejects_duplicate_ids_and_strict_jsonl_errors(tmp_path: Path):
    duplicate = [source_row(1), source_row(1)]
    source_path, source_manifest, output_dir = write_source_fixture(
        tmp_path, rows=duplicate
    )
    config_path = write_validation_config(
        tmp_path, source_path, source_manifest, output_dir, val_size=1, eval_size=1
    )
    with pytest.raises(ValueError, match="duplicate.*train-000001"):
        run_validation_splits(config_path)

    source_path.write_text(
        json.dumps(source_row(2)) + "\nnot-json\n", encoding="utf-8"
    )
    refresh_source_manifest_for_raw_file(
        source_path, source_manifest, row_count=2
    )
    with pytest.raises(ValueError, match=r"line 2"):
        run_validation_splits(config_path)


def test_validation_rejects_nonobject_jsonl_with_line_number(tmp_path: Path):
    source_path, source_manifest, output_dir = write_source_fixture(tmp_path)
    source_path.write_text("[]\n", encoding="utf-8")
    refresh_source_manifest_for_raw_file(
        source_path, source_manifest, row_count=1
    )
    config_path = write_validation_config(
        tmp_path, source_path, source_manifest, output_dir
    )

    with pytest.raises(ValueError, match=r"line 1.*object"):
        run_validation_splits(config_path)


@pytest.mark.parametrize("mode", ["validation", "accepted"])
def test_changed_input_during_build_preserves_old_completed_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
):
    if mode == "validation":
        source_path, source_manifest, output_dir = write_source_fixture(tmp_path)
        config_path = write_validation_config(
            tmp_path, source_path, source_manifest, output_dir
        )
        run_validation_splits(config_path)
        completion = output_dir / "validation_manifest.json"
        changed_path = source_path
        validator_name = "validate_normalized_source"
        runner = run_validation_splits
    else:
        config_path, output_dir, _ = prepare_accepted(tmp_path)
        run_accepted_splits(config_path)
        completion = output_dir / "accepted_splits_manifest.json"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        changed_path = Path(config["teacher_accepted"])
        validator_name = "validate_sft_record"
        runner = run_accepted_splits

    old_manifest = completion.read_bytes()
    real_validator = getattr(splits_module, validator_name)
    changed = False

    def mutate_after_first(row):
        nonlocal changed
        result = real_validator(row)
        if not changed:
            changed_path.write_bytes(changed_path.read_bytes() + b"\n")
            changed = True
        return result

    monkeypatch.setattr(splits_module, validator_name, mutate_after_first)

    with pytest.raises(ValueError, match="changed during"):
        runner(config_path)

    assert completion.read_bytes() == old_manifest


@pytest.mark.parametrize(
    ("mode", "manifest_name", "publish_count"),
    [
        ("validation", "validation_manifest.json", 3),
        ("accepted", "accepted_splits_manifest.json", 2),
    ],
)
def test_old_manifest_is_revoked_before_any_output_publish_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    manifest_name: str,
    publish_count: int,
):
    if mode == "validation":
        source_path, source_manifest, output_dir = write_source_fixture(tmp_path)
        config_path = write_validation_config(
            tmp_path, source_path, source_manifest, output_dir
        )
        runner = run_validation_splits
    else:
        config_path, output_dir, _ = prepare_accepted(tmp_path)
        runner = run_accepted_splits
    runner(config_path)
    real_publish = splits_module.publish_jsonl
    calls = 0

    def fail_publish(path, rows):
        nonlocal calls
        calls += 1
        if calls == publish_count:
            raise OSError("injected publish failure")
        real_publish(path, rows)

    monkeypatch.setattr(splits_module, "publish_jsonl", fail_publish)

    with pytest.raises(OSError, match="injected"):
        runner(config_path)

    assert not (output_dir / manifest_name).exists()


@pytest.mark.parametrize(
    ("mode", "manifest_name", "output_count"),
    [
        ("validation", "validation_manifest.json", 3),
        ("accepted", "accepted_splits_manifest.json", 2),
    ],
)
def test_manifest_change_after_data_publish_fails_second_unchanged_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    manifest_name: str,
    output_count: int,
):
    if mode == "validation":
        source_path, source_manifest, output_dir = write_source_fixture(tmp_path)
        config_path = write_validation_config(
            tmp_path, source_path, source_manifest, output_dir
        )
        changed_manifest = source_manifest
        runner = run_validation_splits
    else:
        config_path, output_dir, _ = prepare_accepted(tmp_path)
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        changed_manifest = Path(config["teacher_manifest"])
        runner = run_accepted_splits
    runner(config_path)

    real_publish = splits_module.publish_jsonl
    calls = 0

    def mutate_after_last_data_publish(path, rows):
        nonlocal calls
        calls += 1
        real_publish(path, rows)
        if calls == output_count:
            changed_manifest.write_bytes(changed_manifest.read_bytes() + b"\n")

    monkeypatch.setattr(splits_module, "publish_jsonl", mutate_after_last_data_publish)

    with pytest.raises(ValueError, match="changed during"):
        runner(config_path)

    assert not (output_dir / manifest_name).exists()


@pytest.mark.parametrize("mode", ["validation", "accepted"])
def test_parent_uses_initial_manifest_snapshot_digest_after_late_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
):
    if mode == "validation":
        source_path, input_manifest, output_dir = write_source_fixture(tmp_path)
        config_path = write_validation_config(
            tmp_path, source_path, input_manifest, output_dir
        )
        runner = run_validation_splits
    else:
        config_path, output_dir, _ = prepare_accepted(tmp_path)
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        input_manifest = Path(config["teacher_manifest"])
        runner = run_accepted_splits
    expected_manifest = load_manifest(input_manifest)
    expected_digest = sha256_file(input_manifest)
    real_artifact_file = splits_module._artifact_file
    mutated = False

    def mutate_during_output_metadata(*args, **kwargs):
        nonlocal mutated
        result = real_artifact_file(*args, **kwargs)
        if not mutated:
            input_manifest.write_bytes(input_manifest.read_bytes() + b"\n")
            mutated = True
        return result

    def inspect_then_abort(path, manifest):
        parent = next(
            item
            for item in manifest.parents
            if item.artifact_id == expected_manifest.artifact_id
        )
        assert parent.sha256 == expected_digest
        raise OSError("stop after parent inspection")

    monkeypatch.setattr(splits_module, "_artifact_file", mutate_during_output_metadata)
    monkeypatch.setattr(splits_module, "publish_manifest", inspect_then_abort)

    with pytest.raises(OSError, match="parent inspection"):
        runner(config_path)


def test_accepted_manifest_records_teacher_and_validation_parents(tmp_path: Path):
    config_path, output_dir, _ = prepare_accepted(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    manifest = run_accepted_splits(config_path)
    teacher_manifest_path = Path(config["teacher_manifest"])
    validation_manifest_path = output_dir / "validation_manifest.json"
    teacher_manifest = load_manifest(teacher_manifest_path)
    validation_manifest = load_manifest(validation_manifest_path)

    assert manifest == load_manifest(output_dir / "accepted_splits_manifest.json")
    assert manifest.stage == "build_accepted_splits"
    assert manifest.parents == (
        ParentArtifact(teacher_manifest.artifact_id, sha256_file(teacher_manifest_path)),
        ParentArtifact(
            validation_manifest.artifact_id, sha256_file(validation_manifest_path)
        ),
    )
    assert manifest.stage_metadata["completed"] is True
    assert manifest.stage_metadata["counts"] == {
        "accepted_pool": 12,
        "sft_train": 5,
        "grpo_train": 3,
    }
    assert [item.relative_path for item in manifest.files] == [
        "sft_train_8k.jsonl",
        "grpo_train_4k.jsonl",
    ]
    assert all(item.field_schema == SFT_SCHEMA for item in manifest.files)


def test_accepted_rejects_teacher_parent_mismatch(tmp_path: Path):
    config_path, output_dir, _ = prepare_accepted(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    teacher_manifest_path = Path(config["teacher_manifest"])
    original = load_manifest(teacher_manifest_path)
    mismatched = ManifestV2.build(
        artifact_type=original.artifact_type,
        stage=original.stage,
        files=original.files,
        parents=[ParentArtifact("a" * 64, "b" * 64)],
        config=original.config,
        global_seed=original.global_seed,
        seed_derivation_version=original.seed_derivation_version,
        git_revision=original.git_revision,
        runtime_versions=original.runtime_versions,
        stage_metadata=original.stage_metadata,
        created_at=original.created_at,
    )
    publish_manifest(teacher_manifest_path, mismatched)

    with pytest.raises(ValueError, match="parent"):
        run_accepted_splits(config_path)

    assert not (output_dir / "accepted_splits_manifest.json").exists()


def test_cli_help_works_from_arbitrary_cwd(tmp_path: Path):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "{validation,accepted}" in result.stdout
