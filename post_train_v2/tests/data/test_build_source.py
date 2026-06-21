from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from post_train_v2.src.artifacts.hashing import (
    sha256_canonical_json,
    sha256_config,
    sha256_file,
)
from post_train_v2.src.artifacts.manifest import ManifestV2, load_manifest
from post_train_v2.src.countdown.bucketing import assign_bucket
from post_train_v2.src.countdown.prompts import build_solution_prompt
from post_train_v2.src.countdown.solver import solve_countdown
from post_train_v2.src.data.schema import validate_normalized_source
from post_train_v2.src.data import source as source_module
from post_train_v2.src.data.source import (
    build_test_source,
    build_train_source,
    run_build_source,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "post_train_v2/scripts/data/build_source.py"
DATA_FILENAMES = (
    "source_all.jsonl",
    "solvable_train.jsonl",
    "unsolved_train.jsonl",
    "test_solved.jsonl",
)
NORMALIZED_SCHEMA = {
    "id": "string",
    "source_index": "integer",
    "numbers": "array[integer]",
    "target": "integer",
    "gold_expr": "string",
    "prompt": "string",
    "bucket": "object",
}
UNSOLVED_SCHEMA = {
    "id": "string",
    "source_index": "integer",
    "numbers": "array[integer]",
    "target": "integer",
    "reason": "string",
}


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def write_fixture(
    tmp_path: Path,
    *,
    train_rows: list[dict] | None = None,
    test_rows: list[dict] | None = None,
) -> tuple[Path, Path, Path, Path]:
    train_path = tmp_path / "inputs" / "train.parquet"
    test_path = tmp_path / "inputs" / "test.json"
    output_dir = tmp_path / "outputs"
    config_path = tmp_path / "configs" / "build_source.yaml"
    train_path.parent.mkdir(parents=True)
    config_path.parent.mkdir(parents=True)
    pd.DataFrame(
        train_rows
        or [
            {"nums": [1, 2], "target": 3},
            {"nums": np.array([3, 3, 8, 8]), "target": 24},
        ]
    ).to_parquet(train_path, index=False)
    test_path.write_text(
        json.dumps(
            test_rows
            or [
                {"id": 7, "numbers": [1, 2], "target": 3},
                {"id": 9, "numbers": "[3, 3, 8, 8]", "target": 24},
            ]
        ),
        encoding="utf-8",
    )
    config = {
        "seed": 42,
        "train_input": str(train_path),
        "test_input": str(test_path),
        "output_dir": str(output_dir),
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path, train_path, test_path, output_dir


def test_build_train_and_test_source_preserve_ids_indexes_prompts_solver_and_bucket():
    frame = pd.DataFrame(
        [
            {"nums": (1, 2), "target": 3},
            {"nums": np.array([3, 3, 8, 8]), "target": 24},
        ],
        index=[40, 99],
    )

    solvable, unsolved = build_train_source(frame)
    test_rows = build_test_source(
        [{"id": 7, "numbers": "[1, 2]", "target": 3}]
    )

    assert unsolved == []
    assert [row["id"] for row in solvable] == [
        "train-000001",
        "train-000002",
    ]
    assert [row["source_index"] for row in solvable] == [1, 2]
    assert test_rows[0]["id"] == "test-000007"
    assert test_rows[0]["source_index"] == 7
    for row in [*solvable, *test_rows]:
        assert row["gold_expr"] == solve_countdown(row["numbers"], row["target"])
        assert row["prompt"] == build_solution_prompt(
            row["numbers"], row["target"]
        )
        assert row["bucket"] == assign_bucket(row["numbers"], row["gold_expr"])
        assert validate_normalized_source(row) == row


@pytest.mark.parametrize("column", ["nums", "numbers"])
@pytest.mark.parametrize(
    "numbers",
    [
        [1, 2],
        (1, 2),
        np.array([1, 2]),
        "[1, 2]",
    ],
)
def test_train_source_keeps_approved_column_and_numbers_container_compatibility(
    column: str,
    numbers,
):
    solvable, unsolved = build_train_source(
        pd.DataFrame([{column: numbers, "target": 3}])
    )

    assert unsolved == []
    assert solvable[0]["numbers"] == [1, 2]


@pytest.mark.parametrize("column", ["nums", "numbers"])
@pytest.mark.parametrize(
    "numbers",
    [
        [1, 2],
        (1, 2),
        np.array([1, 2]),
        "[1, 2]",
    ],
)
def test_test_source_keeps_approved_column_and_numbers_container_compatibility(
    column: str,
    numbers,
):
    solved = build_test_source([{"id": 7, column: numbers, "target": 3}])

    assert solved[0]["numbers"] == [1, 2]


def test_build_train_source_keeps_unsolved_rows_with_exact_contract():
    solvable, unsolved = build_train_source(
        pd.DataFrame(
            [
                {"nums": [1, 2], "target": 3},
                {"nums": [1, 1], "target": 3},
            ]
        )
    )

    assert [row["id"] for row in solvable] == ["train-000001"]
    assert unsolved == [
        {
            "id": "train-000002",
            "source_index": 2,
            "numbers": [1, 1],
            "target": 3,
            "reason": "no_solution",
        }
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("target", True),
        ("target", 3.0),
        ("target", "3"),
        ("nums", [1, True]),
        ("nums", [1, 2.0]),
        ("nums", ["1", 2]),
        ("nums", "not-json"),
        ("nums", '{"not":"a-list"}'),
        ("nums", []),
        ("nums", [-1, 2]),
    ],
)
def test_train_source_rejects_malformed_exact_integer_inputs(field, value):
    row = {"nums": [1, 2], "target": 3}
    row[field] = value

    with pytest.raises(ValueError, match=field.replace("nums", "numbers")):
        build_train_source(pd.DataFrame([row]))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("id", True),
        ("id", 7.0),
        ("id", "7"),
        ("id", 0),
        ("id", -1),
        ("target", True),
        ("target", "3"),
        ("numbers", [1, 2.0]),
    ],
)
def test_test_source_rejects_malformed_exact_integer_inputs(field, value):
    row = {"id": 7, "numbers": [1, 2], "target": 3}
    row[field] = value

    with pytest.raises(ValueError, match=field):
        build_test_source([row])


def test_build_test_source_rejects_duplicate_native_ids():
    with pytest.raises(ValueError, match="duplicate.*7"):
        build_test_source(
            [
                {"id": 7, "numbers": [1, 2], "target": 3},
                {"id": 7, "numbers": [3, 4], "target": 7},
            ]
        )


def test_run_build_source_limit_preserves_original_indexes_and_writes_unsolved(
    tmp_path: Path,
):
    config_path, _, _, output_dir = write_fixture(
        tmp_path,
        train_rows=[
            {"numbers": [1, 2], "target": 3},
            {"numbers": [1, 1], "target": 3},
            {"numbers": [2, 2], "target": 4},
        ],
    )

    manifest = run_build_source(config_path, limit=2)

    assert [row["source_index"] for row in read_jsonl(
        output_dir / "source_all.jsonl"
    )] == [1]
    assert read_jsonl(output_dir / "unsolved_train.jsonl") == [
        {
            "id": "train-000002",
            "source_index": 2,
            "numbers": [1, 1],
            "target": 3,
            "reason": "no_solution",
        }
    ]
    assert manifest.stage_metadata["limit"] == 2
    assert manifest.stage_metadata["counts"] == {
        "train_input": 2,
        "solvable_train": 1,
        "unsolved_train": 1,
        "test_solved": 2,
    }


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "1"])
def test_run_build_source_rejects_invalid_limits(tmp_path: Path, limit):
    config_path, _, _, output_dir = write_fixture(tmp_path)

    with pytest.raises(ValueError, match="limit"):
        run_build_source(config_path, limit=limit)

    assert not output_dir.exists()


@pytest.mark.parametrize("preexisting_manifest", [False, True])
def test_unsolved_test_fails_without_publishing_or_overwriting_manifest(
    tmp_path: Path,
    preexisting_manifest: bool,
):
    config_path, _, _, output_dir = write_fixture(
        tmp_path,
        test_rows=[{"id": 7, "numbers": [1, 1], "target": 3}],
    )
    manifest_path = output_dir / "manifest.json"
    original = b'{"existing":true}'
    if preexisting_manifest:
        output_dir.mkdir(parents=True)
        manifest_path.write_bytes(original)

    with pytest.raises(ValueError, match="test-000007.*no solution"):
        run_build_source(config_path, limit=None)

    if preexisting_manifest:
        assert manifest_path.read_bytes() == original
    else:
        assert not manifest_path.exists()
    assert all(not (output_dir / name).exists() for name in DATA_FILENAMES)


@pytest.mark.parametrize("failed_publish", [1, 2, 3, 4])
def test_rebuild_revokes_old_manifest_before_each_possible_data_publish_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_publish: int,
):
    config_path, _, _, output_dir = write_fixture(tmp_path)
    old_manifest = run_build_source(config_path, limit=None)
    old_data = {
        name: (output_dir / name).read_bytes() for name in DATA_FILENAMES
    }
    calls = 0
    real_publish_jsonl = source_module.publish_jsonl

    def failing_publish(path, rows):
        nonlocal calls
        calls += 1
        if calls == failed_publish:
            raise OSError(f"publish {failed_publish} failed")
        real_publish_jsonl(path, rows)

    monkeypatch.setattr(source_module, "publish_jsonl", failing_publish)

    with pytest.raises(OSError, match=f"publish {failed_publish} failed"):
        run_build_source(config_path, limit=None)

    assert calls == failed_publish
    assert not (output_dir / "manifest.json").exists()
    for name in DATA_FILENAMES[: failed_publish - 1]:
        assert (output_dir / name).read_bytes() == old_data[name]
    assert old_manifest.artifact_id


@pytest.mark.parametrize("failed_publish", [1, 2, 3, 4])
def test_first_build_data_publish_failure_never_creates_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_publish: int,
):
    config_path, _, _, output_dir = write_fixture(tmp_path)
    calls = 0
    real_publish_jsonl = source_module.publish_jsonl

    def failing_publish(path, rows):
        nonlocal calls
        calls += 1
        if calls == failed_publish:
            raise OSError(f"publish {failed_publish} failed")
        real_publish_jsonl(path, rows)

    monkeypatch.setattr(source_module, "publish_jsonl", failing_publish)

    with pytest.raises(OSError, match=f"publish {failed_publish} failed"):
        run_build_source(config_path, limit=None)

    assert calls == failed_publish
    assert not (output_dir / "manifest.json").exists()
    assert all(
        (output_dir / name).exists()
        for name in DATA_FILENAMES[: failed_publish - 1]
    )


def test_run_build_source_manifest_has_exact_files_parents_config_and_metadata(
    tmp_path: Path,
):
    config_path, train_path, test_path, output_dir = write_fixture(tmp_path)
    logical_config = {
        "seed": 42,
        "train_input": "../inputs/train.parquet",
        "test_input": "../inputs/test.json",
        "output_dir": "../outputs",
    }

    manifest = run_build_source(config_path, limit=None)
    loaded = load_manifest(output_dir / "manifest.json")

    assert loaded == manifest
    assert manifest.stage == "build_source"
    assert manifest.artifact_type == "dataset"
    assert manifest.config == logical_config
    assert manifest.config_sha256 == sha256_config(logical_config)
    assert manifest.global_seed == 42
    expected_parent_hashes = [
        ("raw_train", sha256_file(train_path)),
        ("raw_test", sha256_file(test_path)),
    ]
    assert {
        parent.artifact_id: parent.sha256 for parent in manifest.parents
    } == {
        sha256_canonical_json({"kind": kind, "sha256": sha256}): sha256
        for kind, sha256 in expected_parent_hashes
    }
    assert [item.relative_path for item in manifest.files] == list(DATA_FILENAMES)
    expected_rows = {
        "source_all.jsonl": 2,
        "solvable_train.jsonl": 2,
        "unsolved_train.jsonl": 0,
        "test_solved.jsonl": 2,
    }
    for item in manifest.files:
        path = output_dir / item.relative_path
        assert item.sha256 == sha256_file(path)
        assert item.byte_size == path.stat().st_size
        assert item.row_count == expected_rows[item.relative_path]
        assert item.field_schema == (
            UNSOLVED_SCHEMA
            if item.relative_path == "unsolved_train.jsonl"
            else NORMALIZED_SCHEMA
        )
    assert manifest.stage_metadata == {
        "completed": True,
        "counts": {
            "train_input": 2,
            "solvable_train": 2,
            "unsolved_train": 0,
            "test_solved": 2,
        },
        "limit": None,
        "inputs": {
            "train": "../inputs/train.parquet",
            "test": "../inputs/test.json",
        },
        "output": "../outputs",
    }
    metadata_text = json.dumps(manifest.stage_metadata, sort_keys=True)
    assert str(tmp_path.resolve()) not in metadata_text
    assert str(REPO_ROOT.resolve()) not in metadata_text


@pytest.mark.parametrize("changed_input", ["train", "test"])
def test_input_change_during_build_preserves_old_outputs_and_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed_input: str,
):
    config_path, train_path, test_path, output_dir = write_fixture(tmp_path)
    run_build_source(config_path, limit=None)
    manifest_path = output_dir / "manifest.json"
    old_manifest = manifest_path.read_bytes()
    old_data = {
        name: (output_dir / name).read_bytes() for name in DATA_FILENAMES
    }

    if changed_input == "train":
        real_read_parquet = source_module.pd.read_parquet

        def replacing_read_parquet(path):
            frame = real_read_parquet(path)
            pd.DataFrame([{"nums": [4, 5], "target": 9}]).to_parquet(
                train_path,
                index=False,
            )
            return frame

        monkeypatch.setattr(
            source_module.pd,
            "read_parquet",
            replacing_read_parquet,
        )
    else:
        real_solve = source_module.solve_countdown
        replaced = False

        def replacing_solve(numbers, target):
            nonlocal replaced
            if not replaced:
                replaced = True
                test_path.write_text(
                    json.dumps(
                        [{"id": 11, "numbers": [4, 5], "target": 9}]
                    ),
                    encoding="utf-8",
                )
            return real_solve(numbers, target)

        monkeypatch.setattr(source_module, "solve_countdown", replacing_solve)

    with pytest.raises(ValueError, match=f"{changed_input} input changed"):
        run_build_source(config_path, limit=None)

    assert manifest_path.read_bytes() == old_manifest
    assert {
        name: (output_dir / name).read_bytes() for name in DATA_FILENAMES
    } == old_data


def test_manifest_identity_is_stable_across_real_absolute_fixture_roots(
    tmp_path: Path,
):
    roots = [tmp_path / "checkout-a", tmp_path / "checkout-b"]
    manifests = []
    for root in roots:
        train_path = root / "inputs" / "train.parquet"
        test_path = root / "inputs" / "test.json"
        output_dir = root / "outputs"
        config_path = root / "configs" / "build_source.yaml"
        train_path.parent.mkdir(parents=True)
        config_path.parent.mkdir(parents=True)
        pd.DataFrame([{"nums": [1, 2], "target": 3}]).to_parquet(
            train_path,
            index=False,
        )
        test_path.write_text(
            json.dumps([{"id": 7, "numbers": [1, 2], "target": 3}]),
            encoding="utf-8",
        )
        absolute_config = {
            "seed": 42,
            "train_input": str(train_path.resolve()),
            "test_input": str(test_path.resolve()),
            "output_dir": str(output_dir.resolve()),
        }
        config_path.write_text(
            yaml.safe_dump(absolute_config, sort_keys=False),
            encoding="utf-8",
        )
        manifests.append(run_build_source(config_path, limit=None))

    assert manifests[0].artifact_id == manifests[1].artifact_id
    assert manifests[0].parents == manifests[1].parents
    assert manifests[0].config_sha256 == manifests[1].config_sha256
    assert manifests[0].config == manifests[1].config == {
        "seed": 42,
        "train_input": "../inputs/train.parquet",
        "test_input": "../inputs/test.json",
        "output_dir": "../outputs",
    }
    config_text = json.dumps(manifests[0].config, sort_keys=True)
    assert all(str(root.resolve()) not in config_text for root in roots)
    assert manifests[0].stage_metadata == manifests[1].stage_metadata == {
        "completed": True,
        "counts": {
            "train_input": 1,
            "solvable_train": 1,
            "unsolved_train": 0,
            "test_solved": 1,
        },
        "limit": None,
        "inputs": {
            "train": "../inputs/train.parquet",
            "test": "../inputs/test.json",
        },
        "output": "../outputs",
    }


def test_run_build_source_repeated_build_has_deterministic_data_and_artifact_id(
    tmp_path: Path,
):
    config_path, _, _, output_dir = write_fixture(tmp_path)

    first = run_build_source(config_path, limit=None)
    first_bytes = {
        name: (output_dir / name).read_bytes() for name in DATA_FILENAMES
    }
    second = run_build_source(config_path, limit=None)

    assert second.artifact_id == first.artifact_id
    assert {
        name: (output_dir / name).read_bytes() for name in DATA_FILENAMES
    } == first_bytes
    assert second.created_at != ""


def test_default_config_has_approved_phase1_values():
    config = yaml.safe_load(
        (
            REPO_ROOT / "post_train_v2/configs/data/build_source.yaml"
        ).read_text(encoding="utf-8")
    )

    assert config == {
        "seed": 42,
        "train_input": "datasets/raw_train.parquet",
        "test_input": "datasets/raw_test.json",
        "output_dir": "post_train_v2/data/processed",
    }


def test_cli_help_succeeds_from_other_cwd(tmp_path: Path):
    env = dict(os.environ)
    env["PYTHONPATH"] = ""

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout
    assert "--limit" in result.stdout
