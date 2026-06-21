from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

from post_train_v2.src.artifacts.hashing import sha256_file
from post_train_v2.src.artifacts.manifest import ArtifactFile, ManifestV2, publish_manifest
from post_train_v2.src.pipeline.runner import PipelineRunError, run_pipeline


def test_pipeline_runner_skips_complete_stage_and_dry_run_imports_no_ml_libs(
    tmp_path: Path,
    capsys,
) -> None:
    manifest_path = _write_complete_manifest(tmp_path / "done", "build_source")
    config_path = _write_config(
        tmp_path,
        stages=[
            {
                "name": "build_source",
                "command": ["python", "build_source.py"],
                "dependencies": [],
                "manifest_path": str(manifest_path),
                "resources": "cpu",
                "expected_config": {"seed": 1},
            }
        ],
    )
    sys.modules.pop("transformers", None)
    sys.modules.pop("vllm", None)

    summary = run_pipeline(config_path, dry_run=True)

    output = capsys.readouterr().out
    assert "build_source" in output
    assert "SKIP complete" in output
    assert summary["events"][0]["decision"] == "skip"
    assert "transformers" not in sys.modules
    assert "vllm" not in sys.modules


def test_pipeline_runner_stops_on_stale_stage_unless_rebuilt(tmp_path: Path) -> None:
    manifest_path = _write_complete_manifest(tmp_path / "stale", "build_source")
    (manifest_path.parent / "build_source.jsonl").write_text("changed\n", encoding="utf-8")
    config_path = _write_config(
        tmp_path,
        stages=[
            {
                "name": "build_source",
                "command": ["python", "-c", "print('rebuilt')"],
                "dependencies": [],
                "manifest_path": str(manifest_path),
                "resources": "cpu",
                "expected_config": {"seed": 1},
            }
        ],
    )

    with pytest.raises(PipelineRunError, match="stale"):
        run_pipeline(config_path)

    calls: list[list[str]] = []
    summary = run_pipeline(
        config_path,
        rebuild_stages={"build_source"},
        runner=lambda command: calls.append(command) or 0,
    )

    assert calls == [["python", "-c", "print('rebuilt')"]]
    assert summary["events"][0]["decision"] == "run"


def test_pipeline_runner_failed_subprocess_stops_downstream_and_writes_events(
    tmp_path: Path,
) -> None:
    config_path = _write_config(
        tmp_path,
        stages=[
            {
                "name": "build_source",
                "command": ["python", "fail.py"],
                "dependencies": [],
                "manifest_path": str(tmp_path / "missing" / "manifest.json"),
                "resources": "cpu",
            },
            {
                "name": "validation_split",
                "command": ["python", "downstream.py"],
                "dependencies": ["build_source"],
                "manifest_path": str(tmp_path / "downstream" / "manifest.json"),
                "resources": "cpu",
            },
        ],
    )
    calls: list[list[str]] = []

    with pytest.raises(PipelineRunError, match="failed"):
        run_pipeline(config_path, runner=lambda command: calls.append(command) or 7)

    assert calls == [["python", "fail.py"]]
    events = [
        json.loads(line)
        for line in (tmp_path / "pipeline_events.jsonl").read_text().splitlines()
    ]
    assert len(events) == 1
    assert events[0]["stage"] == "build_source"
    assert events[0]["exit_code"] == 7
    assert events[0]["output_manifest_sha256"] is None


def test_pipeline_runner_refuses_production_without_runtime_gate(tmp_path: Path) -> None:
    gate = tmp_path / "runtime_acceptance.json"
    gate.write_text(json.dumps({"level1_runtime_gates_passed": False}), encoding="utf-8")
    config_path = _write_config(
        tmp_path,
        require_runtime_gate=True,
        runtime_acceptance_file=str(gate),
        stages=[
            {
                "name": "build_source",
                "command": ["python", "stage.py"],
                "dependencies": [],
                "manifest_path": str(tmp_path / "missing.json"),
                "resources": "cpu",
            }
        ],
    )

    with pytest.raises(PipelineRunError, match="Level 1 runtime gates"):
        run_pipeline(config_path, runner=lambda command: 0)


def _write_complete_manifest(directory: Path, stage: str) -> Path:
    directory.mkdir(parents=True)
    data_path = directory / f"{stage}.jsonl"
    data_path.write_text("ok\n", encoding="utf-8")
    manifest = ManifestV2.build(
        artifact_type="dataset",
        stage=stage,
        files=[
            ArtifactFile(
                data_path.name,
                sha256_file(data_path),
                data_path.stat().st_size,
                1,
                {"id": "string"},
            )
        ],
        parents=[],
        config={"seed": 1},
        stage_metadata={"completed": True},
        git_revision="fixture",
    )
    manifest_path = directory / "manifest.json"
    publish_manifest(manifest_path, manifest)
    return manifest_path


def _write_config(
    tmp_path: Path,
    *,
    stages: list[dict],
    require_runtime_gate: bool = False,
    runtime_acceptance_file: str | None = None,
) -> Path:
    config = {
        "events_path": str(tmp_path / "pipeline_events.jsonl"),
        "require_runtime_gate": require_runtime_gate,
        "runtime_acceptance_file": runtime_acceptance_file,
        "stages": stages,
    }
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path
