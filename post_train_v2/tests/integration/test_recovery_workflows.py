from __future__ import annotations

import json
from pathlib import Path

import yaml

from post_train_v2.src.artifacts.hashing import sha256_file
from post_train_v2.src.artifacts.lineage import check_artifact_status
from post_train_v2.src.artifacts.manifest import ArtifactFile, ManifestV2, publish_manifest
from post_train_v2.src.generation.teacher_state import derive_resume_state
from post_train_v2.src.pipeline.runner import run_pipeline
from post_train_v2.verl.export.merge_actor import export_grpo_actors


def test_interrupted_teacher_resumes_from_committed_source_position() -> None:
    source_rows = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    accepted = [{"id": "a"}]
    rejected = [{"id": "b"}]

    state = derive_resume_state(
        source_rows,
        accepted,
        rejected,
        created_at="2026-06-16T00:00:00Z",
    )

    assert state.last_committed_position == 1
    assert list(state.accepted) == accepted
    assert list(state.rejected) == rejected


def test_runner_forwards_latest_trainer_checkpoint_when_stage_recovers(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "sft"
    (checkpoint_root / "checkpoint-100").mkdir(parents=True)
    (checkpoint_root / "checkpoint-200").mkdir()
    config_path = _write_resume_config(
        tmp_path,
        stage_name="full_sft",
        command=["torchrun", "post_train_v2/scripts/sft/train_full.py"],
        checkpoint_dir=str(checkpoint_root),
        checkpoint_glob="checkpoint-*",
    )
    calls: list[list[str]] = []

    run_pipeline(config_path, runner=lambda command: calls.append(command) or 0)

    assert calls == [
        [
            "torchrun",
            "post_train_v2/scripts/sft/train_full.py",
            "--resume-from-checkpoint",
            str(checkpoint_root / "checkpoint-200"),
        ]
    ]


def test_runner_forwards_latest_grpo_native_checkpoint_when_stage_recovers(
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "grpo" / "checkpoints"
    (checkpoint_root / "global_step_100").mkdir(parents=True)
    (checkpoint_root / "global_step_300").mkdir()
    config_path = _write_resume_config(
        tmp_path,
        stage_name="grpo_train",
        command=["python", "post_train_v2/scripts/grpo/train_grpo.py"],
        checkpoint_dir=str(checkpoint_root),
        checkpoint_glob="global_step_*",
    )
    calls: list[list[str]] = []

    run_pipeline(config_path, runner=lambda command: calls.append(command) or 0)

    assert calls[0][-2:] == [
        "--resume-from-checkpoint",
        str(checkpoint_root / "global_step_300"),
    ]


def test_grpo_export_can_rerun_without_training(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    for step in (100, 200):
        (run_dir / "checkpoints" / f"global_step_{step}" / "actor").mkdir(parents=True)
    (run_dir / "export").mkdir(parents=True)
    (run_dir / "export" / "selection.json").write_text(
        json.dumps(
            {
                "selected_best_step": 100,
                "final_step": 200,
                "best_checkpoint_path": str(run_dir / "checkpoints/global_step_100/actor"),
                "final_checkpoint_path": str(run_dir / "checkpoints/global_step_200/actor"),
            }
        ),
        encoding="utf-8",
    )
    calls: list[list[str]] = []

    export_grpo_actors(
        run_dir,
        runner=lambda command: _fake_merge(command, calls),
        direct_load_check=lambda path: None,
    )

    assert all("train_grpo.py" not in " ".join(command) for command in calls)
    assert (run_dir / "export" / "export_summary.json").is_file()


def test_changed_input_or_config_refuses_resume_and_temp_files_are_ignored(
    tmp_path: Path,
) -> None:
    manifest_path = _write_complete_manifest(tmp_path / "stage")
    (manifest_path.parent / "stage.jsonl").write_text("changed\n", encoding="utf-8")
    assert check_artifact_status(manifest_path, expected_config={"seed": 1}).state == "stale"

    assert check_artifact_status(manifest_path, expected_config={"seed": 2}).state == "stale"

    temp_path = manifest_path.with_suffix(".json.tmp")
    temp_path.write_text(manifest_path.read_text(encoding="utf-8"), encoding="utf-8")
    manifest_path.unlink()
    assert check_artifact_status(manifest_path).state == "missing"


def _write_resume_config(
    tmp_path: Path,
    *,
    stage_name: str,
    command: list[str],
    checkpoint_dir: str,
    checkpoint_glob: str,
) -> Path:
    config = {
        "events_path": str(tmp_path / "events.jsonl"),
        "stages": [
            {
                "name": stage_name,
                "command": command,
                "dependencies": [],
                "manifest_path": str(tmp_path / "missing" / "manifest.json"),
                "resources": "gpu2_verl" if stage_name == "grpo_train" else "gpu2_ddp",
                "resume": {
                    "arg": "--resume-from-checkpoint",
                    "checkpoint_dir": checkpoint_dir,
                    "glob": checkpoint_glob,
                },
            }
        ],
    }
    path = tmp_path / f"{stage_name}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _write_complete_manifest(directory: Path) -> Path:
    directory.mkdir(parents=True)
    data_path = directory / "stage.jsonl"
    data_path.write_text("ok\n", encoding="utf-8")
    manifest = ManifestV2.build(
        artifact_type="dataset",
        stage="stage",
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


def _fake_merge(command: list[str], calls: list[list[str]]) -> int:
    calls.append(command)
    if "merge" in command:
        target = Path(command[command.index("--target_dir") + 1])
        target.mkdir(parents=True, exist_ok=True)
    return 0
