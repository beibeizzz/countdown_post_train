"""Resumable V2 pipeline runner."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from post_train_v2.src.artifacts.atomic import publish_jsonl
from post_train_v2.src.artifacts.hashing import sha256_file
from post_train_v2.src.artifacts.lineage import check_artifact_status
from post_train_v2.src.pipeline.model import (
    StageSpec,
    select_stage_window,
    validate_stage_specs,
)

Runner = Callable[[list[str]], int]


class PipelineRunError(RuntimeError):
    pass


def run_pipeline(
    config_path: str | Path,
    *,
    from_stage: str | None = None,
    through_stage: str | None = None,
    rebuild_stages: Iterable[str] = (),
    dry_run: bool = False,
    runner: Runner | None = None,
) -> dict[str, Any]:
    config = _load_config(Path(config_path))
    if not dry_run and config.get("require_runtime_gate") is True:
        _require_runtime_gate(config)

    specs = _load_stage_specs(config)
    selected_names = select_stage_window(
        specs,
        from_stage=from_stage,
        through_stage=through_stage,
    )
    rebuild = set(rebuild_stages)
    events_path = Path(config["events_path"])
    runner = runner or _subprocess_runner
    events: list[dict[str, Any]] = []

    for stage_name in selected_names:
        stage = specs[stage_name]
        status = check_artifact_status(
            stage.manifest_path,
            expected_config=config.get("stage_configs", {}).get(stage.name),
            parent_manifest_paths=_parent_manifest_paths(config, stage.name),
        )
        decision = _decision(stage.name, status.state, rebuild)
        event = _base_event(stage, status.reason, decision)
        if dry_run:
            _print_dry_run(stage, decision, status.reason)
            events.append(event)
            continue
        if decision == "skip":
            events.append(event)
            _write_events(events_path, events)
            continue
        if decision == "stop":
            events.append(event)
            _write_events(events_path, events)
            raise PipelineRunError(f"stage {stage.name} is stale: {status.reason}")

        command = _resolved_command(config, stage)
        event["command"] = command
        started_at = _utc_now()
        exit_code = runner(command)
        finished_at = _utc_now()
        event.update(
            {
                "start_time": started_at,
                "end_time": finished_at,
                "exit_code": exit_code,
                "output_manifest_sha256": _optional_sha256(stage.manifest_path),
            }
        )
        events.append(event)
        _write_events(events_path, events)
        if exit_code != 0:
            raise PipelineRunError(f"stage {stage.name} failed with exit {exit_code}")

    return {"events": events, "selected_stages": selected_names}


def _load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("pipeline config must be a mapping")
    if "events_path" not in value:
        raise ValueError("pipeline config missing events_path")
    if "stages" not in value or not isinstance(value["stages"], list):
        raise ValueError("pipeline config missing stages")
    return value


def _load_stage_specs(config: Mapping[str, Any]) -> dict[str, StageSpec]:
    stages = []
    stage_configs: dict[str, Any] = {}
    stage_resume: dict[str, Any] = {}
    for item in config["stages"]:
        if not isinstance(item, Mapping):
            raise ValueError("stage config must be a mapping")
        name = str(item["name"])
        stages.append(
            StageSpec(
                name=name,
                command=tuple(str(part) for part in item["command"]),
                dependencies=tuple(str(part) for part in item.get("dependencies", [])),
                manifest_path=Path(item["manifest_path"]),
                resources=item["resources"],
            )
        )
        if "expected_config" in item:
            stage_configs[name] = dict(item["expected_config"])
        if "resume" in item:
            stage_resume[name] = dict(item["resume"])
    config["stage_configs"] = stage_configs
    config["stage_resume"] = stage_resume
    return validate_stage_specs(stages)


def _decision(stage_name: str, state: str, rebuild_stages: set[str]) -> str:
    if stage_name in rebuild_stages:
        return "run"
    if state == "complete":
        return "skip"
    if state == "missing":
        return "run"
    return "stop"


def _base_event(stage: StageSpec, reason: str, decision: str) -> dict[str, Any]:
    return {
        "stage": stage.name,
        "command": list(stage.command),
        "resources": stage.resources,
        "decision": decision,
        "reason": reason,
        "start_time": None,
        "end_time": None,
        "exit_code": None,
        "input_manifest_hashes": [],
        "output_manifest_sha256": _optional_sha256(stage.manifest_path),
    }


def _print_dry_run(stage: StageSpec, decision: str, reason: str) -> None:
    label = "SKIP complete" if decision == "skip" else decision.upper()
    print(f"{stage.name}: {label} | {' '.join(stage.command)} | {reason}")


def _write_events(path: Path, events: list[Mapping[str, Any]]) -> None:
    publish_jsonl(path, events)


def _optional_sha256(path: Path) -> str | None:
    return sha256_file(path) if path.is_file() else None


def _parent_manifest_paths(config: Mapping[str, Any], stage_name: str) -> list[Path]:
    parents = config.get("parent_manifest_paths", {}).get(stage_name, [])
    if not isinstance(parents, list):
        raise ValueError(f"parent_manifest_paths.{stage_name} must be a list")
    return [Path(item) for item in parents]


def _resolved_command(config: Mapping[str, Any], stage: StageSpec) -> list[str]:
    command = list(stage.command)
    resume = config.get("stage_resume", {}).get(stage.name)
    if not isinstance(resume, Mapping):
        return command
    checkpoint = _latest_checkpoint(
        Path(resume["checkpoint_dir"]),
        str(resume.get("glob", "checkpoint-*")),
    )
    if checkpoint is None:
        return command
    return [*command, str(resume.get("arg", "--resume-from-checkpoint")), str(checkpoint)]


def _latest_checkpoint(checkpoint_dir: Path, pattern: str) -> Path | None:
    if not checkpoint_dir.is_dir():
        return None
    candidates = [path for path in checkpoint_dir.glob(pattern) if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=_checkpoint_sort_key)


def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
    suffix_digits = ""
    for character in reversed(path.name):
        if character.isdigit():
            suffix_digits = character + suffix_digits
        elif suffix_digits:
            break
    step = int(suffix_digits) if suffix_digits else -1
    return step, path.name


def _subprocess_runner(command: list[str]) -> int:
    return subprocess.run(command, check=False).returncode


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _require_runtime_gate(config: Mapping[str, Any]) -> None:
    path_value = config.get("runtime_acceptance_file")
    if not path_value:
        raise PipelineRunError("Level 1 runtime gates acceptance file is required")
    path = Path(path_value)
    if not path.is_file():
        raise PipelineRunError(f"Level 1 runtime gates file is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or value.get("level1_runtime_gates_passed") is not True:
        raise PipelineRunError("Level 1 runtime gates are not recorded as passed")
