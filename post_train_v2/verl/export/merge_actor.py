"""Export verl GRPO best/final actor checkpoints with stock model_merger."""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from post_train_v2.src.artifacts.atomic import publish_json

Runner = Callable[[list[str]], int]
DirectLoadCheck = Callable[[Path], None]


def build_model_merger_command(
    *,
    local_dir: str | Path,
    target_dir: str | Path,
    backend: str = "fsdp",
) -> list[str]:
    return [
        "python",
        "-m",
        "verl.model_merger",
        "merge",
        "--backend",
        backend,
        "--local_dir",
        _path_argument(local_dir),
        "--target_dir",
        _path_argument(target_dir),
    ]


def export_grpo_actors(
    run_dir: str | Path,
    *,
    selection_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    direct_load_check: DirectLoadCheck | None = None,
    runner: Runner | None = None,
    prune: bool = False,
) -> dict[str, Any]:
    base = Path(run_dir)
    selection = _read_selection(
        Path(selection_path) if selection_path is not None else base / "export" / "selection.json"
    )
    destination = Path(output_dir) if output_dir is not None else base / "export"
    runner = runner or _subprocess_runner
    direct_load_check = direct_load_check or _require_direct_loadable

    _run_or_raise(runner, ["python", "-m", "verl.model_merger", "--help"])
    best = _merge_one(
        name="best",
        source_step=int(selection["selected_best_step"]),
        source_path=Path(str(selection["best_checkpoint_path"])),
        destination=destination / "best",
        runner=runner,
        direct_load_check=direct_load_check,
    )
    final = _merge_one(
        name="final",
        source_step=int(selection["final_step"]),
        source_path=Path(str(selection["final_checkpoint_path"])),
        destination=destination / "final",
        runner=runner,
        direct_load_check=direct_load_check,
    )

    summary = {
        "best": best,
        "final": final,
        "selection_path": str(
            Path(selection_path) if selection_path is not None else base / "export" / "selection.json"
        ),
    }
    publish_json(destination / "export_summary.json", summary)

    if prune:
        keep_steps = _retained_checkpoint_steps(
            base,
            best_step=int(selection["selected_best_step"]),
        )
        _prune_old_checkpoints(base, keep_steps)
    return summary


def _merge_one(
    *,
    name: str,
    source_step: int,
    source_path: Path,
    destination: Path,
    runner: Runner,
    direct_load_check: DirectLoadCheck,
) -> dict[str, Any]:
    if not source_path.is_dir():
        raise FileNotFoundError(f"missing source actor checkpoint: {source_path}")
    if destination.exists():
        shutil.rmtree(destination)
    command = build_model_merger_command(local_dir=source_path, target_dir=destination)
    _run_or_raise(runner, command)
    if not destination.is_dir():
        raise FileNotFoundError(f"model_merger did not create target_dir: {destination}")
    direct_load_check(destination)
    manifest = {
        "export_name": name,
        "export_kind": "full_model",
        "direct_loadable": True,
        "source_step": source_step,
        "source_checkpoint_path": str(source_path),
    }
    publish_json(destination / "export_manifest.json", manifest)
    return manifest


def _read_selection(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("selection JSON must be an object")
    for key in (
        "selected_best_step",
        "final_step",
        "best_checkpoint_path",
        "final_checkpoint_path",
    ):
        if key not in value:
            raise ValueError(f"selection JSON missing {key}")
    return value


def _subprocess_runner(command: list[str]) -> int:
    return subprocess.run(command, check=False).returncode


def _run_or_raise(runner: Runner, command: list[str]) -> None:
    returncode = runner(command)
    if returncode != 0:
        raise RuntimeError(f"command failed with exit {returncode}: {' '.join(command)}")


def _require_direct_loadable(path: Path) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=True)
    AutoModelForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        trust_remote_code=True,
        device_map="cpu",
    )


def _retained_checkpoint_steps(run_dir: Path, *, best_step: int) -> set[int]:
    steps = sorted(_checkpoint_steps(run_dir))
    latest_two = set(steps[-2:])
    latest_two.add(best_step)
    return latest_two


def _checkpoint_steps(run_dir: Path) -> list[int]:
    checkpoint_root = run_dir / "checkpoints"
    if not checkpoint_root.is_dir():
        return []
    steps: list[int] = []
    for path in checkpoint_root.iterdir():
        if not path.is_dir() or not path.name.startswith("global_step_"):
            continue
        suffix = path.name.removeprefix("global_step_")
        if suffix.isdigit():
            steps.append(int(suffix))
    return steps


def _prune_old_checkpoints(run_dir: Path, keep_steps: set[int]) -> None:
    checkpoint_root = run_dir / "checkpoints"
    if not checkpoint_root.is_dir():
        return
    for step in _checkpoint_steps(run_dir):
        if step not in keep_steps:
            shutil.rmtree(checkpoint_root / f"global_step_{step}")


def _path_argument(path: str | Path) -> str:
    if isinstance(path, Path):
        return path.as_posix()
    return path
