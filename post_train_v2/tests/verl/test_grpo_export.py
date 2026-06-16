from __future__ import annotations

import json
from pathlib import Path

import pytest

from post_train_v2.verl.export.merge_actor import (
    build_model_merger_command,
    export_grpo_actors,
)


def test_build_model_merger_command_uses_stock_verl_arguments() -> None:
    command = build_model_merger_command(
        local_dir=Path("/tmp/verl-run/global_step_100/actor"),
        target_dir=Path("post_train_v2/outputs/grpo/run/best"),
    )

    assert command == [
        "python",
        "-m",
        "verl.model_merger",
        "merge",
        "--backend",
        "fsdp",
        "--local_dir",
        "/tmp/verl-run/global_step_100/actor",
        "--target_dir",
        "post_train_v2/outputs/grpo/run/best",
    ]


def test_export_grpo_actors_checks_help_merges_and_prunes_after_acceptance(
    tmp_path: Path,
) -> None:
    run_dir = _write_run(tmp_path)
    calls: list[list[str]] = []
    checks: list[Path] = []

    def runner(command: list[str]) -> int:
        calls.append(command)
        if "merge" in command:
            target = Path(command[command.index("--target_dir") + 1])
            target.mkdir(parents=True)
            (target / "config.json").write_text("{}", encoding="utf-8")
        return 0

    def direct_load_check(path: Path) -> None:
        assert not (path / "export_manifest.json").exists()
        checks.append(path)

    summary = export_grpo_actors(
        run_dir,
        direct_load_check=direct_load_check,
        runner=runner,
        prune=True,
    )

    assert calls[0] == ["python", "-m", "verl.model_merger", "--help"]
    assert calls[1] == build_model_merger_command(
        local_dir=run_dir / "checkpoints/global_step_100/actor",
        target_dir=run_dir / "export/best",
    )
    assert calls[2] == build_model_merger_command(
        local_dir=run_dir / "checkpoints/global_step_300/actor",
        target_dir=run_dir / "export/final",
    )
    assert checks == [run_dir / "export/best", run_dir / "export/final"]
    assert summary["best"]["direct_loadable"] is True
    assert json.loads((run_dir / "export/best/export_manifest.json").read_text())[
        "source_step"
    ] == 100
    assert not (run_dir / "checkpoints/global_step_50").exists()
    assert (run_dir / "checkpoints/global_step_100").exists()
    assert (run_dir / "checkpoints/global_step_200").exists()
    assert (run_dir / "checkpoints/global_step_300").exists()


def test_export_grpo_actors_does_not_prune_if_direct_load_fails(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)

    def runner(command: list[str]) -> int:
        if "merge" in command:
            target = Path(command[command.index("--target_dir") + 1])
            target.mkdir(parents=True)
        return 0

    def direct_load_check(path: Path) -> None:
        if path.name == "final":
            raise RuntimeError("not loadable")

    with pytest.raises(RuntimeError, match="not loadable"):
        export_grpo_actors(
            run_dir,
            direct_load_check=direct_load_check,
            runner=runner,
            prune=True,
        )

    assert (run_dir / "checkpoints/global_step_50").exists()


def _write_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    for step in (50, 100, 200, 300):
        (run_dir / "checkpoints" / f"global_step_{step}" / "actor").mkdir(
            parents=True
        )
    selection = {
        "selected_best_step": 100,
        "final_step": 300,
        "best_checkpoint_path": str(run_dir / "checkpoints/global_step_100/actor"),
        "final_checkpoint_path": str(run_dir / "checkpoints/global_step_300/actor"),
    }
    (run_dir / "export").mkdir(parents=True)
    (run_dir / "export" / "selection.json").write_text(
        json.dumps(selection),
        encoding="utf-8",
    )
    return run_dir
