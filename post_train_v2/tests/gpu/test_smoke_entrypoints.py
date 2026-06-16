from __future__ import annotations

from pathlib import Path

from post_train_v2.scripts.env.smoke_v2_training import build_smoke_commands


def test_smoke_commands_use_expected_launchers_and_two_visible_gpus(tmp_path: Path) -> None:
    commands = build_smoke_commands(through_stage="grpo_export", work_dir=tmp_path)
    by_stage = {command.stage: command for command in commands}

    assert by_stage["teacher_pool"].command[0] == "python"
    assert by_stage["teacher_pool"].env["CUDA_VISIBLE_DEVICES"] == "0,1"

    for stage in ("full_sft", "lora_sft", "rft_train", "dpo_train"):
        assert by_stage[stage].command[:3] == [
            "torchrun",
            "--standalone",
            "--nproc_per_node=2",
        ]
        assert by_stage[stage].env["CUDA_VISIBLE_DEVICES"] == "0,1"

    assert by_stage["grpo_train"].command[:2] == ["python", "post_train_v2/scripts/grpo/train_grpo.py"]
    assert by_stage["grpo_train"].env["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert str(tmp_path) in " ".join(by_stage["grpo_export"].command)


def test_smoke_command_selection_stops_at_requested_stage(tmp_path: Path) -> None:
    commands = build_smoke_commands(through_stage="dpo_data", work_dir=tmp_path)

    assert [command.stage for command in commands][-1] == "dpo_data"
    assert "dpo_train" not in {command.stage for command in commands}
    assert all("post_train_v2/outputs/" not in " ".join(command.command) for command in commands)
