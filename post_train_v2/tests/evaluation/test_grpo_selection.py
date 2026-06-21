from __future__ import annotations

import json
from pathlib import Path

import pytest

from post_train_v2.src.artifacts.hashing import sha256_canonical_json, sha256_file
from post_train_v2.verl.export.select_checkpoint import select_grpo_checkpoints


def test_select_grpo_checkpoint_rescores_dumps_and_ties_on_earlier_step(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _write_step(run_dir, 50, ["<answer>1+2+3</answer>", "<answer>1+2-3</answer>"])
    _write_step(run_dir, 100, ["<answer>1+2+3</answer>", "<answer>1+2-3</answer>"])
    _write_step(run_dir, 200, ["<answer>1+2+3</answer>", "missing tag"])

    selection = select_grpo_checkpoints(run_dir, config={"seed": 7})

    assert selection["selected_best_step"] == 50
    assert selection["final_step"] == 200
    assert selection["config_sha256"] == sha256_canonical_json({"seed": 7})
    assert Path(selection["best_checkpoint_path"]).name == "actor"
    assert Path(selection["final_checkpoint_path"]).name == "actor"

    by_step = {candidate["step"]: candidate for candidate in selection["candidates"]}
    assert by_step[50]["metrics"]["accuracy"] == 0.5
    assert by_step[50]["metrics"]["format_rate"] == 1.0
    assert by_step[200]["metrics"]["accuracy"] == 0.5
    assert by_step[200]["metrics"]["format_rate"] == 0.5
    assert by_step[50]["validation_dump_sha256"] == sha256_file(
        run_dir / "validation" / "step_50.jsonl"
    )

    written = json.loads((run_dir / "export" / "selection.json").read_text())
    assert written == selection


def test_select_grpo_checkpoint_requires_dump_checkpoint_pairs(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_step(run_dir, 100, ["<answer>1+2+3</answer>", "<answer>1+2-3</answer>"])
    _write_dump(run_dir / "validation" / "step_200.jsonl", ["<answer>1+2+3</answer>"])

    with pytest.raises(ValueError, match="missing checkpoint"):
        select_grpo_checkpoints(run_dir, config={})


def _write_step(run_dir: Path, step: int, generations: list[str]) -> None:
    (run_dir / "checkpoints" / f"global_step_{step}" / "actor").mkdir(parents=True)
    _write_dump(run_dir / "validation" / f"step_{step}.jsonl", generations)


def _write_dump(path: Path, generations: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, generation in enumerate(generations):
        rows.append(
            {
                "id": f"row-{index}",
                "prompt": "solve",
                "numbers": [1, 2, 3],
                "target": 6,
                "raw_generation": generation,
                "generated_tokens": 8 + index,
                "truncated": False,
            }
        )
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
