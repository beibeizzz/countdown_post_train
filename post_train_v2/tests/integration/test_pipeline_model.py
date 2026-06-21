from __future__ import annotations

from pathlib import Path

import pytest

from post_train_v2.src.pipeline.model import (
    PIPELINE_STAGE_ORDER,
    StageSpec,
    dependency_closure,
    select_stage_window,
    topological_stage_order,
    validate_stage_specs,
)


EXPECTED_ORDER = [
    "build_source",
    "validation_split",
    "teacher_pool",
    "accepted_splits",
    "full_sft",
    "lora_sft",
    "rft_data",
    "rft_train",
    "dpo_data",
    "dpo_train",
    "grpo_convert",
    "grpo_train",
    "grpo_export",
    "final_eval",
]


def test_pipeline_stage_order_is_frozen() -> None:
    assert list(PIPELINE_STAGE_ORDER) == EXPECTED_ORDER
    specs = _specs()
    assert topological_stage_order(specs) == EXPECTED_ORDER


def test_dependency_closure_and_stage_window_selection() -> None:
    specs = _specs()

    assert dependency_closure(specs, "dpo_train") == [
        "build_source",
        "validation_split",
        "teacher_pool",
        "accepted_splits",
        "full_sft",
        "dpo_data",
        "dpo_train",
    ]
    assert select_stage_window(specs, from_stage="dpo_data", through_stage="grpo_train") == [
        "dpo_data",
        "dpo_train",
        "grpo_convert",
        "grpo_train",
    ]


def test_pipeline_model_rejects_cycles_unknown_dependencies_and_bad_windows() -> None:
    specs = _specs()
    cyclic = dict(specs)
    cyclic["build_source"] = StageSpec(
        name="build_source",
        command=("python", "build.py"),
        dependencies=("final_eval",),
        manifest_path=Path("out/build_source_manifest.json"),
        resources="cpu",
    )
    with pytest.raises(ValueError, match="cycle"):
        validate_stage_specs(cyclic.values())

    unknown = [
        StageSpec(
            name="stage",
            command=("python", "stage.py"),
            dependencies=("missing",),
            manifest_path=Path("out/stage_manifest.json"),
            resources="cpu",
        )
    ]
    with pytest.raises(ValueError, match="unknown dependency"):
        validate_stage_specs(unknown)

    with pytest.raises(ValueError, match="from-stage"):
        select_stage_window(specs, from_stage="grpo_train", through_stage="dpo_data")


def _specs() -> dict[str, StageSpec]:
    deps = {
        "build_source": (),
        "validation_split": ("build_source",),
        "teacher_pool": ("validation_split",),
        "accepted_splits": ("teacher_pool",),
        "full_sft": ("accepted_splits",),
        "lora_sft": ("accepted_splits",),
        "rft_data": ("full_sft",),
        "rft_train": ("rft_data",),
        "dpo_data": ("full_sft",),
        "dpo_train": ("dpo_data",),
        "grpo_convert": ("full_sft",),
        "grpo_train": ("grpo_convert",),
        "grpo_export": ("grpo_train",),
        "final_eval": ("full_sft", "lora_sft", "rft_train", "dpo_train", "grpo_export"),
    }
    return {
        name: StageSpec(
            name=name,
            command=("python", f"{name}.py"),
            dependencies=deps[name],
            manifest_path=Path(f"outputs/{name}_manifest.json"),
            resources="cpu",
        )
        for name in EXPECTED_ORDER
    }
