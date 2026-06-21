"""Static V2 pipeline DAG model."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ResourceKind = Literal["cpu", "gpu2_vllm", "gpu2_ddp", "gpu2_verl"]

PIPELINE_STAGE_ORDER: tuple[str, ...] = (
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
)


@dataclass(frozen=True)
class StageSpec:
    name: str
    command: tuple[str, ...]
    dependencies: tuple[str, ...]
    manifest_path: Path
    resources: ResourceKind

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("stage name must be a non-empty string")
        if not self.command or any(not isinstance(item, str) or not item for item in self.command):
            raise ValueError(f"stage {self.name} command must contain non-empty strings")
        if any(not isinstance(item, str) or not item for item in self.dependencies):
            raise ValueError(f"stage {self.name} dependencies must be non-empty strings")
        if self.resources not in {"cpu", "gpu2_vllm", "gpu2_ddp", "gpu2_verl"}:
            raise ValueError(f"stage {self.name} has invalid resources")


def validate_stage_specs(stages: Iterable[StageSpec]) -> dict[str, StageSpec]:
    stage_list = list(stages)
    by_name = {stage.name: stage for stage in stage_list}
    if len(by_name) != len(stage_list):
        raise ValueError("duplicate stage name")
    for stage in by_name.values():
        for dependency in stage.dependencies:
            if dependency not in by_name:
                raise ValueError(f"unknown dependency {dependency} for stage {stage.name}")
    _topological_names(by_name)
    return by_name


def topological_stage_order(stages: Mapping[str, StageSpec] | Iterable[StageSpec]) -> list[str]:
    by_name = _coerce_specs(stages)
    names = _topological_names(by_name)
    return [name for name in PIPELINE_STAGE_ORDER if name in names]


def dependency_closure(
    stages: Mapping[str, StageSpec] | Iterable[StageSpec],
    stage_name: str,
) -> list[str]:
    by_name = _coerce_specs(stages)
    if stage_name not in by_name:
        raise ValueError(f"unknown stage: {stage_name}")
    required: set[str] = set()

    def visit(name: str) -> None:
        if name in required:
            return
        for dependency in by_name[name].dependencies:
            visit(dependency)
        required.add(name)

    visit(stage_name)
    return [name for name in PIPELINE_STAGE_ORDER if name in required]


def select_stage_window(
    stages: Mapping[str, StageSpec] | Iterable[StageSpec],
    *,
    from_stage: str | None = None,
    through_stage: str | None = None,
) -> list[str]:
    ordered = topological_stage_order(stages)
    start = 0 if from_stage is None else _stage_index(ordered, from_stage, "from-stage")
    end = (
        len(ordered) - 1
        if through_stage is None
        else _stage_index(ordered, through_stage, "through-stage")
    )
    if start > end:
        raise ValueError("from-stage must not be after through-stage")
    return ordered[start : end + 1]


def _coerce_specs(stages: Mapping[str, StageSpec] | Iterable[StageSpec]) -> dict[str, StageSpec]:
    if isinstance(stages, Mapping):
        return validate_stage_specs(stages.values())
    return validate_stage_specs(stages)


def _stage_index(ordered: list[str], name: str, label: str) -> int:
    try:
        return ordered.index(name)
    except ValueError as error:
        raise ValueError(f"unknown {label}: {name}") from error


def _topological_names(by_name: Mapping[str, StageSpec]) -> list[str]:
    temporary: set[str] = set()
    permanent: set[str] = set()
    ordered: list[str] = []

    def visit(name: str) -> None:
        if name in permanent:
            return
        if name in temporary:
            raise ValueError("cycle detected in pipeline DAG")
        temporary.add(name)
        for dependency in by_name[name].dependencies:
            visit(dependency)
        temporary.remove(name)
        permanent.add(name)
        ordered.append(name)

    for name in PIPELINE_STAGE_ORDER:
        if name in by_name:
            visit(name)
    for name in sorted(set(by_name) - set(PIPELINE_STAGE_ORDER)):
        visit(name)
    return ordered
