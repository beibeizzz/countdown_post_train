from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import math
import os
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from post_train.src.countdown.config import load_yaml_config
from post_train.src.countdown.io import read_jsonl, write_jsonl, write_manifest
from post_train_v2.src.generation.parallel_vllm import WorkerReady
from post_train_v2.src.generation.seeding import derive_request_seed
from post_train_v2.src.artifacts.hashing import sha256_file
from post_train_v2.src.artifacts.manifest import (
    ArtifactFile,
    ManifestV2,
    publish_manifest,
)
from post_train_v2.src.countdown.bucketing import assign_bucket
from post_train_v2.src.countdown.prompts import build_solution_prompt
from post_train_v2.src.data.schema import (
    validate_normalized_source,
    validate_sft_record,
)
from post_train_v2.src.generation.teacher_state import (
    ResumeState,
    TeacherGenerationConfig,
    TeacherStateStore,
)
from post_train_v2.scripts.generation.build_teacher_pool import (
    PRODUCTION_CONFIG,
    SMOKE_CONFIG,
    _parse_jsonl_bytes,
    build_teacher_payload,
    load_teacher_config,
    run,
    validate_cuda_visibility,
    validate_source_rows,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def source_row(
    row_id: object = "row-0",
    *,
    numbers: list[int] | None = None,
    target: int = 3,
    prompt: str = "Solve it",
) -> dict:
    return {
        "id": row_id,
        "numbers": [1, 2] if numbers is None else numbers,
        "target": target,
        "prompt": prompt,
    }


def make_config(tmp_path: Path, **overrides) -> TeacherGenerationConfig:
    model_path = tmp_path / "model"
    model_path.mkdir(exist_ok=True)
    input_path = tmp_path / "source.jsonl"
    if not input_path.exists():
        write_jsonl(input_path, [source_row()])
    values = {
        "model_path": model_path,
        "input_path": input_path,
        "output_dir": tmp_path / "output",
        "devices": (0, 1),
        "topology": "dual_tp1",
        "batch_size": 64,
        "worker_timeout_seconds": 600.0,
        "gpu_memory_utilization": 0.8,
        "max_model_len": 512,
        "max_new_tokens": 256,
        "temperature": 0.2,
        "top_p": 0.95,
        "seed": 0,
        "enable_thinking": False,
        "stop_after_accepted": 2,
        "cache_root": tmp_path / "cache",
        "schema_version": 1,
    }
    values.update(overrides)
    return TeacherGenerationConfig(**values)


def write_config(path: Path, config: TeacherGenerationConfig) -> None:
    path.write_text(
        "\n".join(
            (
                f"model_path: {config.model_path}",
                f"input_path: {config.input_path}",
                f"output_dir: {config.output_dir}",
                f"devices: [{config.devices[0]}, {config.devices[1]}]",
                f"topology: {config.topology}",
                f"batch_size: {config.batch_size}",
                f"worker_timeout_seconds: {config.worker_timeout_seconds}",
                f"gpu_memory_utilization: {config.gpu_memory_utilization}",
                f"max_model_len: {config.max_model_len}",
                f"max_new_tokens: {config.max_new_tokens}",
                f"temperature: {config.temperature}",
                f"top_p: {config.top_p}",
                f"seed: {config.seed}",
                f"enable_thinking: {str(config.enable_thinking).lower()}",
                f"stop_after_accepted: {config.stop_after_accepted}",
                f"cache_root: {config.cache_root}",
                f"schema_version: {config.schema_version}",
            )
        )
        + "\n",
        encoding="utf-8",
    )


def test_production_and_smoke_configs_have_exact_approved_values() -> None:
    production = load_yaml_config(PRODUCTION_CONFIG)
    smoke = load_yaml_config(SMOKE_CONFIG)
    expected = {
        "model_path": "post_train/model/qwen/qwen3-8b",
        "input_path": "post_train_v2/data/processed/train_candidates.jsonl",
        "output_dir": "post_train_v2/data/teacher_rollouts",
        "devices": [0, 1],
        "topology": "dual_tp1",
        "batch_size": 64,
        "worker_timeout_seconds": 600,
        "gpu_memory_utilization": 0.8,
        "max_model_len": 512,
        "max_new_tokens": 256,
        "temperature": 0.2,
        "top_p": 0.95,
        "seed": 0,
        "enable_thinking": False,
        "stop_after_accepted": 20_000,
        "cache_root": "/tmp/countdown_teacher_vllm",
        "schema_version": 2,
    }

    assert production == expected
    assert smoke == {
        **expected,
        "input_path": "post_train_v2/data/fixtures/teacher_smoke_candidates.jsonl",
        "output_dir": "/tmp/post_train_v2_teacher_smoke",
        "stop_after_accepted": 8,
        "cache_root": "/tmp/countdown_teacher_vllm_smoke",
    }
    fixture_path = REPO_ROOT / smoke["input_path"]
    fixture_rows = read_jsonl(fixture_path)
    assert len(fixture_rows) == 8
    assert [
        validate_normalized_source(row) for row in fixture_rows
    ] == fixture_rows


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        (lambda cfg: replace(cfg, cache_root=Path("relative")), "cache_root.*absolute"),
        (lambda cfg: replace(cfg, devices=(0, 0)), "distinct"),
        (lambda cfg: replace(cfg, topology="tp2"), "dual_tp1"),
        (lambda cfg: replace(cfg, batch_size=0), "batch_size"),
        (lambda cfg: replace(cfg, enable_thinking=True), "enable_thinking"),
        (lambda cfg: replace(cfg, model_path=Path("missing")), "model_path"),
        (lambda cfg: replace(cfg, input_path=Path("missing")), "input_path"),
    ),
)
def test_config_rejects_invalid_fields_paths_and_relative_cache(
    tmp_path: Path, mutation, match: str
) -> None:
    config = mutation(make_config(tmp_path))
    config_path = tmp_path / "config.yaml"
    write_config(config_path, config)

    with pytest.raises(ValueError, match=match):
        load_teacher_config(config_path, repo_root=tmp_path)


@pytest.mark.parametrize("visible", (None, "", "  ", "0,1", "0, 1"))
def test_visibility_accepts_unset_blank_or_exact_order(
    tmp_path: Path, visible: str | None
) -> None:
    validate_cuda_visibility(make_config(tmp_path), visible)


@pytest.mark.parametrize("visible", ("0", "1,0", "0,0", "GPU-a,GPU-b", "0,1,2"))
def test_visibility_rejects_missing_reordered_duplicate_or_ambiguous_masks(
    tmp_path: Path, visible: str
) -> None:
    with pytest.raises(ValueError, match="exact ordered list.*0,1"):
        validate_cuda_visibility(make_config(tmp_path), visible)


@pytest.mark.parametrize(
    ("rows", "match"),
    (
        ([None], "object"),
        ([source_row("")], "id.*nonempty"),
        ([source_row("same"), source_row("same")], "duplicate.*same"),
        ([source_row(prompt=" ")], "prompt.*nonempty"),
        ([source_row(numbers=(1, 2))], "numbers.*list"),
        ([source_row(numbers=[1, True])], "numbers.*exact integers"),
        ([source_row(target=True)], "target.*exact integer"),
    ),
)
def test_validate_source_rows_rejects_malformed_rows(rows, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        validate_source_rows(rows)


def test_local_jsonl_parser_matches_file_iteration_for_unicode_separator(
    tmp_path: Path,
) -> None:
    path = tmp_path / "source.jsonl"
    data = (
        '{"id":"a\u2028b","numbers":[1,2],"target":3,"prompt":"solve"}\n'
    ).encode("utf-8")

    assert _parse_jsonl_bytes(data, path) == [
        {
            "id": "a\u2028b",
            "numbers": [1, 2],
            "target": 3,
            "prompt": "solve",
        }
    ]


def test_payload_retains_full_stripped_response_and_validation() -> None:
    payload = build_teacher_payload(
        source_row(numbers=[7, 3, 8, 2], target=24),
        " reasoning\n<answer> (7-3)*(8-2) </answer>\n ",
    )

    assert payload["response"] == "reasoning\n<answer> (7-3)*(8-2) </answer>"
    assert payload["validation"] == {
        "ok": True,
        "error": None,
        "value": "24/1",
        "used_numbers": [7, 3, 8, 2],
        "expression": "(7-3)*(8-2)",
    }
    assert payload["provenance"] == {
        "generator": "qwen3-8b-teacher",
        "stage": "teacher",
        "rollout_index": 0,
    }
    empty = build_teacher_payload(source_row(), "")
    assert empty["validation"] == {
        "ok": False,
        "error": "missing_answer_tag",
        "value": None,
        "used_numbers": [],
        "expression": None,
    }


def test_payload_from_canonical_source_is_a_valid_sft_record() -> None:
    numbers = [7, 3, 8, 2]
    gold_expr = "(7-3)*(8-2)"
    source = {
        "id": "train-000001",
        "source_index": 1,
        "numbers": numbers,
        "target": 24,
        "gold_expr": gold_expr,
        "prompt": build_solution_prompt(numbers, 24),
        "bucket": assign_bucket(numbers, gold_expr),
    }

    payload = build_teacher_payload(
        source,
        "Reasoning.\n<answer>(7-3)*(8-2)</answer>",
    )

    assert validate_sft_record(payload) == payload


class FakeLock:
    def __init__(
        self,
        events: list,
        *,
        release_error: BaseException | None = None,
        recovered_stale: bool = False,
        **kwargs,
    ) -> None:
        self.events = events
        self.kwargs = kwargs
        self.path = kwargs.get("path", Path("fake.lock"))
        self.release_error = release_error
        self.recovered_stale = recovered_stale
        self.released = False

    def acquire(self, recover_stale: bool = False) -> None:
        self.events.append(("lock.acquire", recover_stale))

    def release(self) -> None:
        self.released = True
        self.events.append(("lock.release",))
        if self.release_error is not None:
            raise self.release_error


class FakeStore:
    def __init__(
        self,
        events: list,
        state: ResumeState,
        *,
        commit_error: Exception | None = None,
        has_v2_manifest: bool = False,
        load_hook=None,
    ) -> None:
        self.events = events
        self.state = state
        self.commit_error = commit_error
        self._has_v2_manifest = has_v2_manifest
        self.load_hook = load_hook
        self.commits: list[dict] = []

    def has_v2_manifest(self) -> bool:
        self.events.append(("store.has_v2_manifest",))
        return self._has_v2_manifest

    def load_resume_state(self, rows, config, adopt_legacy_state=False):
        self.events.append(("store.load", adopt_legacy_state))
        if self.load_hook is not None:
            self.load_hook()
        return self.state

    def commit(self, **kwargs) -> None:
        self.events.append(("store.commit", kwargs["batch_id"]))
        if self.commit_error is not None:
            raise self.commit_error
        self.commits.append(kwargs)
        self._has_v2_manifest = True


class FakeEngine:
    AUTO = object()

    def __init__(
        self,
        events: list,
        responses,
        *,
        generate_error=None,
        close_error: BaseException | None = None,
        worker_latencies: tuple[float, float] = (0.01, 0.02),
        worker_runtime_info=AUTO,
        worker_result_counts=AUTO,
        worker_nonempty_counts=AUTO,
    ) -> None:
        self.events = events
        self.responses = list(responses)
        self.generate_error = generate_error
        self.close_error = close_error
        self.calls: list[tuple[int, tuple]] = []
        self.closed = False
        self.worker_exitcodes = (0, 0)
        self.last_worker_latencies = worker_latencies
        self.worker_runtime_info = (
            (
                WorkerReady(0, 1000, "0", "/cache/gpu0"),
                WorkerReady(1, 1001, "1", "/cache/gpu1"),
            )
            if worker_runtime_info is self.AUTO
            else worker_runtime_info
        )
        self._worker_result_counts = worker_result_counts
        self._worker_nonempty_counts = worker_nonempty_counts
        self.last_worker_result_counts = None
        self.last_worker_nonempty_counts = None

    def start(self):
        self.events.append(("engine.start",))
        return self

    def generate(self, batch_id, items):
        self.calls.append((batch_id, tuple(items)))
        self.events.append(("engine.generate", batch_id))
        if self.generate_error is not None:
            raise self.generate_error
        response = self.responses.pop(0)
        shard_stop = (len(items) + 1) // 2
        self.last_worker_result_counts = (
            (shard_stop, len(items) - shard_stop)
            if self._worker_result_counts is self.AUTO
            else self._worker_result_counts
        )
        self.last_worker_nonempty_counts = (
            (
                sum(bool(text.strip()) for _, text in response[:shard_stop]),
                sum(bool(text.strip()) for _, text in response[shard_stop:]),
            )
            if self._worker_nonempty_counts is self.AUTO
            else self._worker_nonempty_counts
        )
        return response

    def close(self) -> None:
        self.closed = True
        self.events.append(("engine.close",))
        if self.close_error is not None:
            raise self.close_error


class RecordingTeacherStateStore(TeacherStateStore):
    def __init__(self, output_dir: Path) -> None:
        super().__init__(output_dir)
        self.commit_calls: list[dict] = []

    def commit(self, **kwargs) -> None:
        self.commit_calls.append(kwargs)
        super().commit(**kwargs)


def write_legacy_outputs(
    config: TeacherGenerationConfig,
    accepted: list[dict],
    rejected: list[dict],
) -> RecordingTeacherStateStore:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(config.output_dir / "teacher_accepted_20k.jsonl", accepted)
    write_jsonl(config.output_dir / "teacher_rejected.jsonl", rejected)
    return RecordingTeacherStateStore(config.output_dir)


def write_builder_legacy_manifest(
    store: TeacherStateStore,
    config: TeacherGenerationConfig,
    accepted: list[dict],
    rejected: list[dict],
) -> None:
    write_manifest(
        store.manifest_path,
        {
            "name": "teacher_accepted_pool",
            "model": str(config.model_path.resolve()),
            "num_accepted": len(accepted),
            "num_rejected": len(rejected),
            "max_new_tokens": config.max_new_tokens,
            "enable_thinking": config.enable_thinking,
        },
    )


def run_with_fakes(
    tmp_path: Path,
    rows: list[dict],
    responses,
    *,
    target: int = 2,
    state: ResumeState | None = None,
    has_v2_manifest: bool = False,
    engine_error: BaseException | None = None,
    commit_error: Exception | None = None,
    recover_stale: bool = False,
    adopt_legacy: bool = False,
    engine_options: dict | None = None,
):
    config = make_config(tmp_path, stop_after_accepted=target)
    write_jsonl(config.input_path, rows)
    config_path = tmp_path / "config.yaml"
    write_config(config_path, config)
    events: list = []
    state = state or ResumeState((), (), 0, None, "2026-06-14T00:00:00+00:00")
    store = FakeStore(
        events,
        state,
        commit_error=commit_error,
        has_v2_manifest=has_v2_manifest,
    )
    lock_holder = {}
    engine_holder = {}
    engine_kwargs = {}

    def lock_factory(**kwargs):
        lock = FakeLock(events, **kwargs)
        lock_holder["lock"] = lock
        return lock

    def store_factory(output_dir):
        assert Path(output_dir) == config.output_dir
        return store

    def engine_factory(**kwargs):
        engine_kwargs.update(kwargs)
        engine = FakeEngine(
            events,
            responses,
            generate_error=engine_error,
            **(engine_options or {}),
        )
        engine_holder["engine"] = engine
        return engine

    result = run(
        config_path,
        recover_stale_lock=recover_stale,
        adopt_legacy_state=adopt_legacy,
        engine_factory=engine_factory,
        state_store_factory=store_factory,
        lock_factory=lock_factory,
        now=lambda: datetime(2026, 6, 14, tzinfo=timezone.utc),
        cuda_visible_devices=None,
    )
    return result, store, lock_holder, engine_holder, engine_kwargs, events


def test_adopted_nonempty_state_materializes_v2_then_continues_with_batch_gt_zero(
    tmp_path: Path,
) -> None:
    rows = [source_row("a"), source_row("b")]
    config = make_config(tmp_path, stop_after_accepted=2)
    write_jsonl(config.input_path, rows)
    config_path = tmp_path / "config.yaml"
    write_config(config_path, config)
    accepted = [build_teacher_payload(rows[0], "<answer>1+2</answer>")]
    config.output_dir.mkdir(parents=True)
    store = RecordingTeacherStateStore(config.output_dir)
    write_jsonl(store.accepted_path, accepted)
    write_builder_legacy_manifest(store, config, accepted, [])
    engine = FakeEngine([], [[(1, "<answer>1+2</answer>")]])

    code = run(
        config_path,
        adopt_legacy_state=True,
        engine_factory=lambda **kwargs: engine,
        state_store_factory=lambda output_dir: store,
        now=lambda: datetime(2026, 6, 14, tzinfo=timezone.utc),
        cuda_visible_devices=None,
    )

    assert code == 0
    assert [(call["batch_id"], call["submitted_start"], call["submitted_stop"]) for call in store.commit_calls] == [
        (0, 1, 1),
        (1, 1, 2),
    ]
    assert store.has_v2_manifest() is True
    assert [row["id"] for row in read_jsonl(store.accepted_path)] == ["a", "b"]


def test_adopted_already_complete_state_gets_v2_manifest_without_engine(
    tmp_path: Path,
) -> None:
    row = source_row("a")
    config = make_config(tmp_path, stop_after_accepted=1)
    write_jsonl(config.input_path, [row])
    config_path = tmp_path / "config.yaml"
    write_config(config_path, config)
    accepted = [build_teacher_payload(row, "<answer>1+2</answer>")]
    store = write_legacy_outputs(
        config,
        accepted,
        [],
    )
    write_builder_legacy_manifest(store, config, accepted, [])
    constructed = []

    assert run(
        config_path,
        adopt_legacy_state=True,
        engine_factory=lambda **kwargs: constructed.append(kwargs),
        state_store_factory=lambda output_dir: store,
        now=lambda: datetime(2026, 6, 14, tzinfo=timezone.utc),
        cuda_visible_devices=None,
    ) == 0

    assert constructed == []
    assert len(store.commit_calls) == 1
    assert store.commit_calls[0]["batch_id"] == 0
    assert store.commit_calls[0]["submitted_start"] == 1
    assert store.commit_calls[0]["manifest"]["completed"] is True
    assert store.has_v2_manifest() is True


def test_adopted_exhausted_state_gets_incomplete_v2_manifest(
    tmp_path: Path,
) -> None:
    rows = [source_row("a", target=99), source_row("b", target=99)]
    config = make_config(tmp_path, stop_after_accepted=2)
    write_jsonl(config.input_path, rows)
    config_path = tmp_path / "config.yaml"
    write_config(config_path, config)
    accepted = []
    rejected = [
        build_teacher_payload(rows[0], ""),
        build_teacher_payload(rows[1], ""),
    ]
    config.output_dir.mkdir(parents=True)
    store = RecordingTeacherStateStore(config.output_dir)
    write_jsonl(store.rejected_path, rejected)
    write_builder_legacy_manifest(store, config, accepted, rejected)

    assert run(
        config_path,
        adopt_legacy_state=True,
        engine_factory=lambda **kwargs: pytest.fail("engine constructed"),
        state_store_factory=lambda output_dir: store,
        now=lambda: datetime(2026, 6, 14, tzinfo=timezone.utc),
        cuda_visible_devices=None,
    ) == 2

    assert len(store.commit_calls) == 1
    assert store.commit_calls[0]["submitted_start"] == 2
    assert store.commit_calls[0]["submitted_stop"] == 2
    assert store.commit_calls[0]["manifest"]["completed"] is False
    assert store.has_v2_manifest() is True


def test_adopted_source_field_poisoning_fails_before_engine(
    tmp_path: Path,
) -> None:
    row = source_row("a")
    config = make_config(tmp_path, stop_after_accepted=2)
    write_jsonl(config.input_path, [row])
    config_path = tmp_path / "config.yaml"
    write_config(config_path, config)
    poisoned = build_teacher_payload(row, "<answer>1+2</answer>")
    poisoned["prompt"] = "poisoned"
    config.output_dir.mkdir(parents=True)
    store = RecordingTeacherStateStore(config.output_dir)
    write_jsonl(store.accepted_path, [poisoned])
    write_builder_legacy_manifest(store, config, [poisoned], [])
    constructed = []

    with pytest.raises(ValueError, match="prompt"):
        run(
            config_path,
            adopt_legacy_state=True,
            engine_factory=lambda **kwargs: constructed.append(kwargs),
            state_store_factory=lambda output_dir: store,
            cuda_visible_devices=None,
        )

    assert constructed == []
    assert store.commit_calls == []


def test_preexisting_empty_snapshots_get_v2_manifest_before_exhaustion_return(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path, stop_after_accepted=1)
    config.input_path.write_bytes(b"")
    config_path = tmp_path / "config.yaml"
    write_config(config_path, config)
    store = write_legacy_outputs(config, [], [])

    assert run(
        config_path,
        engine_factory=lambda **kwargs: pytest.fail("engine constructed"),
        state_store_factory=lambda output_dir: store,
        now=lambda: datetime(2026, 6, 14, tzinfo=timezone.utc),
        cuda_visible_devices=None,
    ) == 2

    assert len(store.commit_calls) == 1
    assert store.commit_calls[0]["batch_id"] == 0
    assert store.commit_calls[0]["submitted_start"] == 0
    assert store.has_v2_manifest() is True


def test_source_hash_uses_exact_bytes_parsed_under_lock(tmp_path: Path) -> None:
    row = source_row("a")
    source_bytes = (
        json.dumps(row, separators=(", ", ": ")).encode("utf-8") + b"\r\n"
    )
    config = make_config(tmp_path, stop_after_accepted=1)
    config.input_path.write_bytes(source_bytes)
    config_path = tmp_path / "config.yaml"
    write_config(config_path, config)
    store = write_legacy_outputs(
        config,
        [build_teacher_payload(row, "<answer>1+2</answer>")],
        [],
    )

    assert run(
        config_path,
        adopt_legacy_state=True,
        engine_factory=lambda **kwargs: pytest.fail("engine constructed"),
        state_store_factory=lambda output_dir: store,
        now=lambda: datetime(2026, 6, 14, tzinfo=timezone.utc),
        cuda_visible_devices=None,
    ) == 0

    manifest = ManifestV2.parse(
        json.loads(store.manifest_path.read_text(encoding="utf-8"))
    )
    assert (
        manifest.stage_metadata["teacher_state"]["source_sha256"]
        == hashlib.sha256(source_bytes).hexdigest()
    )


def test_source_mutation_during_state_load_fails_before_engine(tmp_path: Path) -> None:
    rows = [source_row("a")]
    config = make_config(tmp_path)
    write_jsonl(config.input_path, rows)
    config_path = tmp_path / "config.yaml"
    write_config(config_path, config)
    events: list = []
    store = FakeStore(
        events,
        ResumeState((), (), 0, None, "2026-06-14T00:00:00+00:00"),
        has_v2_manifest=True,
        load_hook=lambda: write_jsonl(config.input_path, [source_row("changed")]),
    )
    constructed = []

    with pytest.raises(ValueError, match="source.*changed"):
        run(
            config_path,
            engine_factory=lambda **kwargs: constructed.append(kwargs),
            state_store_factory=lambda output_dir: store,
            lock_factory=lambda **kwargs: FakeLock(events, **kwargs),
            cuda_visible_devices=None,
        )

    assert constructed == []


def test_source_mutation_immediately_before_state_load_fails_before_engine(
    tmp_path: Path,
) -> None:
    rows = [source_row("a")]
    config = make_config(tmp_path)
    write_jsonl(config.input_path, rows)
    config_path = tmp_path / "config.yaml"
    write_config(config_path, config)
    events: list = []
    store = FakeStore(
        events,
        ResumeState((), (), 0, None, "2026-06-14T00:00:00+00:00"),
        has_v2_manifest=True,
    )
    constructed = []

    def store_factory(output_dir):
        write_jsonl(config.input_path, [source_row("b")])
        return store

    with pytest.raises(ValueError, match="source.*changed"):
        run(
            config_path,
            engine_factory=lambda **kwargs: constructed.append(kwargs),
            state_store_factory=store_factory,
            lock_factory=lambda **kwargs: FakeLock(events, **kwargs),
            cuda_visible_devices=None,
        )

    assert constructed == []
    assert ("store.load", False) not in events


def test_coordinator_orders_payloads_stops_exactly_midbatch_and_commits_once(
    tmp_path: Path,
) -> None:
    rows = [
        source_row("a"),
        source_row("b", target=99),
        source_row("c"),
        source_row("d"),
    ]
    positioned = [
        (0, " <answer>1+2</answer> "),
        (1, ""),
        (2, "<answer>1+2</answer>"),
        (3, "<answer>1+2</answer>"),
    ]

    code, store, _, engine_holder, _, _ = run_with_fakes(
        tmp_path, rows, [positioned], target=2
    )

    assert code == 0
    assert len(store.commits) == 2
    initial, batch = store.commits
    assert (initial["batch_id"], initial["submitted_start"], initial["submitted_stop"]) == (
        0,
        0,
        0,
    )
    assert [row["id"] for row in batch["accepted"]] == ["a", "c"]
    assert [row["id"] for row in batch["rejected"]] == ["b"]
    assert batch["rejected"][0]["response"] == ""
    assert batch["submitted_stop"] == 3
    assert batch["manifest"]["completed"] is True
    assert engine_holder["engine"].closed is True


def test_engine_receives_two_specs_runtime_and_absolute_positions(tmp_path: Path) -> None:
    rows = [source_row(str(index)) for index in range(65)]
    first = [(index, "") for index in range(64)]
    second = [(64, "")]

    code, store, _, engine_holder, kwargs, _ = run_with_fakes(
        tmp_path, rows, [first, second], target=66
    )

    assert code == 2
    specs = kwargs["worker_specs"]
    assert [(spec.worker_index, spec.device, Path(spec.cache_root).name) for spec in specs] == [
        (0, 0, "gpu0"),
        (1, 1, "gpu1"),
    ]
    assert kwargs["gpu_memory_utilization"] == 0.8
    assert kwargs["max_model_len"] == 512
    assert kwargs["max_new_tokens"] == 256
    assert kwargs["timeout_seconds"] == 600.0
    calls = engine_holder["engine"].calls
    assert [len(items) for _, items in calls] == [64, 1]
    assert [item.position for item in calls[0][1]] == list(range(64))
    assert [item.seed for item in calls[0][1]] == [
        derive_request_seed(0, "teacher", str(index), 0)
        for index in range(64)
    ]
    assert [batch_id for batch_id, _ in calls] == [1, 2]
    assert store.commits[-1]["manifest"]["completed"] is False


@pytest.mark.parametrize(
    "runtime_info",
    (
        None,
        (WorkerReady(0, 1000, "0", "/cache/gpu0"),),
        (
            WorkerReady(1, 1001, "1", "/cache/gpu1"),
            WorkerReady(0, 1000, "0", "/cache/gpu0"),
        ),
    ),
)
def test_invalid_worker_runtime_info_fails_before_generate_or_commit(
    tmp_path: Path,
    runtime_info,
) -> None:
    config = make_config(tmp_path, stop_after_accepted=1)
    write_jsonl(config.input_path, [source_row()])
    config_path = tmp_path / "config.yaml"
    write_config(config_path, config)
    events: list = []
    store = FakeStore(
        events,
        ResumeState((), (), 0, None, "2026-06-14T00:00:00+00:00"),
        has_v2_manifest=True,
    )
    engine = FakeEngine(
        events,
        [[(0, "<answer>1+2</answer>")]],
        worker_runtime_info=runtime_info,
    )

    with pytest.raises(ValueError, match="worker runtime info"):
        run(
            config_path,
            engine_factory=lambda **kwargs: engine,
            state_store_factory=lambda output_dir: store,
            lock_factory=lambda **kwargs: FakeLock(events, **kwargs),
            cuda_visible_devices=None,
        )

    assert engine.calls == []
    assert store.commits == []
    assert engine.closed is True


def test_resume_batch_id_is_nonzero_and_derived_after_committed_position(
    tmp_path: Path,
) -> None:
    rows = [source_row(str(index)) for index in range(70)]
    accepted = tuple(
        build_teacher_payload(rows[index], "<answer>1+2</answer>")
        for index in range(2)
    )
    rejected = tuple(build_teacher_payload(rows[index], "") for index in range(2, 64))
    state = ResumeState(
        accepted,
        rejected,
        64,
        63,
        "2026-06-14T00:00:00+00:00",
    )
    positioned = [(index, "") for index in range(64, 70)]

    code, store, _, engine_holder, _, _ = run_with_fakes(
        tmp_path,
        rows,
        [positioned],
        target=9,
        state=state,
        has_v2_manifest=True,
    )

    assert code == 2
    assert engine_holder["engine"].calls[0][0] == 2
    assert store.commits[0]["batch_id"] == 2


def test_lock_precedes_state_load_and_flags_are_forwarded(tmp_path: Path) -> None:
    code, _, lock_holder, _, _, events = run_with_fakes(
        tmp_path,
        [source_row()],
        [[(0, "")]],
        target=2,
        recover_stale=True,
        adopt_legacy=True,
    )

    assert code == 2
    assert events.index(("lock.acquire", True)) < events.index(("store.load", True))
    lock = lock_holder["lock"]
    assert lock.kwargs["path"].name == ".teacher_pool.lock"
    assert lock.kwargs["topology"] == "dual_tp1"
    assert lock.released is True


@pytest.mark.parametrize(
    ("recovered_stale", "expected", "unexpected"),
    (
        (False, "output lock acquired normally", "stale output lock recovered"),
        (True, "stale output lock recovered and acquired", "acquired normally"),
    ),
)
def test_lock_logs_actual_recovery_outcome_after_acquire(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    recovered_stale: bool,
    expected: str,
    unexpected: str,
) -> None:
    config = make_config(tmp_path, stop_after_accepted=1)
    config.input_path.write_bytes(b"")
    config_path = tmp_path / "config.yaml"
    write_config(config_path, config)
    events: list = []
    lock = FakeLock(
        events,
        path=config.output_dir / ".teacher_pool.lock",
        recovered_stale=recovered_stale,
    )

    with caplog.at_level(logging.INFO):
        assert run(
            config_path,
            recover_stale_lock=True,
            engine_factory=lambda **kwargs: pytest.fail("engine constructed"),
            lock_factory=lambda **kwargs: lock,
            cuda_visible_devices=None,
        ) == 2

    assert caplog.text.index("stale lock recovery requested") < caplog.text.index(
        expected
    )
    assert unexpected not in caplog.text


def test_already_complete_does_not_construct_engine(tmp_path: Path) -> None:
    row = source_row()
    state = ResumeState(
        (build_teacher_payload(row, "<answer>1+2</answer>"),),
        (),
        1,
        0,
        "2026-06-14T00:00:00+00:00",
    )

    code, store, _, engine_holder, _, _ = run_with_fakes(
        tmp_path, [row], [], target=1, state=state, has_v2_manifest=True
    )

    assert code == 0
    assert engine_holder == {}
    assert store.commits == []


@pytest.mark.parametrize(
    ("engine_error", "commit_error"),
    ((RuntimeError("worker failed"), None), (None, OSError("disk failed"))),
)
def test_failure_does_not_commit_current_batch_and_closes_engine(
    tmp_path: Path,
    engine_error: Exception | None,
    commit_error: Exception | None,
) -> None:
    with pytest.raises(type(engine_error or commit_error)):
        run_with_fakes(
            tmp_path,
            [source_row()],
            [[(0, "")]],
            target=2,
            has_v2_manifest=commit_error is not None,
            engine_error=engine_error,
            commit_error=commit_error,
        )


def test_engine_failure_preserves_commit_count_and_closes_engine(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path, stop_after_accepted=2)
    write_jsonl(config.input_path, [source_row()])
    config_path = tmp_path / "config.yaml"
    write_config(config_path, config)
    events: list = []
    store = FakeStore(
        events,
        ResumeState((), (), 0, None, "2026-06-14T00:00:00+00:00"),
        has_v2_manifest=True,
    )
    engine = FakeEngine(
        events,
        [],
        generate_error=RuntimeError("worker failed"),
    )
    commits_before = len(store.commits)

    with pytest.raises(RuntimeError, match="worker failed"):
        run(
            config_path,
            engine_factory=lambda **kwargs: engine,
            state_store_factory=lambda output_dir: store,
            lock_factory=lambda **kwargs: FakeLock(events, **kwargs),
            cuda_visible_devices=None,
        )

    assert len(store.commits) == commits_before
    assert engine.closed is True


def test_persistence_failure_after_generation_closes_engine(tmp_path: Path) -> None:
    config = make_config(tmp_path, stop_after_accepted=2)
    write_jsonl(config.input_path, [source_row()])
    config_path = tmp_path / "config.yaml"
    write_config(config_path, config)
    events: list = []
    store = FakeStore(
        events,
        ResumeState((), (), 0, None, "2026-06-14T00:00:00+00:00"),
        commit_error=OSError("disk failed"),
        has_v2_manifest=True,
    )
    engine = FakeEngine(events, [[(0, "")]])

    with pytest.raises(OSError, match="disk failed"):
        run(
            config_path,
            engine_factory=lambda **kwargs: engine,
            state_store_factory=lambda output_dir: store,
            lock_factory=lambda **kwargs: FakeLock(events, **kwargs),
            now=lambda: datetime(2026, 6, 14, tzinfo=timezone.utc),
            cuda_visible_devices=None,
        )

    assert engine.closed is True


@pytest.mark.parametrize(
    "latencies",
    (
        None,
        (0.1,),
        (0.1, 0.2, 0.3),
        (-0.1, 0.2),
        (math.nan, 0.2),
        (math.inf, 0.2),
        (True, 0.2),
    ),
)
def test_invalid_worker_latency_metadata_fails_before_batch_commit(
    tmp_path: Path,
    latencies,
) -> None:
    config = make_config(tmp_path, stop_after_accepted=1)
    write_jsonl(config.input_path, [source_row()])
    config_path = tmp_path / "config.yaml"
    write_config(config_path, config)
    events: list = []
    store = FakeStore(
        events,
        ResumeState((), (), 0, None, "2026-06-14T00:00:00+00:00"),
        has_v2_manifest=True,
    )
    engine = FakeEngine(
        events,
        [[(0, "<answer>1+2</answer>")]],
        worker_latencies=latencies,
    )
    commits_before = len(store.commits)

    with pytest.raises(ValueError, match="worker latencies"):
        run(
            config_path,
            engine_factory=lambda **kwargs: engine,
            state_store_factory=lambda output_dir: store,
            lock_factory=lambda **kwargs: FakeLock(events, **kwargs),
            cuda_visible_devices=None,
        )

    assert len(store.commits) == commits_before
    assert engine.closed is True


@pytest.mark.parametrize(
    ("result_counts", "nonempty_counts", "match"),
    (
        (None, (1, 0), "worker result counts"),
        ((1,), (1, 0), "worker result counts"),
        ((0, 1), (0, 1), "expected shard sizes"),
        ((True, 0), (1, 0), "worker result counts"),
        ((1, 0), None, "worker nonempty counts"),
        ((1, 0), (1,), "worker nonempty counts"),
        ((1, 0), (-1, 0), "worker nonempty counts"),
        ((1, 0), (2, 0), "worker nonempty counts"),
    ),
)
def test_invalid_worker_count_metadata_fails_before_batch_commit(
    tmp_path: Path,
    result_counts,
    nonempty_counts,
    match: str,
) -> None:
    config = make_config(tmp_path, stop_after_accepted=1)
    write_jsonl(config.input_path, [source_row()])
    config_path = tmp_path / "config.yaml"
    write_config(config_path, config)
    events: list = []
    store = FakeStore(
        events,
        ResumeState((), (), 0, None, "2026-06-14T00:00:00+00:00"),
        has_v2_manifest=True,
    )
    engine = FakeEngine(
        events,
        [[(0, "<answer>1+2</answer>")]],
        worker_result_counts=result_counts,
        worker_nonempty_counts=nonempty_counts,
    )

    with pytest.raises(ValueError, match=match):
        run(
            config_path,
            engine_factory=lambda **kwargs: engine,
            state_store_factory=lambda output_dir: store,
            lock_factory=lambda **kwargs: FakeLock(events, **kwargs),
            cuda_visible_devices=None,
        )

    assert store.commits == []
    assert engine.closed is True


def test_zero_worker_nonempty_counts_are_valid_rejected_outputs(
    tmp_path: Path,
) -> None:
    code, store, _, engine_holder, _, _ = run_with_fakes(
        tmp_path,
        [source_row("a"), source_row("b")],
        [[(0, ""), (1, "")]],
        target=3,
    )

    assert code == 2
    assert len(store.commits) == 2
    assert engine_holder["engine"].last_worker_nonempty_counts == (0, 0)


def cleanup_case(
    tmp_path: Path,
    *,
    generate_error: BaseException | None = None,
    close_error: BaseException | None = None,
    release_error: BaseException | None = None,
):
    config = make_config(tmp_path, stop_after_accepted=1)
    write_jsonl(config.input_path, [source_row()])
    config_path = tmp_path / "config.yaml"
    write_config(config_path, config)
    events: list = []
    lock = FakeLock(
        events,
        path=Path("unused"),
        release_error=release_error,
    )
    store = FakeStore(
        events,
        ResumeState((), (), 0, None, "2026-06-14T00:00:00+00:00"),
        has_v2_manifest=True,
    )
    engine = FakeEngine(
        events,
        [[(0, "<answer>1+2</answer>")]],
        generate_error=generate_error,
        close_error=close_error,
    )

    def invoke():
        return run(
            config_path,
            engine_factory=lambda **kwargs: engine,
            state_store_factory=lambda output_dir: store,
            lock_factory=lambda **kwargs: lock,
            now=lambda: datetime(2026, 6, 14, tzinfo=timezone.utc),
            cuda_visible_devices=None,
        )
    return invoke, engine, lock, store, events


def test_engine_close_failure_without_primary_propagates_and_releases_lock(
    tmp_path: Path,
) -> None:
    invoke, engine, lock, _, _ = cleanup_case(
        tmp_path,
        close_error=RuntimeError("close failed"),
    )

    with pytest.raises(RuntimeError, match="close failed"):
        invoke()

    assert engine.closed is True
    assert lock.released is True


def test_lock_release_failure_without_primary_propagates_after_engine_close(
    tmp_path: Path,
) -> None:
    invoke, engine, lock, _, _ = cleanup_case(
        tmp_path,
        release_error=RuntimeError("release failed"),
    )

    with pytest.raises(RuntimeError, match="release failed"):
        invoke()

    assert engine.closed is True
    assert lock.released is True


def test_both_cleanup_failures_without_primary_raise_exception_group(
    tmp_path: Path,
) -> None:
    invoke, engine, lock, _, _ = cleanup_case(
        tmp_path,
        close_error=RuntimeError("close failed"),
        release_error=RuntimeError("release failed"),
    )

    with pytest.raises(ExceptionGroup) as exc_info:
        invoke()

    assert {str(error) for error in exc_info.value.exceptions} == {
        "close failed",
        "release failed",
    }
    assert engine.closed is True
    assert lock.released is True


def test_primary_failure_is_preserved_with_both_cleanup_failures(
    tmp_path: Path,
) -> None:
    invoke, engine, lock, store, _ = cleanup_case(
        tmp_path,
        generate_error=RuntimeError("worker failed"),
        close_error=RuntimeError("close failed"),
        release_error=RuntimeError("release failed"),
    )
    commits_before = len(store.commits)

    with pytest.raises(RuntimeError, match="worker failed") as exc_info:
        invoke()

    assert isinstance(exc_info.value.__cause__, ExceptionGroup)
    assert len(store.commits) == commits_before
    assert engine.closed is True
    assert lock.released is True


@pytest.mark.parametrize(
    ("close_error", "release_error", "cleanup_message"),
    (
        (RuntimeError("close failed"), None, "close failed"),
        (None, RuntimeError("release failed"), "release failed"),
    ),
)
def test_primary_failure_is_preserved_with_each_cleanup_failure(
    tmp_path: Path,
    close_error: BaseException | None,
    release_error: BaseException | None,
    cleanup_message: str,
) -> None:
    invoke, engine, lock, _, _ = cleanup_case(
        tmp_path,
        generate_error=RuntimeError("worker failed"),
        close_error=close_error,
        release_error=release_error,
    )

    with pytest.raises(RuntimeError, match="worker failed") as exc_info:
        invoke()

    assert str(exc_info.value.__cause__) == cleanup_message
    assert engine.closed is True
    assert lock.released is True


def test_keyboard_interrupt_returns_130_closes_engine_and_releases_lock(
    tmp_path: Path,
) -> None:
    code, store, lock_holder, engine_holder, _, _ = run_with_fakes(
        tmp_path,
        [source_row()],
        [],
        target=2,
        engine_error=KeyboardInterrupt(),
    )

    assert code == 130
    assert len(store.commits) == 1
    assert engine_holder["engine"].closed is True
    assert lock_holder["lock"].released is True


def test_keyboard_interrupt_keeps_130_when_both_cleanup_steps_fail(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    invoke, engine, lock, _, _ = cleanup_case(
        tmp_path,
        generate_error=KeyboardInterrupt(),
        close_error=RuntimeError("close failed"),
        release_error=RuntimeError("release failed"),
    )

    with caplog.at_level(logging.ERROR):
        assert invoke() == 130

    assert engine.closed is True
    assert lock.released is True
    assert "close failed" in caplog.text
    assert "release failed" in caplog.text
    assert "workers closed" not in caplog.text


def test_logs_lifecycle_shards_latency_counts_and_exitcodes(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    rows = [source_row("a"), source_row("b")]
    config = make_config(tmp_path, stop_after_accepted=2)
    write_jsonl(config.input_path, rows)
    config_path = tmp_path / "config.yaml"
    write_config(config_path, config)
    events: list = []
    store = FakeStore(
        events,
        ResumeState((), (), 0, None, "2026-06-14T00:00:00+00:00"),
    )
    engine = FakeEngine(
        events,
        [[(0, "<answer>1+2</answer>"), (1, "<answer>1+2</answer>")]],
        worker_latencies=(0.125, 0.25),
    )

    with caplog.at_level(logging.INFO):
        code = run(
            config_path,
            recover_stale_lock=True,
            engine_factory=lambda **kwargs: engine,
            state_store_factory=lambda output_dir: store,
            lock_factory=lambda **kwargs: FakeLock(events, **kwargs),
            now=lambda: datetime(2026, 6, 14, tzinfo=timezone.utc),
            cuda_visible_devices=None,
        )

    assert code == 0
    messages = caplog.text
    assert "stale lock recovery requested" in messages
    assert "acquiring output lock" in messages
    assert "output lock acquired normally" in messages
    assert "source rows=2" in messages
    assert "resume processed=0 accepted=0 rejected=0" in messages
    assert "engine starting" in messages
    assert "engine ready" in messages
    assert (
        "worker0 runtime pid=1000 visible_device=0 cache_root=/cache/gpu0"
        in messages
    )
    assert (
        "worker1 runtime pid=1001 visible_device=1 cache_root=/cache/gpu1"
        in messages
    )
    assert "batch=1 global_range=[0,2)" in messages
    assert (
        "worker0_shard=1 worker0_results=1 worker0_nonempty=1 "
        "worker0_latency_seconds=0.125 worker1_shard=1 worker1_results=1 "
        "worker1_nonempty=1 worker1_latency_seconds=0.250"
    ) in messages
    assert "accepted=2 rejected=0" in messages
    assert "workers closed" in messages
    assert "worker shutdown exitcodes=(0, 0) runtime_pids=(1000, 1001)" in messages
    assert "workers: gpu0 device=" not in messages
    assert "stale output lock recovered" not in messages
    assert "<answer>" not in messages


def test_source_schema_failure_under_lock_happens_before_engine(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    write_jsonl(config.input_path, [source_row(prompt="")])
    config_path = tmp_path / "config.yaml"
    write_config(config_path, config)
    events: list = []
    engine_constructed = []
    lock = FakeLock(events, path=config.output_dir / ".teacher_pool.lock")

    with pytest.raises(ValueError, match="prompt"):
        run(
            config_path,
            engine_factory=lambda **kwargs: engine_constructed.append(kwargs),
            lock_factory=lambda **kwargs: lock,
            cuda_visible_devices=None,
        )
    assert engine_constructed == []
    assert events == [("lock.acquire", False), ("lock.release",)]


def test_initial_real_store_commit_materializes_empty_contract(tmp_path: Path) -> None:
    row = source_row()
    config = make_config(tmp_path, stop_after_accepted=1)
    write_jsonl(config.input_path, [row])
    config_path = tmp_path / "config.yaml"
    write_config(config_path, config)
    engine = FakeEngine([], [[(0, "<answer>1+2</answer>")]])

    assert run(
        config_path,
        engine_factory=lambda **kwargs: engine,
        now=lambda: datetime(2026, 6, 14, tzinfo=timezone.utc),
        cuda_visible_devices=None,
    ) == 0

    accepted = (config.output_dir / "teacher_accepted_20k.jsonl").read_text(
        encoding="utf-8"
    )
    rejected = (config.output_dir / "teacher_rejected.jsonl").read_text(
        encoding="utf-8"
    )
    manifest_text = (config.output_dir / "manifest.json").read_text(encoding="utf-8")
    assert accepted
    assert rejected == ""
    assert hashlib.sha256(b"").hexdigest() in manifest_text


def test_exhausted_real_store_publishes_partial_manifest_v2(
    tmp_path: Path,
) -> None:
    row = source_row()
    config = make_config(tmp_path, stop_after_accepted=2)
    write_jsonl(config.input_path, [row])
    config_path = tmp_path / "config.yaml"
    write_config(config_path, config)
    engine = FakeEngine([], [[(0, "")]])

    assert run(
        config_path,
        engine_factory=lambda **kwargs: engine,
        now=lambda: datetime(2026, 6, 14, tzinfo=timezone.utc),
        cuda_visible_devices=None,
    ) == 2

    manifest = ManifestV2.parse(
        json.loads(
            (config.output_dir / "manifest.json").read_text(encoding="utf-8")
        )
    )
    assert manifest.artifact_type == "partial_teacher_pool"
    assert manifest.stage == "teacher_accepted_pool"
    assert manifest.stage_metadata["completed"] is False
    assert manifest.stage_metadata["accepted_count"] == 0
    assert manifest.stage_metadata["target_accepted_count"] == 2
    assert manifest.stage_metadata["teacher_state"]["completed"] is False
    assert manifest.stage_metadata["teacher_state"]["accepted_count"] == 0
    assert "accepted_pool_artifact_id" not in manifest.stage_metadata


def test_complete_teacher_manifest_links_validation_parent(
    tmp_path: Path,
) -> None:
    row = source_row()
    config = make_config(tmp_path, stop_after_accepted=1)
    write_jsonl(config.input_path, [row])
    config_path = tmp_path / "config.yaml"
    write_config(config_path, config)
    validation_manifest_path = config.input_path.parent / "validation_manifest.json"
    source_bytes = config.input_path.read_bytes()
    validation_manifest = ManifestV2.build(
        artifact_type="dataset_split",
        stage="build_validation_splits",
        files=(
            ArtifactFile(
                relative_path=config.input_path.name,
                sha256=hashlib.sha256(source_bytes).hexdigest(),
                byte_size=len(source_bytes),
                row_count=1,
                field_schema={},
            ),
        ),
        parents=(),
        config={"seed": 0},
        stage_metadata={"completed": True},
    )
    publish_manifest(validation_manifest_path, validation_manifest)
    engine = FakeEngine([], [[(0, "<answer>1+2</answer>")]])

    assert run(
        config_path,
        engine_factory=lambda **kwargs: engine,
        now=lambda: datetime(2026, 6, 14, tzinfo=timezone.utc),
        cuda_visible_devices=None,
    ) == 0

    teacher_manifest = ManifestV2.parse(
        json.loads(
            (config.output_dir / "manifest.json").read_text(encoding="utf-8")
        )
    )
    assert teacher_manifest.artifact_type == "teacher_accepted_pool"
    assert teacher_manifest.stage_metadata["completed"] is True
    assert teacher_manifest.parents[0].artifact_id == validation_manifest.artifact_id
    assert teacher_manifest.parents[0].sha256 == sha256_file(
        validation_manifest_path
    )


def test_script_help_imports_without_vllm() -> None:
    script = REPO_ROOT / "post_train_v2/scripts/generation/build_teacher_pool.py"
    env = dict(os.environ)
    env["PYTHONPATH"] = ""
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=Path("/tmp") if Path("/tmp").exists() else REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert "--recover-stale-lock" in result.stdout
    assert "--adopt-legacy-state" in result.stdout
    assert importlib.util.find_spec("vllm") is None or "vllm" not in sys.modules
