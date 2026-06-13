from __future__ import annotations

import hashlib
import importlib.util
import os
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from post_train.src.countdown.config import load_yaml_config
from post_train.src.countdown.io import write_jsonl
from post_train_v2.src.generation.teacher_state import (
    ResumeState,
    TeacherGenerationConfig,
)
from post_train_v2.scripts.generation.build_teacher_pool import (
    PRODUCTION_CONFIG,
    SMOKE_CONFIG,
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
        "input_path": "post_train/data/processed/train_pool.jsonl",
        "output_dir": "post_train/data/teacher_rollouts",
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
        "schema_version": 1,
    }

    assert production == expected
    assert smoke == {
        **expected,
        "output_dir": "/tmp/post_train_v2_teacher_smoke",
        "stop_after_accepted": 8,
        "cache_root": "/tmp/countdown_teacher_vllm_smoke",
    }


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


def test_payload_retains_full_stripped_response_and_validation() -> None:
    payload = build_teacher_payload(
        source_row(numbers=[7, 3, 8, 2], target=24),
        " reasoning\n<answer> (7-3)*(8-2) </answer>\n ",
    )

    assert payload["response"] == "reasoning\n<answer> (7-3)*(8-2) </answer>"
    assert payload["teacher_expr"] == "(7-3)*(8-2)"
    assert payload["validation"] == {"ok": True, "error": None, "value": 24}
    empty = build_teacher_payload(source_row(), "")
    assert empty["validation"] == {
        "ok": False,
        "error": "missing_answer_tag",
        "value": None,
    }


class FakeLock:
    def __init__(self, events: list, **kwargs) -> None:
        self.events = events
        self.kwargs = kwargs
        self.released = False

    def acquire(self, recover_stale: bool = False) -> None:
        self.events.append(("lock.acquire", recover_stale))

    def release(self) -> None:
        self.released = True
        self.events.append(("lock.release",))


class FakeStore:
    def __init__(
        self,
        events: list,
        state: ResumeState,
        *,
        commit_error: Exception | None = None,
        state_exists: bool = False,
    ) -> None:
        self.events = events
        self.state = state
        self.commit_error = commit_error
        self.state_exists = state_exists
        self.commits: list[dict] = []

    def load_resume_state(self, rows, config, adopt_legacy_state=False):
        self.events.append(("store.load", adopt_legacy_state))
        return self.state

    def commit(self, **kwargs) -> None:
        self.events.append(("store.commit", kwargs["batch_id"]))
        if self.commit_error is not None:
            raise self.commit_error
        self.commits.append(kwargs)
        self.state_exists = True


class FakeEngine:
    def __init__(self, events: list, responses, *, generate_error=None) -> None:
        self.events = events
        self.responses = list(responses)
        self.generate_error = generate_error
        self.calls: list[tuple[int, tuple]] = []
        self.closed = False

    def start(self):
        self.events.append(("engine.start",))
        return self

    def generate(self, batch_id, items):
        self.calls.append((batch_id, tuple(items)))
        self.events.append(("engine.generate", batch_id))
        if self.generate_error is not None:
            raise self.generate_error
        response = self.responses.pop(0)
        return response

    def close(self) -> None:
        self.closed = True
        self.events.append(("engine.close",))


def run_with_fakes(
    tmp_path: Path,
    rows: list[dict],
    responses,
    *,
    target: int = 2,
    state: ResumeState | None = None,
    state_exists: bool = False,
    engine_error: BaseException | None = None,
    commit_error: Exception | None = None,
    recover_stale: bool = False,
    adopt_legacy: bool = False,
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
        state_exists=state_exists,
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
        engine = FakeEngine(events, responses, generate_error=engine_error)
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
    assert [batch_id for batch_id, _ in calls] == [1, 2]
    assert store.commits[-1]["manifest"]["completed"] is False


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
        state_exists=True,
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
        tmp_path, [row], [], target=1, state=state, state_exists=False
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
            state_exists=commit_error is not None,
            engine_error=engine_error,
            commit_error=commit_error,
        )


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
        state_exists=True,
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


def test_engine_close_failure_still_releases_owned_lock(tmp_path: Path) -> None:
    config = make_config(tmp_path, stop_after_accepted=1)
    write_jsonl(config.input_path, [source_row()])
    config_path = tmp_path / "config.yaml"
    write_config(config_path, config)
    events: list = []
    lock = FakeLock(events, path=Path("unused"))
    store = FakeStore(
        events,
        ResumeState((), (), 0, None, "2026-06-14T00:00:00+00:00"),
        state_exists=True,
    )

    class CloseFailureEngine(FakeEngine):
        def close(self) -> None:
            super().close()
            raise RuntimeError("close failed")

    engine = CloseFailureEngine(events, [[(0, "<answer>1+2</answer>")]])

    with pytest.raises(RuntimeError, match="close failed"):
        run(
            config_path,
            engine_factory=lambda **kwargs: engine,
            state_store_factory=lambda output_dir: store,
            lock_factory=lambda **kwargs: lock,
            now=lambda: datetime(2026, 6, 14, tzinfo=timezone.utc),
            cuda_visible_devices=None,
        )

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


def test_source_schema_failure_happens_before_engine_or_output_lock(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    write_jsonl(config.input_path, [source_row(prompt="")])
    config_path = tmp_path / "config.yaml"
    write_config(config_path, config)
    constructed = []

    with pytest.raises(ValueError, match="prompt"):
        run(
            config_path,
            engine_factory=lambda **kwargs: constructed.append(kwargs),
            lock_factory=lambda **kwargs: constructed.append(kwargs),
            cuda_visible_devices=None,
        )
    assert constructed == []


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
