import json
import sys
from pathlib import Path

import pytest

from post_train.scripts.data import build_teacher_pool
from post_train.scripts.data.build_teacher_pool import (
    atomic_write_jsonl,
    build_teacher_payload,
    collect_processed_ids,
    process_teacher_responses,
    validate_resume_state,
    validate_source_rows,
)
from post_train.src.countdown.io import read_jsonl


def teacher_config(*, target: int = 0) -> dict:
    return {
        "model_path": "model",
        "stop_after_accepted": target,
        "max_new_tokens": 128,
        "temperature": 0.2,
        "top_p": 0.95,
        "enable_thinking": False,
        "batch_size": 4,
    }


def patch_run_paths(
    monkeypatch,
    tmp_path: Path,
    *,
    target: int = 0,
) -> tuple[Path, Path, Path, Path]:
    config_path = tmp_path / "teacher.yaml"
    input_path = tmp_path / "train.jsonl"
    model_path = tmp_path / "model"
    output_dir = tmp_path / "output"
    resolved = {
        "config": config_path,
        "input": input_path,
        "model": model_path,
        build_teacher_pool.OUTPUT_DIR: output_dir,
    }
    monkeypatch.setattr(
        build_teacher_pool,
        "resolve_path",
        lambda value, root: resolved[str(value)],
    )
    monkeypatch.setattr(
        build_teacher_pool,
        "load_yaml_config",
        lambda path: teacher_config(target=target),
    )
    return config_path, input_path, model_path, output_dir


class FakeLock:
    def __init__(
        self,
        *,
        events: list,
        recovered_stale: bool = False,
        acquire_error: BaseException | None = None,
        release_error: BaseException | None = None,
        **kwargs,
    ):
        self.events = events
        self.recovered_stale = recovered_stale
        self.acquire_error = acquire_error
        self.release_error = release_error
        self.kwargs = kwargs

    def acquire(self, recover_stale: bool = False) -> None:
        self.events.append(("acquire", recover_stale))
        if self.acquire_error is not None:
            raise self.acquire_error

    def release(self) -> None:
        self.events.append("release")
        if self.release_error is not None:
            raise self.release_error


def test_collect_processed_ids_includes_accepted_and_rejected_rows():
    accepted = [{"id": "train-000001"}, {"id": "train-000002"}]
    rejected = [{"id": "train-000003"}]

    assert collect_processed_ids(accepted, rejected) == {
        "train-000001",
        "train-000002",
        "train-000003",
    }


def test_validate_resume_state_rejects_duplicate_ids_across_outputs():
    accepted = [{"id": "train-000001"}]
    rejected = [{"id": "train-000001"}]

    with pytest.raises(ValueError, match="duplicate id.*train-000001"):
        validate_resume_state(accepted, rejected, target=10)


def test_validate_resume_state_rejects_accepted_count_over_target():
    accepted = [{"id": "train-000001"}, {"id": "train-000002"}]

    with pytest.raises(ValueError, match="accepted rows.*exceeds target"):
        validate_resume_state(accepted, [], target=1)


def test_validate_source_rows_rejects_duplicate_ids():
    rows = [{"id": "train-000001"}, {"id": "train-000001"}]

    with pytest.raises(ValueError, match="duplicate id.*train-000001"):
        validate_source_rows(rows)


def test_atomic_write_jsonl_replaces_existing_file(tmp_path):
    path = tmp_path / "teacher_accepted_20k.jsonl"
    temp_path = path.with_name(f"{path.name}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        path.write_text('{"id": "old"}\n', encoding="utf-8")

        atomic_write_jsonl(path, [{"id": "new"}])

        assert read_jsonl(path) == [{"id": "new"}]
        assert not temp_path.exists()
    finally:
        if path.exists():
            path.unlink()
        if temp_path.exists():
            temp_path.unlink()


def test_build_teacher_payload_preserves_row_and_records_validation():
    row = {
        "id": "train-000001",
        "numbers": [7, 3, 8, 2],
        "target": 24,
        "prompt": "solve",
    }

    payload = build_teacher_payload(
        row,
        "reasoning\n<answer> (7-3)*(8-2) </answer>\n",
    )

    assert payload["id"] == row["id"]
    assert payload["prompt"] == row["prompt"]
    assert payload["response"] == "reasoning\n<answer> (7-3)*(8-2) </answer>"
    assert payload["teacher_expr"] == "(7-3)*(8-2)"
    assert payload["validation"] == {
        "ok": True,
        "error": None,
        "value": 24,
    }


def test_process_teacher_responses_stops_before_validating_after_target():
    rows = [
        {
            "id": "train-000001",
            "numbers": [7, 3, 8, 2],
            "target": 24,
            "prompt": "solve",
        },
        {
            "id": "train-000002",
            "numbers": [1, 2, 3, 4],
            "target": 99,
            "prompt": "solve",
        },
    ]
    responses = [
        "<answer> (7-3)*(8-2) </answer>",
        "<answer> 1+2+3+4 </answer>",
    ]
    accepted = []
    rejected = []
    processed_ids = set()

    process_teacher_responses(
        rows=rows,
        responses=responses,
        accepted=accepted,
        rejected=rejected,
        processed_ids=processed_ids,
        target=1,
    )

    assert [row["id"] for row in accepted] == ["train-000001"]
    assert rejected == []
    assert processed_ids == {"train-000001"}


def test_cli_help_includes_recover_stale_lock(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["build_teacher_pool.py", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        build_teacher_pool.parse_args()

    assert exc_info.value.code == 0
    assert "--recover-stale-lock" in capsys.readouterr().out


def test_run_acquires_lock_before_manifest_outputs_and_generator(
    tmp_path: Path,
    monkeypatch,
):
    config_path, input_path, model_path, output_dir = patch_run_paths(
        monkeypatch,
        tmp_path,
        target=1,
    )
    output_dir.mkdir()
    (output_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
    accepted_path = output_dir / build_teacher_pool.ACCEPTED_FILENAME
    rejected_path = output_dir / build_teacher_pool.REJECTED_FILENAME
    accepted_path.touch()
    rejected_path.touch()
    events = []
    source_rows = [
        {
            "id": "train-000001",
            "numbers": [7, 3, 8, 2],
            "target": 24,
            "prompt": "solve",
        }
    ]

    def fake_read_manifest(path):
        events.append("manifest")
        return {}

    def fake_read_jsonl(path):
        path = Path(path)
        if path == input_path:
            events.append("input")
            return source_rows
        if path == accepted_path:
            events.append("accepted")
            return []
        if path == rejected_path:
            events.append("rejected")
            return []
        raise AssertionError(path)

    class FakeGenerator:
        def __init__(self, path):
            events.append(("generator", path))

        def generate(self, prompts, generation_config):
            return ["<answer> (7-3)*(8-2) </answer>"]

    locks = []

    def lock_factory(**kwargs):
        lock = FakeLock(events=events, **kwargs)
        locks.append(lock)
        return lock

    monkeypatch.setattr(build_teacher_pool, "read_manifest", fake_read_manifest, raising=False)
    monkeypatch.setattr(build_teacher_pool, "read_jsonl", fake_read_jsonl)
    monkeypatch.setattr(build_teacher_pool, "atomic_write_jsonl", lambda path, rows: None)
    monkeypatch.setattr(build_teacher_pool, "atomic_write_manifest", lambda path, payload: None)

    build_teacher_pool.run(
        "config",
        "input",
        lock_factory=lock_factory,
        generator_factory=FakeGenerator,
    )

    assert events[:6] == [
        ("acquire", False),
        "manifest",
        "input",
        "accepted",
        "rejected",
        ("generator", str(model_path)),
    ]
    assert events[-1] == "release"
    assert locks[0].kwargs == {
        "path": output_dir / ".teacher_pool.lock",
        "config_path": config_path,
        "output_dir": output_dir,
        "topology": "legacy_single_tp1",
    }


@pytest.mark.parametrize("recover_stale", [False, True])
def test_run_delegates_lock_recovery_flag(
    tmp_path: Path,
    monkeypatch,
    recover_stale: bool,
):
    patch_run_paths(monkeypatch, tmp_path)
    events = []
    monkeypatch.setattr(build_teacher_pool, "read_jsonl", lambda path: [])
    monkeypatch.setattr(build_teacher_pool, "atomic_write_jsonl", lambda path, rows: None)
    monkeypatch.setattr(build_teacher_pool, "atomic_write_manifest", lambda path, payload: None)

    build_teacher_pool.run(
        "config",
        "input",
        recover_stale_lock=recover_stale,
        lock_factory=lambda **kwargs: FakeLock(events=events, **kwargs),
        generator_factory=lambda path: pytest.fail("generator must not be constructed"),
    )

    assert events == [("acquire", recover_stale), "release"]


def test_run_propagates_lock_acquisition_failure_without_release(
    tmp_path: Path,
    monkeypatch,
):
    patch_run_paths(monkeypatch, tmp_path)
    events = []
    active_lock = RuntimeError("active output lock")

    with pytest.raises(RuntimeError, match="active output lock") as exc_info:
        build_teacher_pool.run(
            "config",
            "input",
            lock_factory=lambda **kwargs: FakeLock(
                events=events,
                acquire_error=active_lock,
                **kwargs,
            ),
        )

    assert exc_info.value is active_lock
    assert events == [("acquire", False)]


@pytest.mark.parametrize(
    "manifest",
    [
        {"generation_contract_fingerprint": "abc123"},
        {"generation_contract": {"topology": "dual_gpu_teacher"}},
        {"stage": "teacher_accepted_pool", "schema_version": 1},
    ],
)
def test_run_rejects_v2_manifest_before_generator(
    tmp_path: Path,
    monkeypatch,
    manifest: dict,
):
    _, _, _, output_dir = patch_run_paths(monkeypatch, tmp_path, target=1)
    output_dir.mkdir()
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest) + "\n",
        encoding="utf-8",
    )
    events = []
    monkeypatch.setattr(
        build_teacher_pool,
        "read_jsonl",
        lambda path: pytest.fail("JSONL reads must not occur for V2 state"),
    )

    with pytest.raises(RuntimeError, match="archive or remove.*V2"):
        build_teacher_pool.run(
            "config",
            "input",
            lock_factory=lambda **kwargs: FakeLock(events=events, **kwargs),
            generator_factory=lambda path: pytest.fail(
                "generator must not be constructed for V2 state"
            ),
        )

    assert events == [("acquire", False), "release"]


@pytest.mark.parametrize(
    "fingerprint",
    ["", None, False, 0],
)
def test_run_rejects_v2_manifest_with_falsey_fingerprint(
    tmp_path: Path,
    monkeypatch,
    fingerprint,
):
    _, _, _, output_dir = patch_run_paths(monkeypatch, tmp_path)
    output_dir.mkdir()
    (output_dir / "manifest.json").write_text(
        json.dumps({"generation_contract_fingerprint": fingerprint}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="archive or remove.*V2"):
        build_teacher_pool.run("config", "input")


@pytest.mark.parametrize(
    "contract",
    [{}, None, False, 0, ""],
)
def test_run_rejects_v2_manifest_with_falsey_contract(
    tmp_path: Path,
    monkeypatch,
    contract,
):
    _, _, _, output_dir = patch_run_paths(monkeypatch, tmp_path)
    output_dir.mkdir()
    (output_dir / "manifest.json").write_text(
        json.dumps({"generation_contract": contract}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="archive or remove.*V2"):
        build_teacher_pool.run("config", "input")


@pytest.mark.parametrize("schema_version", [None, "", False, 0, 2])
def test_run_rejects_v2_stage_with_schema_version_key_present(
    tmp_path: Path,
    monkeypatch,
    schema_version,
):
    _, _, _, output_dir = patch_run_paths(monkeypatch, tmp_path)
    output_dir.mkdir()
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "stage": "teacher_accepted_pool",
                "schema_version": schema_version,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="archive or remove.*V2"):
        build_teacher_pool.run("config", "input")


def test_run_accepts_legacy_envelope_manifest(
    tmp_path: Path,
    monkeypatch,
):
    _, input_path, _, output_dir = patch_run_paths(monkeypatch, tmp_path, target=1)
    output_dir.mkdir()
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "manifest_version": 1,
                "schema": "countdown.post_train.manifest.v1",
                "name": "teacher_accepted_pool",
                "stage": "teacher_accepted_pool",
                "created_at": "2026-06-14T00:00:00+00:00",
                "model": str(tmp_path / "model"),
                "num_accepted": 0,
                "num_rejected": 0,
                "max_new_tokens": 128,
                "enable_thinking": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    source_rows = [
        {
            "id": "train-000001",
            "numbers": [7, 3, 8, 2],
            "target": 24,
            "prompt": "solve",
        }
    ]
    generator_constructed = []

    def fake_read_jsonl(path):
        return source_rows if Path(path) == input_path else []

    class FakeGenerator:
        def __init__(self, path):
            generator_constructed.append(path)

        def generate(self, prompts, generation_config):
            return ["<answer> (7-3)*(8-2) </answer>"]

    monkeypatch.setattr(build_teacher_pool, "read_jsonl", fake_read_jsonl)
    monkeypatch.setattr(build_teacher_pool, "atomic_write_jsonl", lambda path, rows: None)
    monkeypatch.setattr(build_teacher_pool, "atomic_write_manifest", lambda path, payload: None)

    build_teacher_pool.run(
        "config",
        "input",
        lock_factory=lambda **kwargs: FakeLock(events=[], **kwargs),
        generator_factory=FakeGenerator,
    )

    assert len(generator_constructed) == 1


def test_run_holds_real_output_lock_during_output_access(
    tmp_path: Path,
    monkeypatch,
):
    _, input_path, _, output_dir = patch_run_paths(monkeypatch, tmp_path)
    lock_path = output_dir / ".teacher_pool.lock"
    access_events = []

    def fake_read_jsonl(path):
        assert Path(path) == input_path
        assert lock_path.exists()
        access_events.append("read")
        return []

    def fake_atomic_write_jsonl(path, rows):
        assert lock_path.exists()
        access_events.append(("write_jsonl", Path(path).name))

    def fake_atomic_write_manifest(path, payload):
        assert lock_path.exists()
        access_events.append(("write_manifest", Path(path).name))

    monkeypatch.setattr(build_teacher_pool, "read_jsonl", fake_read_jsonl)
    monkeypatch.setattr(
        build_teacher_pool,
        "atomic_write_jsonl",
        fake_atomic_write_jsonl,
    )
    monkeypatch.setattr(
        build_teacher_pool,
        "atomic_write_manifest",
        fake_atomic_write_manifest,
    )

    build_teacher_pool.run(
        "config",
        "input",
        generator_factory=lambda path: pytest.fail(
            "generator must not be constructed"
        ),
    )

    assert access_events == [
        "read",
        ("write_jsonl", build_teacher_pool.ACCEPTED_FILENAME),
        ("write_jsonl", build_teacher_pool.REJECTED_FILENAME),
        ("write_manifest", build_teacher_pool.MANIFEST_FILENAME),
    ]
    assert not lock_path.exists()


def test_run_rejects_v2_transaction_journal_before_generator(
    tmp_path: Path,
    monkeypatch,
):
    _, _, _, output_dir = patch_run_paths(monkeypatch, tmp_path, target=1)
    output_dir.mkdir()
    (output_dir / ".teacher_pool.transaction.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    events = []
    monkeypatch.setattr(
        build_teacher_pool,
        "read_jsonl",
        lambda path: pytest.fail("JSONL reads must not occur with a transaction"),
    )

    with pytest.raises(RuntimeError, match="transaction.*archive or remove"):
        build_teacher_pool.run(
            "config",
            "input",
            lock_factory=lambda **kwargs: FakeLock(events=events, **kwargs),
            generator_factory=lambda path: pytest.fail(
                "generator must not be constructed with a transaction"
            ),
        )

    assert events == [("acquire", False), "release"]


def test_run_propagates_release_failure_without_primary_error(
    tmp_path: Path,
    monkeypatch,
):
    patch_run_paths(monkeypatch, tmp_path)
    release_error = RuntimeError("release failed")
    monkeypatch.setattr(build_teacher_pool, "read_jsonl", lambda path: [])
    monkeypatch.setattr(build_teacher_pool, "atomic_write_jsonl", lambda path, rows: None)
    monkeypatch.setattr(build_teacher_pool, "atomic_write_manifest", lambda path, payload: None)

    with pytest.raises(RuntimeError, match="release failed") as exc_info:
        build_teacher_pool.run(
            "config",
            "input",
            lock_factory=lambda **kwargs: FakeLock(
                events=[],
                release_error=release_error,
                **kwargs,
            ),
        )

    assert exc_info.value is release_error


def test_run_preserves_primary_error_when_release_also_fails(
    tmp_path: Path,
    monkeypatch,
):
    patch_run_paths(monkeypatch, tmp_path)
    primary_error = ValueError("generation failed")
    release_error = RuntimeError("release failed")

    def fail_read(path):
        raise primary_error

    monkeypatch.setattr(build_teacher_pool, "read_jsonl", fail_read)

    with pytest.raises(ValueError, match="generation failed") as exc_info:
        build_teacher_pool.run(
            "config",
            "input",
            lock_factory=lambda **kwargs: FakeLock(
                events=[],
                release_error=release_error,
                **kwargs,
            ),
        )

    assert exc_info.value is primary_error
    assert exc_info.value.__cause__ is release_error
    assert any("release failed" in note for note in exc_info.value.__notes__)
