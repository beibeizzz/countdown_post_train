from __future__ import annotations

import errno
import hashlib
import json
import os
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import post_train_v2.src.generation.teacher_state as teacher_state
from post_train.src.countdown.io import read_jsonl, write_jsonl
from post_train_v2.src.generation.teacher_state import (
    TeacherGenerationConfig,
    TeacherStateStore,
    build_generation_contract,
    build_manifest,
    derive_resume_state,
    fingerprint_contract,
    sha256_file,
)


def make_config(tmp_path: Path, **overrides) -> TeacherGenerationConfig:
    model_path = tmp_path / "model"
    model_path.mkdir(exist_ok=True)
    input_path = tmp_path / "source.jsonl"
    if not input_path.exists():
        write_jsonl(input_path, source_rows())
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
        "stop_after_accepted": 20_000,
        "cache_root": tmp_path / "cache",
        "schema_version": 1,
    }
    values.update(overrides)
    return TeacherGenerationConfig(**values)


def source_rows() -> list[dict]:
    return [
        {"id": "a", "numbers": [1, 2], "target": 3},
        {"id": "b", "numbers": [2, 2], "target": 4},
        {"id": "c", "numbers": [3, 2], "target": 5},
        {"id": "d", "numbers": [4, 2], "target": 6},
    ]


def accepted_row(row_id: str, numbers: list[int], target: int) -> dict:
    return {
        "id": row_id,
        "numbers": numbers,
        "target": target,
        "response": f"<answer>{numbers[0]}+{numbers[1]}</answer>",
    }


def rejected_row(row_id: str, numbers: list[int], target: int) -> dict:
    return {
        "id": row_id,
        "numbers": numbers,
        "target": target,
        "response": f"<answer>{numbers[0]}-{numbers[1]}</answer>",
    }


def test_config_is_frozen_validates_and_resolves_without_mutation(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    original_model = config.model_path

    assert config.validate() is None
    assert config.resolved().model_path == config.model_path.resolve()
    assert config.resolved().input_path == config.input_path.resolve()
    assert config.resolved().output_dir == config.output_dir.resolve()
    assert config.model_path == original_model
    with pytest.raises(FrozenInstanceError):
        config.seed = 3  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("devices", (0,), "exactly two"),
        ("devices", (0, 0), "distinct"),
        ("devices", (-1, 1), "nonnegative"),
        ("devices", (False, 1), "exact integers"),
        ("devices", (0.0, 1), "exact integers"),
        ("topology", "tp2", "dual_tp1"),
        ("batch_size", 0, "batch_size"),
        ("batch_size", True, "batch_size"),
        ("worker_timeout_seconds", 0, "worker_timeout_seconds"),
        ("worker_timeout_seconds", float("nan"), "worker_timeout_seconds"),
        ("worker_timeout_seconds", float("inf"), "worker_timeout_seconds"),
        ("worker_timeout_seconds", float("-inf"), "worker_timeout_seconds"),
        ("gpu_memory_utilization", 0, "gpu_memory_utilization"),
        ("gpu_memory_utilization", 1.01, "gpu_memory_utilization"),
        ("gpu_memory_utilization", float("nan"), "gpu_memory_utilization"),
        ("gpu_memory_utilization", float("inf"), "gpu_memory_utilization"),
        ("gpu_memory_utilization", float("-inf"), "gpu_memory_utilization"),
        ("max_model_len", 0, "max_model_len"),
        ("max_new_tokens", 0, "max_new_tokens"),
        ("temperature", -0.01, "temperature"),
        ("temperature", float("nan"), "temperature"),
        ("temperature", float("inf"), "temperature"),
        ("temperature", float("-inf"), "temperature"),
        ("top_p", 0, "top_p"),
        ("top_p", 1.01, "top_p"),
        ("top_p", float("nan"), "top_p"),
        ("top_p", float("inf"), "top_p"),
        ("top_p", float("-inf"), "top_p"),
        ("seed", -1, "seed"),
        ("seed", False, "seed"),
        ("schema_version", -1, "schema_version"),
        ("schema_version", 1.0, "schema_version"),
        ("enable_thinking", True, "enable_thinking"),
        ("enable_thinking", 0, "enable_thinking"),
        ("stop_after_accepted", 0, "stop_after_accepted"),
    ),
)
def test_config_rejects_invalid_boundaries_and_types(
    tmp_path: Path,
    field: str,
    value,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        make_config(tmp_path, **{field: value}).validate()


@pytest.mark.parametrize("missing_field", ("model_path", "input_path"))
def test_config_requires_existing_model_and_input(
    tmp_path: Path,
    missing_field: str,
) -> None:
    config = make_config(tmp_path)
    missing = tmp_path / f"missing-{missing_field}"

    with pytest.raises(ValueError, match=missing_field):
        replace(config, **{missing_field: missing}).validate()


def test_sha256_contract_and_fingerprint_are_exact(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    config.input_path.write_bytes(b"first\nsecond\n")

    assert sha256_file(config.input_path) == hashlib.sha256(b"first\nsecond\n").hexdigest()
    contract = build_generation_contract(config, source_sha256="abc")
    assert contract == {
        "schema_version": 1,
        "source_sha256": "abc",
        "model_path": str(config.model_path.resolve()),
        "topology": "dual_tp1",
        "batch_size": 64,
        "max_model_len": 512,
        "max_new_tokens": 256,
        "temperature": 0.2,
        "top_p": 0.95,
        "seed": 0,
        "enable_thinking": False,
    }
    canonical = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert fingerprint_contract(contract) == hashlib.sha256(canonical).hexdigest()


def test_build_manifest_resolves_cache_roots_to_absolute_logical_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    config = make_config(tmp_path, cache_root=Path("relative-cache"))
    manifest = manifest_for_rows(
        config,
        [],
        [],
        created_at="2026-06-13T00:00:00+00:00",
        updated_at="2026-06-13T00:00:01+00:00",
    )

    assert manifest["cache_roots"] == [
        str((tmp_path / "relative-cache" / "gpu0").resolve()),
        str((tmp_path / "relative-cache" / "gpu1").resolve()),
    ]


def test_resume_prefix_is_derived_from_combined_source_positions() -> None:
    rows = source_rows()
    state = derive_resume_state(
        rows,
        accepted=[{"id": "a"}, {"id": "c"}],
        rejected=[{"id": "b"}],
        created_at="created",
    )

    assert state.processed_count == 3
    assert state.last_committed_position == 2
    assert state.accepted == ({"id": "a"}, {"id": "c"})
    assert state.rejected == ({"id": "b"},)
    assert state.created_at == "created"


@pytest.mark.parametrize(
    ("rows", "accepted", "rejected", "match"),
    (
        (source_rows(), [{"id": "a"}, {"id": "d"}], [{"id": "b"}], "prefix"),
        (source_rows(), [{"id": "a"}], [{"id": "a"}], "duplicate"),
        (source_rows(), [{"id": "unknown"}], [], "unknown"),
        (source_rows(), [{"id": ""}], [], "nonempty"),
        ([{"id": "a"}, {"id": "a"}], [], [], "source.*duplicate"),
        ([{"id": ""}], [], [], "source.*nonempty"),
    ),
)
def test_resume_prefix_rejects_arbitrary_duplicate_unknown_and_empty_ids(
    rows: list[dict],
    accepted: list[dict],
    rejected: list[dict],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        derive_resume_state(rows, accepted, rejected, created_at="created")


def write_legacy_state(
    config: TeacherGenerationConfig,
    accepted: list[dict],
    rejected: list[dict],
    manifest: dict | None = None,
) -> TeacherStateStore:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(config.output_dir / "teacher_accepted_20k.jsonl", accepted)
    write_jsonl(config.output_dir / "teacher_rejected.jsonl", rejected)
    if manifest is not None:
        (config.output_dir / "manifest.json").write_text(
            json.dumps(manifest) + "\n",
            encoding="utf-8",
        )
    return TeacherStateStore(config.output_dir)


def test_legacy_rows_require_explicit_adoption(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    store = write_legacy_state(
        config,
        [accepted_row("a", [1, 2], 3)],
        [rejected_row("b", [2, 2], 4)],
    )

    with pytest.raises(ValueError, match="adopt_legacy_state"):
        store.load_resume_state(source_rows(), config, adopt_legacy_state=False)

    state = store.load_resume_state(source_rows(), config, adopt_legacy_state=True)
    assert state.processed_count == 2
    assert state.last_committed_position == 1


@pytest.mark.parametrize(
    ("accepted", "rejected", "target", "match"),
    (
        ([rejected_row("a", [1, 2], 3)], [], 3, "accepted.*incorrect"),
        ([], [accepted_row("a", [1, 2], 3)], 3, "rejected.*correct"),
        ([{"id": "a", "numbers": [1, 2], "target": 3}], [], 3, "response"),
        ([{"id": "a", "response": "<answer>1+2</answer>", "target": 3}], [], 3, "numbers"),
        ([{"id": "a", "response": "<answer>1+2</answer>", "numbers": [1, 2]}], [], 3, "target"),
        ([accepted_row("a", [1, 2], 3), accepted_row("b", [2, 2], 4)], [], 1, "target"),
    ),
)
def test_legacy_adoption_revalidates_rows_and_target(
    tmp_path: Path,
    accepted: list[dict],
    rejected: list[dict],
    target: int,
    match: str,
) -> None:
    config = make_config(tmp_path, stop_after_accepted=target)
    store = write_legacy_state(config, accepted, rejected)

    with pytest.raises(ValueError, match=match):
        store.load_resume_state(source_rows(), config, adopt_legacy_state=True)


def test_empty_state_initializes_without_legacy_adoption(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    state = TeacherStateStore(config.output_dir).load_resume_state(
        source_rows(),
        config,
        adopt_legacy_state=False,
    )

    assert state.processed_count == 0
    assert state.last_committed_position is None
    assert state.accepted == ()
    assert state.rejected == ()
    assert state.created_at


def test_resume_requires_store_output_dir_to_match_config_before_recovery(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    wrong_output = tmp_path / "wrong-output"
    wrong_output.mkdir()
    store = TeacherStateStore(wrong_output)
    store.transaction_path.write_text("{invalid", encoding="utf-8")

    with pytest.raises(ValueError, match="output_dir"):
        store.load_resume_state(source_rows(), config)

    assert store.transaction_path.exists()


def test_resume_requires_supplied_source_rows_to_equal_configured_jsonl(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    supplied = source_rows()
    supplied[1] = {**supplied[1], "target": 999}

    with pytest.raises(ValueError, match="source_rows.*input_path"):
        TeacherStateStore(config.output_dir).load_resume_state(supplied, config)


@pytest.mark.parametrize(
    "manifest",
    (
        {"stage": "teacher_accepted_pool"},
        {"generation_contract": {}},
        {
            "stage": "teacher_accepted_pool",
            "generation_contract": {},
            "schema_version": 1,
        },
    ),
)
def test_empty_v2_looking_manifest_without_fingerprint_is_corrupt(
    tmp_path: Path,
    manifest: dict,
) -> None:
    config = make_config(tmp_path)
    config.output_dir.mkdir()
    store = TeacherStateStore(config.output_dir)
    store.manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="corrupt.*manifest"):
        store.load_resume_state(source_rows(), config)


def materialize_v2_state(
    config: TeacherGenerationConfig,
    accepted: list[dict],
    rejected: list[dict],
    *,
    created_at: str = "2026-06-13T00:00:00+00:00",
) -> tuple[TeacherStateStore, dict]:
    store = write_legacy_state(config, accepted, rejected)
    source_hash = sha256_file(config.input_path)
    accepted_hash = sha256_file(store.accepted_path)
    rejected_hash = sha256_file(store.rejected_path)
    contract = build_generation_contract(config, source_sha256=source_hash)
    processed = len(accepted) + len(rejected)
    manifest = build_manifest(
        config=config,
        processed_count=processed,
        accepted_count=len(accepted),
        rejected_count=len(rejected),
        last_committed_position=processed - 1 if processed else None,
        completed=len(accepted) == config.stop_after_accepted,
        generation_contract=contract,
        source_sha256=source_hash,
        accepted_sha256=accepted_hash,
        rejected_sha256=rejected_hash,
        created_at=created_at,
        updated_at="2026-06-13T01:00:00+00:00",
    )
    store.manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return store, manifest


def test_v2_resume_verifies_exact_manifest_hashes_counts_and_contract(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    accepted = [accepted_row("a", [1, 2], 3)]
    rejected = [rejected_row("b", [2, 2], 4)]
    store, manifest = materialize_v2_state(config, accepted, rejected)

    state = store.load_resume_state(source_rows(), config, adopt_legacy_state=False)

    assert state.processed_count == 2
    assert state.created_at == manifest["created_at"]


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ({"processed_count": 1}, "processed_count"),
        ({"accepted_count": 0}, "accepted_count"),
        ({"rejected_count": 0}, "rejected_count"),
        ({"last_committed_position": 0}, "last_committed_position"),
        ({"target_accepted_count": 7}, "target"),
        ({"completed": True}, "completed"),
        ({"source_sha256": "0" * 64}, "source_sha256"),
        ({"accepted_sha256": "0" * 64}, "accepted_sha256"),
        ({"rejected_sha256": "0" * 64}, "rejected_sha256"),
        ({"generation_contract_fingerprint": "0" * 64}, "fingerprint"),
    ),
)
def test_v2_resume_rejects_manifest_mutations(
    tmp_path: Path,
    mutation: dict,
    match: str,
) -> None:
    config = make_config(tmp_path)
    store, manifest = materialize_v2_state(
        config,
        [accepted_row("a", [1, 2], 3)],
        [rejected_row("b", [2, 2], 4)],
    )
    manifest.update(mutation)
    store.manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        store.load_resume_state(source_rows(), config, adopt_legacy_state=False)


def test_v2_resume_requires_exact_integer_counts(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    store, manifest = materialize_v2_state(
        config,
        [accepted_row("a", [1, 2], 3)],
        [],
    )
    manifest["accepted_count"] = True
    store.manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="accepted_count.*exact integer"):
        store.load_resume_state(source_rows(), config, adopt_legacy_state=False)


def test_v2_resume_rejects_output_byte_mutation_and_immutable_config_change(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    store, _ = materialize_v2_state(
        config,
        [accepted_row("a", [1, 2], 3)],
        [],
    )
    store.accepted_path.write_text('{"id":"a","changed":true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="accepted_sha256"):
        store.load_resume_state(source_rows(), config, adopt_legacy_state=False)

    store, _ = materialize_v2_state(
        config,
        [accepted_row("a", [1, 2], 3)],
        [],
    )
    with pytest.raises(ValueError, match="generation contract"):
        store.load_resume_state(
            source_rows(),
            replace(config, temperature=0.7),
            adopt_legacy_state=False,
        )


def test_v2_resume_allows_operational_timeout_and_cache_changes(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    store, _ = materialize_v2_state(
        config,
        [accepted_row("a", [1, 2], 3)],
        [],
    )
    operationally_changed = replace(
        config,
        worker_timeout_seconds=12.5,
        gpu_memory_utilization=0.65,
        cache_root=tmp_path / "new-cache",
    )

    state = store.load_resume_state(
        source_rows(),
        operationally_changed,
        adopt_legacy_state=False,
    )
    assert state.processed_count == 1


def test_v2_resume_allows_current_device_change(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    store, _ = materialize_v2_state(
        config,
        [accepted_row("a", [1, 2], 3)],
        [],
    )

    state = store.load_resume_state(
        source_rows(),
        replace(config, devices=(2, 3)),
        adopt_legacy_state=False,
    )

    assert state.processed_count == 1


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("max_worker_batch_size", 31, "max_worker_batch_size"),
        ("devices", [0], "devices"),
        ("devices", [0, False], "devices"),
        ("worker_timeout_seconds", 0, "worker_timeout_seconds"),
        ("worker_timeout_seconds", float("nan"), "worker_timeout_seconds"),
        ("worker_timeout_seconds", float("inf"), "worker_timeout_seconds"),
        ("gpu_memory_utilization", 0, "gpu_memory_utilization"),
        ("gpu_memory_utilization", float("nan"), "gpu_memory_utilization"),
        ("gpu_memory_utilization", float("inf"), "gpu_memory_utilization"),
        ("cache_roots", [], "cache_roots"),
        ("cache_roots", ["relative/gpu0", "/tmp/cache/gpu1"], "cache_roots"),
        ("cache_roots", ["/tmp/cache/gpu1", "/tmp/cache/gpu0"], "cache_roots"),
        ("created_at", "not-a-time", "created_at"),
        ("created_at", "2026-06-13T00:00:00", "created_at"),
        ("created_at", "2026-06-13T08:00:00+08:00", "created_at"),
        ("updated_at", "not-a-time", "updated_at"),
        ("updated_at", "2026-06-12T23:59:59+00:00", "updated_at"),
    ),
)
def test_v2_resume_rejects_invalid_operational_manifest_fields(
    tmp_path: Path,
    field: str,
    value,
    match: str,
) -> None:
    config = make_config(tmp_path)
    store, manifest = materialize_v2_state(
        config,
        [accepted_row("a", [1, 2], 3)],
        [],
    )
    manifest[field] = value
    store.manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        store.load_resume_state(read_jsonl(config.input_path), config)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("stage", "other-stage", "stage"),
        ("model_path", "/different/model", "model_path"),
        ("source_path", "/different/source", "source_path"),
    ),
)
def test_v2_resume_rejects_manifest_fields_incoherent_with_contract(
    tmp_path: Path,
    field: str,
    value,
    match: str,
) -> None:
    config = make_config(tmp_path)
    store, manifest = materialize_v2_state(
        config,
        [accepted_row("a", [1, 2], 3)],
        [],
    )
    manifest[field] = value
    store.manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        store.load_resume_state(source_rows(), config, adopt_legacy_state=False)


def manifest_for_rows(
    config: TeacherGenerationConfig,
    accepted: list[dict],
    rejected: list[dict],
    *,
    created_at: str,
    updated_at: str,
) -> dict:
    processed = len(accepted) + len(rejected)
    source_hash = sha256_file(config.input_path)
    contract = build_generation_contract(config, source_sha256=source_hash)
    return build_manifest(
        config=config,
        processed_count=processed,
        accepted_count=len(accepted),
        rejected_count=len(rejected),
        last_committed_position=processed - 1 if processed else None,
        completed=len(accepted) == config.stop_after_accepted,
        generation_contract=contract,
        source_sha256=source_hash,
        accepted_sha256="store-fills-this",
        rejected_sha256="store-fills-this",
        created_at=created_at,
        updated_at=updated_at,
    )


def manifest_for_existing_snapshots(
    store: TeacherStateStore,
    config: TeacherGenerationConfig,
    accepted: list[dict],
    rejected: list[dict],
) -> dict:
    manifest = manifest_for_rows(
        config,
        accepted,
        rejected,
        created_at="2026-06-13T00:00:00+00:00",
        updated_at="2026-06-13T00:00:01+00:00",
    )
    manifest["accepted_sha256"] = sha256_file(store.accepted_path)
    manifest["rejected_sha256"] = sha256_file(store.rejected_path)
    return manifest


def commit_rows(
    store: TeacherStateStore,
    config: TeacherGenerationConfig,
    *,
    batch_id: int,
    start: int,
    accepted: list[dict],
    rejected: list[dict],
    created_at: str = "2026-06-13T00:00:00+00:00",
    updated_at: str | None = None,
) -> dict:
    if updated_at is None:
        updated_at = (
            datetime(2026, 6, 13, 1, tzinfo=timezone.utc)
            + timedelta(seconds=batch_id)
        ).isoformat()
    if (
        not store.manifest_path.exists()
        and not store.accepted_path.exists()
        and not store.rejected_path.exists()
        and (accepted or rejected)
    ):
        initial_manifest = manifest_for_rows(
            config,
            [],
            [],
            created_at=created_at,
            updated_at=created_at,
        )
        store.commit(
            batch_id=0,
            submitted_start=0,
            submitted_stop=0,
            accepted=[],
            rejected=[],
            manifest=initial_manifest,
        )
    manifest = manifest_for_rows(
        config,
        accepted,
        rejected,
        created_at=created_at,
        updated_at=updated_at,
    )
    store.commit(
        batch_id=batch_id,
        submitted_start=start,
        submitted_stop=len(accepted) + len(rejected),
        accepted=accepted,
        rejected=rejected,
        manifest=manifest,
    )
    return json.loads(store.manifest_path.read_text(encoding="utf-8"))


def test_initial_empty_commit_materializes_exact_files_and_hashes(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    store = TeacherStateStore(config.output_dir)
    manifest = manifest_for_rows(
        config,
        [],
        [],
        created_at="2026-06-13T00:00:00+00:00",
        updated_at="2026-06-13T00:00:01+00:00",
    )

    store.commit(
        batch_id=0,
        submitted_start=0,
        submitted_stop=0,
        accepted=[],
        rejected=[],
        manifest=manifest,
    )

    assert store.accepted_path.read_bytes() == b""
    assert store.rejected_path.read_bytes() == b""
    committed = json.loads(store.manifest_path.read_text(encoding="utf-8"))
    empty_hash = hashlib.sha256(b"").hexdigest()
    assert committed == {
        **manifest,
        "accepted_sha256": empty_hash,
        "rejected_sha256": empty_hash,
    }
    assert store.manifest_path.read_bytes().endswith(b"\n")
    assert not store.transaction_path.exists()


def test_initial_commit_allows_updated_at_equal_created_at(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    store = TeacherStateStore(config.output_dir)
    timestamp = "2026-06-13T00:00:00+00:00"
    manifest = manifest_for_rows(
        config,
        [],
        [],
        created_at=timestamp,
        updated_at=timestamp,
    )

    store.commit(
        batch_id=0,
        submitted_start=0,
        submitted_stop=0,
        accepted=[],
        rejected=[],
        manifest=manifest,
    )

    assert read_json(store.manifest_path)["updated_at"] == timestamp


def test_has_v2_manifest_requires_nonempty_fingerprint(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    store = TeacherStateStore(config.output_dir)

    assert store.has_v2_manifest() is False

    config.output_dir.mkdir()
    store.manifest_path.write_text(
        json.dumps({"generation_contract_fingerprint": ""}) + "\n",
        encoding="utf-8",
    )
    assert store.has_v2_manifest() is False

    store.manifest_path.write_text(
        json.dumps({"generation_contract_fingerprint": "fingerprint"}) + "\n",
        encoding="utf-8",
    )
    assert store.has_v2_manifest() is True


@pytest.mark.parametrize(
    ("accepted", "rejected", "target"),
    (
        (
            [accepted_row("a", [1, 2], 3)],
            [rejected_row("b", [2, 2], 4)],
            1,
        ),
        (
            [accepted_row("a", [1, 2], 3)],
            [
                rejected_row("b", [2, 2], 4),
                rejected_row("c", [3, 2], 5),
                rejected_row("d", [4, 2], 6),
            ],
            10,
        ),
    ),
)
def test_commit_materializes_manifest_over_nonempty_legacy_snapshots(
    tmp_path: Path,
    accepted: list[dict],
    rejected: list[dict],
    target: int,
) -> None:
    config = make_config(tmp_path, stop_after_accepted=target)
    store = write_legacy_state(config, accepted, rejected)
    accepted_bytes = store.accepted_path.read_bytes()
    rejected_bytes = store.rejected_path.read_bytes()
    manifest = manifest_for_existing_snapshots(
        store,
        config,
        accepted,
        rejected,
    )
    processed = len(accepted) + len(rejected)

    store.commit(
        batch_id=0,
        submitted_start=processed,
        submitted_stop=processed,
        accepted=accepted,
        rejected=rejected,
        manifest=manifest,
    )

    assert store.accepted_path.read_bytes() == accepted_bytes
    assert store.rejected_path.read_bytes() == rejected_bytes
    assert read_json(store.manifest_path) == manifest
    assert store.has_v2_manifest() is True
    assert not store.transaction_path.exists()


def test_commit_materializes_manifest_over_preexisting_empty_snapshots(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    store = write_legacy_state(config, [], [])
    manifest = manifest_for_existing_snapshots(store, config, [], [])

    store.commit(
        batch_id=0,
        submitted_start=0,
        submitted_stop=0,
        accepted=[],
        rejected=[],
        manifest=manifest,
    )

    assert store.accepted_path.read_bytes() == b""
    assert store.rejected_path.read_bytes() == b""
    assert read_json(store.manifest_path) == manifest


def test_snapshot_materialization_rejects_changed_rows_before_journal(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    accepted = [accepted_row("a", [1, 2], 3)]
    rejected = [rejected_row("b", [2, 2], 4)]
    store = write_legacy_state(config, accepted, rejected)
    changed_accepted = [{**accepted[0], "response": "<answer>2+1</answer>"}]
    manifest = manifest_for_rows(
        config,
        changed_accepted,
        rejected,
        created_at="2026-06-13T00:00:00+00:00",
        updated_at="2026-06-13T00:00:01+00:00",
    )

    with pytest.raises(ValueError, match="byte-for-byte"):
        store.commit(
            batch_id=0,
            submitted_start=2,
            submitted_stop=2,
            accepted=changed_accepted,
            rejected=rejected,
            manifest=manifest,
        )

    assert not store.transaction_path.exists()
    assert not store.manifest_path.exists()


@pytest.mark.parametrize(
    ("batch_id", "start", "stop", "match"),
    (
        (1, 2, 2, "batch_id"),
        (0, 1, 2, "zero-length"),
    ),
)
def test_snapshot_materialization_rejects_wrong_batch_or_range_before_journal(
    tmp_path: Path,
    batch_id: int,
    start: int,
    stop: int,
    match: str,
) -> None:
    config = make_config(tmp_path)
    accepted = [accepted_row("a", [1, 2], 3)]
    rejected = [rejected_row("b", [2, 2], 4)]
    store = write_legacy_state(config, accepted, rejected)
    manifest = manifest_for_existing_snapshots(
        store,
        config,
        accepted,
        rejected,
    )

    with pytest.raises(ValueError, match=match):
        store.commit(
            batch_id=batch_id,
            submitted_start=start,
            submitted_stop=stop,
            accepted=accepted,
            rejected=rejected,
            manifest=manifest,
        )

    assert not store.transaction_path.exists()
    assert not store.manifest_path.exists()


def test_snapshot_materialization_rejects_wrong_existing_count_before_journal(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    accepted = [accepted_row("a", [1, 2], 3)]
    rejected = [rejected_row("b", [2, 2], 4)]
    store = write_legacy_state(config, accepted, rejected)
    supplied_rejected: list[dict] = []
    manifest = manifest_for_rows(
        config,
        accepted,
        supplied_rejected,
        created_at="2026-06-13T00:00:00+00:00",
        updated_at="2026-06-13T00:00:01+00:00",
    )

    with pytest.raises(ValueError, match="existing snapshot row count"):
        store.commit(
            batch_id=0,
            submitted_start=1,
            submitted_stop=1,
            accepted=accepted,
            rejected=supplied_rejected,
            manifest=manifest,
        )

    assert not store.transaction_path.exists()
    assert not store.manifest_path.exists()


def test_no_manifest_commit_cannot_add_rows_to_absent_snapshots(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    store = TeacherStateStore(config.output_dir)
    accepted = [accepted_row("a", [1, 2], 3)]
    manifest = manifest_for_rows(
        config,
        accepted,
        [],
        created_at="2026-06-13T00:00:00+00:00",
        updated_at="2026-06-13T00:00:01+00:00",
    )

    with pytest.raises(ValueError, match="initial commit.*empty"):
        store.commit(
            batch_id=0,
            submitted_start=0,
            submitted_stop=1,
            accepted=accepted,
            rejected=[],
            manifest=manifest,
        )

    assert not store.transaction_path.exists()
    assert not store.manifest_path.exists()


class FailAfterDestination:
    def __init__(self, destination: Path) -> None:
        self.destination = destination
        self.failed = False

    def __call__(self, source: str | Path, destination: str | Path) -> None:
        os.replace(source, destination)
        if Path(destination) == self.destination and not self.failed:
            self.failed = True
            raise OSError(f"injected failure after {self.destination.name}")


def test_failed_snapshot_materialization_recovers_exact_files_and_no_manifest(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    accepted = [accepted_row("a", [1, 2], 3)]
    rejected = [rejected_row("b", [2, 2], 4)]
    initial_store = write_legacy_state(config, accepted, rejected)
    accepted_bytes = initial_store.accepted_path.read_bytes()
    rejected_bytes = initial_store.rejected_path.read_bytes()
    manifest = manifest_for_existing_snapshots(
        initial_store,
        config,
        accepted,
        rejected,
    )
    failing_store = TeacherStateStore(
        config.output_dir,
        replace_file=FailAfterDestination(initial_store.rejected_path),
    )

    with pytest.raises(OSError, match="injected failure"):
        failing_store.commit(
            batch_id=0,
            submitted_start=2,
            submitted_stop=2,
            accepted=accepted,
            rejected=rejected,
            manifest=manifest,
        )

    assert failing_store.transaction_path.exists()
    TeacherStateStore(config.output_dir).recover_transaction()
    assert initial_store.accepted_path.read_bytes() == accepted_bytes
    assert initial_store.rejected_path.read_bytes() == rejected_bytes
    assert not initial_store.manifest_path.exists()
    assert not initial_store.transaction_path.exists()


@pytest.mark.parametrize(
    "failed_name",
    (
        "teacher_accepted_20k.jsonl",
        "teacher_rejected.jsonl",
        "manifest.json",
    ),
)
def test_recovery_rolls_back_exact_old_state_after_each_replacement(
    tmp_path: Path,
    failed_name: str,
) -> None:
    config = make_config(tmp_path)
    old_store = TeacherStateStore(config.output_dir)
    old_accepted = [accepted_row("a", [1, 2], 3)]
    old_rejected = [rejected_row("b", [2, 2], 4)]
    old_manifest = commit_rows(
        old_store,
        config,
        batch_id=1,
        start=0,
        accepted=old_accepted,
        rejected=old_rejected,
    )
    old_accepted_bytes = old_store.accepted_path.read_bytes()
    old_rejected_bytes = old_store.rejected_path.read_bytes()
    old_accepted_hash = sha256_file(old_store.accepted_path)
    old_rejected_hash = sha256_file(old_store.rejected_path)

    failing_store = TeacherStateStore(
        config.output_dir,
        replace_file=FailAfterDestination(config.output_dir / failed_name),
    )
    new_accepted = [*old_accepted, accepted_row("c", [3, 2], 5)]
    new_rejected = old_rejected
    with pytest.raises(OSError, match="injected failure"):
        commit_rows(
            failing_store,
            config,
            batch_id=2,
            start=2,
            accepted=new_accepted,
            rejected=new_rejected,
            created_at=old_manifest["created_at"],
        )

    assert failing_store.transaction_path.exists()
    TeacherStateStore(config.output_dir).recover_transaction()

    assert old_store.accepted_path.read_bytes() == old_accepted_bytes
    assert old_store.rejected_path.read_bytes() == old_rejected_bytes
    assert sha256_file(old_store.accepted_path) == old_accepted_hash
    assert sha256_file(old_store.rejected_path) == old_rejected_hash
    assert read_json(old_store.manifest_path) == old_manifest
    assert not old_store.transaction_path.exists()


def test_recovery_restores_all_absent_pre_state(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    failing_store = TeacherStateStore(
        config.output_dir,
        replace_file=FailAfterDestination(config.output_dir / ACCEPTED_NAME),
    )
    manifest = manifest_for_rows(
        config,
        [],
        [],
        created_at="2026-06-13T00:00:00+00:00",
        updated_at="2026-06-13T00:00:00+00:00",
    )

    with pytest.raises(OSError, match="injected failure"):
        failing_store.commit(
            batch_id=0,
            submitted_start=0,
            submitted_stop=0,
            accepted=[],
            rejected=[],
            manifest=manifest,
        )

    assert failing_store.transaction_path.exists()
    TeacherStateStore(config.output_dir).recover_transaction()
    assert not failing_store.accepted_path.exists()
    assert not failing_store.rejected_path.exists()
    assert not failing_store.manifest_path.exists()
    assert not failing_store.transaction_path.exists()


ACCEPTED_NAME = "teacher_accepted_20k.jsonl"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def leave_failed_transaction(
    tmp_path: Path,
) -> tuple[TeacherGenerationConfig, TeacherStateStore, dict]:
    config = make_config(tmp_path)
    store = TeacherStateStore(config.output_dir)
    old_accepted = [accepted_row("a", [1, 2], 3)]
    old_manifest = commit_rows(
        store,
        config,
        batch_id=0,
        start=0,
        accepted=old_accepted,
        rejected=[],
    )
    failing = TeacherStateStore(
        config.output_dir,
        replace_file=FailAfterDestination(store.manifest_path),
    )
    with pytest.raises(OSError):
        commit_rows(
            failing,
            config,
            batch_id=1,
            start=1,
            accepted=[*old_accepted, accepted_row("c", [3, 2], 5)],
            rejected=[],
            created_at=old_manifest["created_at"],
        )
    return config, failing, old_manifest


def test_recovery_rolls_back_when_all_replacements_finished_but_journal_remains(
    tmp_path: Path,
) -> None:
    _, store, old_manifest = leave_failed_transaction(tmp_path)
    assert read_json(store.manifest_path)["processed_count"] == 2

    store = TeacherStateStore(store.output_dir)
    store.recover_transaction()

    assert read_json(store.accepted_path)["id"] == "a"
    assert store.rejected_path.read_bytes() == b""
    assert read_json(store.manifest_path) == old_manifest
    assert not store.transaction_path.exists()


@pytest.mark.parametrize(
    ("damage", "match"),
    (
        ("too_short", "at least"),
        ("prefix_mismatch", "prefix"),
        ("path_injection", "journal keys"),
    ),
)
def test_unrecoverable_journal_hard_fails_and_remains(
    tmp_path: Path,
    damage: str,
    match: str,
) -> None:
    _, store, _ = leave_failed_transaction(tmp_path)
    if damage == "too_short":
        store.accepted_path.write_bytes(b"")
    elif damage == "prefix_mismatch":
        store.accepted_path.write_text('{"id":"wrong"}\n', encoding="utf-8")
    else:
        journal = read_json(store.transaction_path)
        journal["accepted_path"] = str(tmp_path / "attacker-controlled")
        store.transaction_path.write_text(json.dumps(journal) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        TeacherStateStore(store.output_dir).recover_transaction()

    assert store.transaction_path.exists()


def test_invalid_json_journal_hard_fails_and_remains(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    config.output_dir.mkdir(parents=True)
    store = TeacherStateStore(config.output_dir)
    store.transaction_path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(ValueError, match="transaction journal"):
        store.recover_transaction()

    assert store.transaction_path.exists()


def test_journal_range_must_match_recorded_pre_state_and_manifest(
    tmp_path: Path,
) -> None:
    _, store, _ = leave_failed_transaction(tmp_path)
    journal = read_json(store.transaction_path)
    journal["submitted_start"] = 0
    store.transaction_path.write_text(json.dumps(journal) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="submitted_start"):
        TeacherStateStore(store.output_dir).recover_transaction()

    assert store.transaction_path.exists()


def test_recovery_fully_validates_embedded_pre_manifest(
    tmp_path: Path,
) -> None:
    _, store, _ = leave_failed_transaction(tmp_path)
    journal = read_json(store.transaction_path)
    journal["manifest"]["payload"]["max_worker_batch_size"] = 1
    store.transaction_path.write_text(json.dumps(journal) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="max_worker_batch_size"):
        TeacherStateStore(store.output_dir).recover_transaction()

    assert store.transaction_path.exists()


@pytest.mark.parametrize(
    ("batch_id", "start", "stop", "manifest_change", "match"),
    (
        (False, 0, 0, {}, "batch_id"),
        (0.0, 0, 0, {}, "batch_id"),
        (0, False, 0, {}, "submitted_start"),
        (0, 1, 0, {}, "range"),
        (0, 0, 1, {}, "processed_count"),
        (0, 0, 0, {"processed_count": False}, "processed_count.*exact integer"),
        (0, 0, 0, {"accepted_count": 1}, "accepted_count"),
        (0, 0, 0, {"last_committed_position": 0}, "last_committed_position"),
        (0, 0, 0, {"source_sha256": "0" * 64}, "source_sha256"),
        (
            0,
            0,
            0,
            {"generation_contract_fingerprint": "0" * 64},
            "fingerprint",
        ),
    ),
)
def test_commit_rejects_range_count_and_contract_incoherence_before_replacement(
    tmp_path: Path,
    batch_id,
    start,
    stop,
    manifest_change: dict,
    match: str,
) -> None:
    config = make_config(tmp_path)
    replacements: list[tuple[Path, Path]] = []

    def record_replace(source: str | Path, destination: str | Path) -> None:
        replacements.append((Path(source), Path(destination)))
        os.replace(source, destination)

    store = TeacherStateStore(config.output_dir, replace_file=record_replace)
    manifest = manifest_for_rows(
        config,
        [],
        [],
        created_at="2026-06-13T00:00:00+00:00",
        updated_at="2026-06-13T00:00:01+00:00",
    )
    manifest.update(manifest_change)

    with pytest.raises(ValueError, match=match):
        store.commit(
            batch_id=batch_id,
            submitted_start=start,
            submitted_stop=stop,
            accepted=[],
            rejected=[],
            manifest=manifest,
        )

    assert replacements == []
    assert not store.transaction_path.exists()


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("max_worker_batch_size", 31, "max_worker_batch_size"),
        ("devices", [0, False], "devices"),
        ("worker_timeout_seconds", float("nan"), "worker_timeout_seconds"),
        ("gpu_memory_utilization", float("inf"), "gpu_memory_utilization"),
        ("cache_roots", ["relative/gpu0", "relative/gpu1"], "cache_roots"),
        ("created_at", "not-a-time", "created_at"),
        ("updated_at", "2026-06-12T23:59:59+00:00", "updated_at"),
    ),
)
def test_commit_rejects_invalid_operational_manifest_before_replacement(
    tmp_path: Path,
    field: str,
    value,
    match: str,
) -> None:
    config = make_config(tmp_path)
    replacements: list[tuple[Path, Path]] = []

    def record_replace(source: str | Path, destination: str | Path) -> None:
        replacements.append((Path(source), Path(destination)))
        os.replace(source, destination)

    store = TeacherStateStore(config.output_dir, replace_file=record_replace)
    manifest = manifest_for_rows(
        config,
        [],
        [],
        created_at="2026-06-13T00:00:00+00:00",
        updated_at="2026-06-13T00:00:01+00:00",
    )
    manifest[field] = value

    with pytest.raises(ValueError, match=match):
        store.commit(
            batch_id=0,
            submitted_start=0,
            submitted_stop=0,
            accepted=[],
            rejected=[],
            manifest=manifest,
        )

    assert replacements == []
    assert not store.transaction_path.exists()


def test_commit_rejects_duplicate_output_ids_before_replacement(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    store = TeacherStateStore(config.output_dir)
    accepted = [accepted_row("a", [1, 2], 3)]
    rejected = [rejected_row("a", [1, 2], 3)]
    manifest = manifest_for_rows(
        config,
        accepted,
        rejected,
        created_at="2026-06-13T00:00:00+00:00",
        updated_at="2026-06-13T00:00:01+00:00",
    )

    with pytest.raises(ValueError, match="duplicate"):
        store.commit(
            batch_id=0,
            submitted_start=0,
            submitted_stop=2,
            accepted=accepted,
            rejected=rejected,
            manifest=manifest,
        )

    assert not store.transaction_path.exists()


def test_commit_requires_previous_processed_count_to_match_submitted_start(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    store = TeacherStateStore(config.output_dir)
    old_manifest = commit_rows(
        store,
        config,
        batch_id=0,
        start=0,
        accepted=[],
        rejected=[],
    )

    with pytest.raises(ValueError, match="submitted_start"):
        commit_rows(
            store,
            config,
            batch_id=2,
            start=1,
            accepted=[accepted_row("a", [1, 2], 3)],
            rejected=[],
            created_at=old_manifest["created_at"],
        )

    assert not store.transaction_path.exists()


def test_commit_preserves_created_at_from_previous_manifest(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    store = TeacherStateStore(config.output_dir)
    old_manifest = commit_rows(
        store,
        config,
        batch_id=0,
        start=0,
        accepted=[],
        rejected=[],
    )

    with pytest.raises(ValueError, match="created_at"):
        commit_rows(
            store,
            config,
            batch_id=1,
            start=0,
            accepted=[accepted_row("a", [1, 2], 3)],
            rejected=[],
            created_at="2026-06-13T09:00:00+00:00",
        )

    assert read_json(store.manifest_path) == old_manifest
    assert not store.transaction_path.exists()


@pytest.mark.parametrize(
    "updated_at",
    (
        "2026-06-13T01:00:00+00:00",
        "2026-06-13T00:59:59+00:00",
    ),
)
def test_non_initial_commit_requires_updated_at_strictly_later(
    tmp_path: Path,
    updated_at: str,
) -> None:
    config = make_config(tmp_path)
    store = TeacherStateStore(config.output_dir)
    old_manifest = commit_rows(
        store,
        config,
        batch_id=0,
        start=0,
        accepted=[],
        rejected=[],
    )

    with pytest.raises(ValueError, match="updated_at.*strictly later"):
        commit_rows(
            store,
            config,
            batch_id=1,
            start=0,
            accepted=[accepted_row("a", [1, 2], 3)],
            rejected=[],
            created_at=old_manifest["created_at"],
            updated_at=updated_at,
        )

    assert not store.transaction_path.exists()


def test_non_initial_commit_accepts_strictly_later_updated_at(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    store = TeacherStateStore(config.output_dir)
    old_manifest = commit_rows(
        store,
        config,
        batch_id=0,
        start=0,
        accepted=[],
        rejected=[],
    )

    committed = commit_rows(
        store,
        config,
        batch_id=1,
        start=0,
        accepted=[accepted_row("a", [1, 2], 3)],
        rejected=[],
        created_at=old_manifest["created_at"],
        updated_at="2026-06-13T01:00:01+00:00",
    )

    assert committed["updated_at"] == "2026-06-13T01:00:01+00:00"


@pytest.mark.parametrize(
    "changed_config",
    (
        {"schema_version": 2},
        {"topology": "dual_tp1_changed"},
        {"batch_size": 32},
        {"max_model_len": 1024},
        {"max_new_tokens": 128},
        {"temperature": 0.7},
        {"top_p": 0.8},
        {"seed": 9},
    ),
)
def test_commit_rejects_immutable_generation_transition_before_journal(
    tmp_path: Path,
    changed_config: dict,
) -> None:
    config = make_config(tmp_path)
    store = TeacherStateStore(config.output_dir)
    old_manifest = commit_rows(
        store,
        config,
        batch_id=0,
        start=0,
        accepted=[],
        rejected=[],
    )
    next_config = replace(config, **changed_config)
    next_manifest = manifest_for_rows(
        next_config,
        [accepted_row("a", [1, 2], 3)],
        [],
        created_at=old_manifest["created_at"],
        updated_at="2026-06-13T02:00:00+00:00",
    )

    with pytest.raises(ValueError, match="immutable"):
        store.commit(
            batch_id=1,
            submitted_start=0,
            submitted_stop=1,
            accepted=[accepted_row("a", [1, 2], 3)],
            rejected=[],
            manifest=next_manifest,
        )

    assert not store.transaction_path.exists()


def test_commit_rejects_immutable_model_and_source_path_transition(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    store = TeacherStateStore(config.output_dir)
    old_manifest = commit_rows(
        store,
        config,
        batch_id=0,
        start=0,
        accepted=[],
        rejected=[],
    )
    new_model = tmp_path / "new-model"
    new_model.mkdir()
    new_source = tmp_path / "new-source.jsonl"
    write_jsonl(new_source, source_rows())
    next_config = replace(
        config,
        model_path=new_model,
        input_path=new_source,
    )
    next_manifest = manifest_for_rows(
        next_config,
        [accepted_row("a", [1, 2], 3)],
        [],
        created_at=old_manifest["created_at"],
        updated_at="2026-06-13T02:00:00+00:00",
    )

    with pytest.raises(ValueError, match="immutable"):
        store.commit(
            batch_id=1,
            submitted_start=0,
            submitted_stop=1,
            accepted=[accepted_row("a", [1, 2], 3)],
            rejected=[],
            manifest=next_manifest,
        )

    assert not store.transaction_path.exists()


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ({"processed_count": 0}, "processed_count"),
        ({"accepted_count": 0}, "accepted_count"),
        ({"accepted_sha256": "0" * 64}, "accepted_sha256"),
        ({"stage": "wrong-stage"}, "stage"),
    ),
)
def test_commit_rejects_incoherent_pre_state_before_journal_and_recovery_is_noop(
    tmp_path: Path,
    mutation: dict,
    match: str,
) -> None:
    config = make_config(tmp_path)
    store = TeacherStateStore(config.output_dir)
    old_accepted = [accepted_row("a", [1, 2], 3)]
    old_manifest = commit_rows(
        store,
        config,
        batch_id=0,
        start=0,
        accepted=old_accepted,
        rejected=[],
    )
    old_accepted_bytes = store.accepted_path.read_bytes()
    damaged_manifest = {**old_manifest, **mutation}
    store.manifest_path.write_text(
        json.dumps(damaged_manifest) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=match):
        commit_rows(
            store,
            config,
            batch_id=1,
            start=1,
            accepted=[*old_accepted, accepted_row("c", [3, 2], 5)],
            rejected=[],
            created_at=old_manifest["created_at"],
        )

    assert not store.transaction_path.exists()
    assert store.accepted_path.read_bytes() == old_accepted_bytes
    assert read_json(store.manifest_path) == damaged_manifest
    store.recover_transaction()
    assert read_json(store.manifest_path) == damaged_manifest


def test_posix_directory_open_error_propagates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(teacher_state.os, "name", "posix")
    monkeypatch.setattr(
        teacher_state.os,
        "open",
        lambda path, flags: (_ for _ in ()).throw(OSError(errno.EIO, "disk error")),
    )

    with pytest.raises(OSError) as exc_info:
        teacher_state._fsync_directory_path(tmp_path)

    assert exc_info.value.errno == errno.EIO


def test_posix_directory_fsync_error_propagates_and_closes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed: list[int] = []
    monkeypatch.setattr(teacher_state.os, "name", "posix")
    monkeypatch.setattr(teacher_state.os, "open", lambda path, flags: 41)
    monkeypatch.setattr(
        teacher_state.os,
        "fsync",
        lambda descriptor: (_ for _ in ()).throw(OSError(errno.EIO, "disk error")),
    )
    monkeypatch.setattr(teacher_state.os, "close", closed.append)

    with pytest.raises(OSError) as exc_info:
        teacher_state._fsync_directory_path(tmp_path)

    assert exc_info.value.errno == errno.EIO
    assert closed == [41]


@pytest.mark.parametrize("failure_point", ("open", "fsync"))
def test_windows_unsupported_directory_fsync_is_treated_as_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    closed: list[int] = []
    monkeypatch.setattr(teacher_state.os, "name", "nt")
    if failure_point == "open":
        monkeypatch.setattr(
            teacher_state.os,
            "open",
            lambda path, flags: (_ for _ in ()).throw(
                OSError(errno.EACCES, "directory handles unsupported")
            ),
        )
    else:
        monkeypatch.setattr(teacher_state.os, "open", lambda path, flags: 42)
        monkeypatch.setattr(
            teacher_state.os,
            "fsync",
            lambda descriptor: (_ for _ in ()).throw(
                OSError(errno.EINVAL, "directory fsync unsupported")
            ),
        )
        monkeypatch.setattr(teacher_state.os, "close", closed.append)

    teacher_state._fsync_directory_path(tmp_path)

    assert closed == ([] if failure_point == "open" else [42])


def test_windows_directory_eio_still_propagates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(teacher_state.os, "name", "nt")
    monkeypatch.setattr(teacher_state.os, "open", lambda path, flags: 43)
    monkeypatch.setattr(
        teacher_state.os,
        "fsync",
        lambda descriptor: (_ for _ in ()).throw(OSError(errno.EIO, "disk error")),
    )
    monkeypatch.setattr(teacher_state.os, "close", lambda descriptor: None)

    with pytest.raises(OSError) as exc_info:
        teacher_state._fsync_directory_path(tmp_path)

    assert exc_info.value.errno == errno.EIO


class FailNthCall:
    def __init__(self, call_number: int, message: str) -> None:
        self.call_number = call_number
        self.message = message
        self.calls = 0

    def __call__(self, path: Path) -> None:
        self.calls += 1
        if self.calls == self.call_number:
            raise OSError(errno.EIO, self.message)


class FailCalls:
    def __init__(self, failures: dict[int, str]) -> None:
        self.failures = failures
        self.calls = 0

    def __call__(self, path: Path) -> None:
        self.calls += 1
        if self.calls in self.failures:
            raise OSError(errno.EIO, self.failures[self.calls])


def test_commit_file_fsync_failure_preserves_journal_for_recovery(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    failing_fsync = FailNthCall(1, "file fsync failed")
    store = TeacherStateStore(config.output_dir, fsync_file=failing_fsync)
    manifest = manifest_for_rows(
        config,
        [],
        [],
        created_at="2026-06-13T00:00:00+00:00",
        updated_at="2026-06-13T00:00:01+00:00",
    )

    with pytest.raises(OSError, match="file fsync failed"):
        store.commit(
            batch_id=0,
            submitted_start=0,
            submitted_stop=0,
            accepted=[],
            rejected=[],
            manifest=manifest,
        )

    assert store.transaction_path.exists()
    TeacherStateStore(config.output_dir).recover_transaction()
    assert not store.transaction_path.exists()


def test_commit_final_directory_fsync_failure_restores_journal(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    old_store = TeacherStateStore(config.output_dir)
    old_manifest = commit_rows(
        old_store,
        config,
        batch_id=0,
        start=0,
        accepted=[],
        rejected=[],
    )
    failing_directory_fsync = FailNthCall(5, "final directory fsync failed")
    store = TeacherStateStore(
        config.output_dir,
        fsync_directory=failing_directory_fsync,
    )

    with pytest.raises(OSError, match="final directory fsync failed"):
        commit_rows(
            store,
            config,
            batch_id=1,
            start=0,
            accepted=[accepted_row("a", [1, 2], 3)],
            rejected=[],
            created_at=old_manifest["created_at"],
        )

    assert store.transaction_path.exists()
    TeacherStateStore(config.output_dir).recover_transaction()
    assert read_json(store.manifest_path) == old_manifest
    assert not store.transaction_path.exists()


def test_recovery_final_directory_fsync_failure_restores_journal(
    tmp_path: Path,
) -> None:
    _, store, _ = leave_failed_transaction(tmp_path)
    journal_bytes = store.transaction_path.read_bytes()
    failing_directory_fsync = FailNthCall(4, "recovery directory fsync failed")
    recovering_store = TeacherStateStore(
        store.output_dir,
        fsync_directory=failing_directory_fsync,
    )

    with pytest.raises(OSError, match="recovery directory fsync failed"):
        recovering_store.recover_transaction()

    assert recovering_store.transaction_path.read_bytes() == journal_bytes
    TeacherStateStore(store.output_dir).recover_transaction()
    assert not store.transaction_path.exists()


def test_commit_reports_final_and_marker_directory_fsync_failures(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    old_store = TeacherStateStore(config.output_dir)
    old_manifest = commit_rows(
        old_store,
        config,
        batch_id=0,
        start=0,
        accepted=[],
        rejected=[],
    )
    failures = FailCalls(
        {
            5: "final directory fsync failed",
            6: "marker directory fsync failed",
        }
    )
    store = TeacherStateStore(config.output_dir, fsync_directory=failures)

    with pytest.raises(BaseException) as exc_info:
        commit_rows(
            store,
            config,
            batch_id=1,
            start=0,
            accepted=[accepted_row("a", [1, 2], 3)],
            rejected=[],
            created_at=old_manifest["created_at"],
        )

    text = str(exc_info.value)
    assert "final directory fsync failed" in text
    assert "marker directory fsync failed" in text
    assert store.transaction_path.exists()


def test_recovery_reports_final_and_marker_directory_fsync_failures(
    tmp_path: Path,
) -> None:
    _, store, _ = leave_failed_transaction(tmp_path)
    failures = FailCalls(
        {
            4: "recovery directory fsync failed",
            5: "marker directory fsync failed",
        }
    )
    recovering = TeacherStateStore(store.output_dir, fsync_directory=failures)

    with pytest.raises(BaseException) as exc_info:
        recovering.recover_transaction()

    text = str(exc_info.value)
    assert "recovery directory fsync failed" in text
    assert "marker directory fsync failed" in text
    assert recovering.transaction_path.exists()
