from __future__ import annotations

import hashlib
import json
import os
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from post_train.src.countdown.io import write_jsonl
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
        input_path.write_text('{"id":"a"}\n', encoding="utf-8")
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
        ("gpu_memory_utilization", 0, "gpu_memory_utilization"),
        ("gpu_memory_utilization", 1.01, "gpu_memory_utilization"),
        ("max_model_len", 0, "max_model_len"),
        ("max_new_tokens", 0, "max_new_tokens"),
        ("temperature", -0.01, "temperature"),
        ("top_p", 0, "top_p"),
        ("top_p", 1.01, "top_p"),
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
        cache_root=tmp_path / "new-cache",
    )

    state = store.load_resume_state(
        source_rows(),
        operationally_changed,
        adopt_legacy_state=False,
    )
    assert state.processed_count == 1


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


def commit_rows(
    store: TeacherStateStore,
    config: TeacherGenerationConfig,
    *,
    batch_id: int,
    start: int,
    accepted: list[dict],
    rejected: list[dict],
    created_at: str = "2026-06-13T00:00:00+00:00",
    updated_at: str = "2026-06-13T01:00:00+00:00",
) -> dict:
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


class FailAfterDestination:
    def __init__(self, destination: Path) -> None:
        self.destination = destination
        self.failed = False

    def __call__(self, source: str | Path, destination: str | Path) -> None:
        os.replace(source, destination)
        if Path(destination) == self.destination and not self.failed:
            self.failed = True
            raise OSError(f"injected failure after {self.destination.name}")


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
    accepted = [accepted_row("a", [1, 2], 3)]

    with pytest.raises(OSError, match="injected failure"):
        commit_rows(
            failing_store,
            config,
            batch_id=0,
            start=0,
            accepted=accepted,
            rejected=[],
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
