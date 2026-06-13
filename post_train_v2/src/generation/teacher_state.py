from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from post_train.src.countdown.io import read_jsonl
from post_train.src.countdown.validation import validate_countdown_response


ACCEPTED_FILENAME = "teacher_accepted_20k.jsonl"
REJECTED_FILENAME = "teacher_rejected.jsonl"
MANIFEST_FILENAME = "manifest.json"
TRANSACTION_FILENAME = ".teacher_pool.transaction.json"
STAGE = "teacher_accepted_pool"
TRANSACTION_SCHEMA_VERSION = 1
CONTRACT_KEYS = {
    "schema_version",
    "source_sha256",
    "model_path",
    "topology",
    "batch_size",
    "max_model_len",
    "max_new_tokens",
    "temperature",
    "top_p",
    "seed",
    "enable_thinking",
}
JOURNAL_KEYS = {
    "schema_version",
    "batch_id",
    "submitted_start",
    "submitted_stop",
    "accepted",
    "rejected",
    "manifest",
}
SNAPSHOT_KEYS = {"existed", "row_count", "sha256"}
MANIFEST_SNAPSHOT_KEYS = {"existed", "payload"}

MANIFEST_KEYS = {
    "schema_version",
    "stage",
    "model_path",
    "source_path",
    "topology",
    "devices",
    "batch_size",
    "max_worker_batch_size",
    "worker_timeout_seconds",
    "gpu_memory_utilization",
    "max_model_len",
    "max_new_tokens",
    "temperature",
    "top_p",
    "seed",
    "enable_thinking",
    "cache_roots",
    "processed_count",
    "accepted_count",
    "rejected_count",
    "last_committed_position",
    "target_accepted_count",
    "completed",
    "generation_contract",
    "generation_contract_fingerprint",
    "source_sha256",
    "accepted_sha256",
    "rejected_sha256",
    "created_at",
    "updated_at",
}


def _is_exact_int(value: object) -> bool:
    return type(value) is int


def _require_positive_int(name: str, value: object) -> None:
    if not _is_exact_int(value) or value <= 0:
        raise ValueError(f"{name} must be a positive exact integer")


def _require_number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    return float(value)


@dataclass(frozen=True)
class TeacherGenerationConfig:
    model_path: Path
    input_path: Path
    output_dir: Path
    devices: tuple[int, int]
    topology: str
    batch_size: int
    worker_timeout_seconds: float
    gpu_memory_utilization: float
    max_model_len: int
    max_new_tokens: int
    temperature: float
    top_p: float
    seed: int
    enable_thinking: bool
    stop_after_accepted: int
    cache_root: Path
    schema_version: int

    def validate(self) -> None:
        if not isinstance(self.devices, tuple) or len(self.devices) != 2:
            raise ValueError("devices must contain exactly two entries")
        if any(not _is_exact_int(device) for device in self.devices):
            raise ValueError("devices must be exact integers")
        if any(device < 0 for device in self.devices):
            raise ValueError("devices must be nonnegative")
        if self.devices[0] == self.devices[1]:
            raise ValueError("devices must be distinct")
        if self.topology != "dual_tp1":
            raise ValueError("topology must be dual_tp1")

        _require_positive_int("batch_size", self.batch_size)
        timeout = _require_number(
            "worker_timeout_seconds", self.worker_timeout_seconds
        )
        if timeout <= 0:
            raise ValueError("worker_timeout_seconds must be positive")
        utilization = _require_number(
            "gpu_memory_utilization", self.gpu_memory_utilization
        )
        if not 0 < utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        _require_positive_int("max_model_len", self.max_model_len)
        _require_positive_int("max_new_tokens", self.max_new_tokens)
        temperature = _require_number("temperature", self.temperature)
        if temperature < 0:
            raise ValueError("temperature must be nonnegative")
        top_p = _require_number("top_p", self.top_p)
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if not _is_exact_int(self.seed) or self.seed < 0:
            raise ValueError("seed must be a nonnegative exact integer")
        if type(self.enable_thinking) is not bool or self.enable_thinking:
            raise ValueError("enable_thinking must be False")
        _require_positive_int("stop_after_accepted", self.stop_after_accepted)
        if not _is_exact_int(self.schema_version) or self.schema_version < 0:
            raise ValueError("schema_version must be a nonnegative exact integer")
        if not self.model_path.exists():
            raise ValueError(f"model_path does not exist: {self.model_path}")
        if not self.input_path.exists():
            raise ValueError(f"input_path does not exist: {self.input_path}")

    def resolved(self) -> TeacherGenerationConfig:
        return replace(
            self,
            model_path=self.model_path.resolve(),
            input_path=self.input_path.resolve(),
            output_dir=self.output_dir.resolve(),
            cache_root=self.cache_root.resolve(),
        )


@dataclass(frozen=True)
class ResumeState:
    accepted: tuple[dict[str, Any], ...]
    rejected: tuple[dict[str, Any], ...]
    processed_count: int
    last_committed_position: int | None
    created_at: str


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def build_generation_contract(
    config: TeacherGenerationConfig,
    *,
    source_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": config.schema_version,
        "source_sha256": source_sha256,
        "model_path": str(config.model_path.resolve()),
        "topology": config.topology,
        "batch_size": config.batch_size,
        "max_model_len": config.max_model_len,
        "max_new_tokens": config.max_new_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "seed": config.seed,
        "enable_thinking": config.enable_thinking,
    }


def fingerprint_contract(contract: dict[str, Any]) -> str:
    payload = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _row_id(row: dict[str, Any], *, label: str) -> Any:
    if not isinstance(row, dict):
        raise ValueError(f"{label} row must be an object")
    if "id" not in row:
        raise ValueError(f"{label} row id must be nonempty")
    row_id = row["id"]
    if row_id is None or row_id == "":
        raise ValueError(f"{label} row id must be nonempty")
    try:
        hash(row_id)
    except TypeError as exc:
        raise ValueError(f"{label} row id must be hashable") from exc
    return row_id


def _source_positions(source_rows: list[dict[str, Any]]) -> dict[Any, int]:
    positions: dict[Any, int] = {}
    for position, row in enumerate(source_rows):
        row_id = _row_id(row, label="source")
        if row_id in positions:
            raise ValueError(f"source contains duplicate id: {row_id}")
        positions[row_id] = position
    return positions


def derive_resume_state(
    source_rows: list[dict[str, Any]],
    accepted: Iterable[dict[str, Any]],
    rejected: Iterable[dict[str, Any]],
    *,
    created_at: str,
) -> ResumeState:
    accepted_tuple = tuple(accepted)
    rejected_tuple = tuple(rejected)
    positions = _source_positions(source_rows)
    output_positions: list[int] = []
    seen: set[Any] = set()
    for row in (*accepted_tuple, *rejected_tuple):
        row_id = _row_id(row, label="output")
        if row_id in seen:
            raise ValueError(f"teacher outputs contain duplicate id: {row_id}")
        seen.add(row_id)
        if row_id not in positions:
            raise ValueError(f"teacher outputs contain unknown id: {row_id}")
        output_positions.append(positions[row_id])

    output_positions.sort()
    processed_count = len(output_positions)
    if output_positions != list(range(processed_count)):
        raise ValueError("teacher output IDs do not form an exact source prefix")
    return ResumeState(
        accepted=accepted_tuple,
        rejected=rejected_tuple,
        processed_count=processed_count,
        last_committed_position=processed_count - 1 if processed_count else None,
        created_at=created_at,
    )


def build_manifest(
    *,
    config: TeacherGenerationConfig,
    processed_count: int,
    accepted_count: int,
    rejected_count: int,
    last_committed_position: int | None,
    completed: bool,
    generation_contract: dict[str, Any],
    source_sha256: str,
    accepted_sha256: str,
    rejected_sha256: str,
    created_at: str,
    updated_at: str,
) -> dict[str, Any]:
    return {
        "schema_version": config.schema_version,
        "stage": STAGE,
        "model_path": str(config.model_path.resolve()),
        "source_path": str(config.input_path.resolve()),
        "topology": config.topology,
        "devices": list(config.devices),
        "batch_size": config.batch_size,
        "max_worker_batch_size": (config.batch_size + 1) // 2,
        "worker_timeout_seconds": config.worker_timeout_seconds,
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "max_model_len": config.max_model_len,
        "max_new_tokens": config.max_new_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "seed": config.seed,
        "enable_thinking": config.enable_thinking,
        "cache_roots": [
            str(config.cache_root / "gpu0"),
            str(config.cache_root / "gpu1"),
        ],
        "processed_count": processed_count,
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "last_committed_position": last_committed_position,
        "target_accepted_count": config.stop_after_accepted,
        "completed": completed,
        "generation_contract": generation_contract,
        "generation_contract_fingerprint": fingerprint_contract(
            generation_contract
        ),
        "source_sha256": source_sha256,
        "accepted_sha256": accepted_sha256,
        "rejected_sha256": rejected_sha256,
        "created_at": created_at,
        "updated_at": updated_at,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fsync_file_path(path: Path) -> None:
    with path.open("rb+") as handle:
        os.fsync(handle.fileno())


def _fsync_directory_path(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON file: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return value


def _require_legacy_fields(row: dict[str, Any], *, kind: str) -> tuple[str, list[int], int]:
    for field in ("response", "numbers", "target", "id"):
        if field not in row:
            raise ValueError(f"legacy {kind} row missing {field}")
    response = row["response"]
    numbers = row["numbers"]
    target = row["target"]
    if not isinstance(response, str):
        raise ValueError(f"legacy {kind} row response must be a string")
    if not isinstance(numbers, list):
        raise ValueError(f"legacy {kind} row numbers must be a list")
    if not _is_exact_int(target):
        raise ValueError(f"legacy {kind} row target must be an exact integer")
    return response, numbers, target


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")


def _jsonl_suffix_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(row, ensure_ascii=False) + "\n"
        ).encode("utf-8")
        for row in rows
    )


def _rows_from_jsonl_bytes(data: bytes, *, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not valid UTF-8") from exc
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label}:{line_number}: invalid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{label}:{line_number}: row must be an object")
        rows.append(row)
    return rows


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _prefix_with_hash(
    data: bytes,
    *,
    row_count: int,
    expected_hash: str,
    label: str,
) -> bytes:
    current_rows = _rows_from_jsonl_bytes(data, label=label)
    if len(current_rows) < row_count:
        raise ValueError(
            f"{label} must contain at least {row_count} rows for recovery"
        )

    candidates: list[bytes] = []
    seen_rows = 0
    if row_count == 0:
        candidates.append(b"")
    offset = 0
    for line in data.splitlines(keepends=True):
        offset += len(line)
        if line.strip():
            seen_rows += 1
        if seen_rows == row_count:
            candidates.append(data[:offset])
        elif seen_rows > row_count:
            break
    for candidate in candidates:
        if _sha256_bytes(candidate) == expected_hash:
            return candidate
    raise ValueError(f"{label} prefix hash does not match transaction journal")


def _validate_exact_nonnegative_int(name: str, value: object) -> int:
    if not _is_exact_int(value) or value < 0:
        raise ValueError(f"{name} must be a nonnegative exact integer")
    return value


class TeacherStateStore:
    def __init__(
        self,
        output_dir: str | Path,
        *,
        replace_file: Callable[[str | Path, str | Path], Any] = os.replace,
        now: Callable[[], str] = _utc_now,
        fsync_file: Callable[[Path], None] = _fsync_file_path,
        fsync_directory: Callable[[Path], None] = _fsync_directory_path,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.accepted_path = self.output_dir / ACCEPTED_FILENAME
        self.rejected_path = self.output_dir / REJECTED_FILENAME
        self.manifest_path = self.output_dir / MANIFEST_FILENAME
        self.transaction_path = self.output_dir / TRANSACTION_FILENAME
        self._replace_file = replace_file
        self._now = now
        self._fsync_file = fsync_file
        self._fsync_directory = fsync_directory

    def load_resume_state(
        self,
        source_rows: list[dict[str, Any]],
        config: TeacherGenerationConfig,
        adopt_legacy_state: bool = False,
    ) -> ResumeState:
        config.validate()
        self.recover_transaction()
        accepted = read_jsonl(self.accepted_path) if self.accepted_path.exists() else []
        rejected = read_jsonl(self.rejected_path) if self.rejected_path.exists() else []
        manifest = _read_json(self.manifest_path) if self.manifest_path.exists() else None
        has_rows = bool(accepted or rejected)
        is_v2 = bool(
            manifest
            and manifest.get("generation_contract_fingerprint") is not None
        )

        if has_rows and not is_v2:
            if not adopt_legacy_state:
                raise ValueError(
                    "existing rows require adopt_legacy_state=True"
                )
            if len(accepted) > config.stop_after_accepted:
                raise ValueError("legacy accepted count exceeds configured target")
            for row in accepted:
                response, numbers, target = _require_legacy_fields(
                    row, kind="accepted"
                )
                if not validate_countdown_response(response, numbers, target).ok:
                    raise ValueError("legacy accepted row is incorrect")
            for row in rejected:
                response, numbers, target = _require_legacy_fields(
                    row, kind="rejected"
                )
                if validate_countdown_response(response, numbers, target).ok:
                    raise ValueError("legacy rejected row is correct")
            return derive_resume_state(
                source_rows,
                accepted,
                rejected,
                created_at=self._now(),
            )

        if not is_v2:
            return derive_resume_state(
                source_rows,
                accepted,
                rejected,
                created_at=self._now(),
            )

        assert manifest is not None
        if set(manifest) != MANIFEST_KEYS:
            missing = sorted(MANIFEST_KEYS - set(manifest))
            extra = sorted(set(manifest) - MANIFEST_KEYS)
            raise ValueError(
                f"manifest keys mismatch; missing={missing}, extra={extra}"
            )
        state = derive_resume_state(
            source_rows,
            accepted,
            rejected,
            created_at=manifest["created_at"],
        )
        source_hash = sha256_file(config.input_path)
        expected_contract = build_generation_contract(
            config, source_sha256=source_hash
        )
        if manifest["generation_contract"] != expected_contract:
            raise ValueError("generation contract mismatch")
        if (
            manifest["generation_contract_fingerprint"]
            != fingerprint_contract(expected_contract)
        ):
            raise ValueError("generation contract fingerprint mismatch")
        if manifest["stage"] != STAGE:
            raise ValueError(f"manifest stage must be {STAGE}")
        if manifest["model_path"] != expected_contract["model_path"]:
            raise ValueError("manifest model_path disagrees with generation contract")
        if manifest["source_path"] != str(config.input_path.resolve()):
            raise ValueError("manifest source_path disagrees with configured source")
        for field in (
            "schema_version",
            "topology",
            "batch_size",
            "max_model_len",
            "max_new_tokens",
            "temperature",
            "top_p",
            "seed",
            "enable_thinking",
        ):
            if manifest[field] != expected_contract[field]:
                raise ValueError(
                    f"manifest {field} disagrees with generation contract"
                )
        for field in (
            "processed_count",
            "accepted_count",
            "rejected_count",
            "target_accepted_count",
        ):
            if not _is_exact_int(manifest[field]):
                raise ValueError(f"manifest {field} must be an exact integer")
        last_position = manifest["last_committed_position"]
        if last_position is not None and not _is_exact_int(last_position):
            raise ValueError(
                "manifest last_committed_position must be an exact integer or null"
            )
        checks = {
            "source_sha256": source_hash,
            "accepted_sha256": sha256_file(self.accepted_path),
            "rejected_sha256": sha256_file(self.rejected_path),
            "processed_count": state.processed_count,
            "accepted_count": len(accepted),
            "rejected_count": len(rejected),
            "last_committed_position": state.last_committed_position,
            "target_accepted_count": config.stop_after_accepted,
        }
        for field, expected in checks.items():
            if manifest[field] != expected:
                raise ValueError(
                    f"manifest {field} mismatch: "
                    f"expected {expected!r}, got {manifest[field]!r}"
                )
        if len(accepted) > config.stop_after_accepted:
            raise ValueError("accepted count exceeds configured target")
        expected_completed = len(accepted) == config.stop_after_accepted
        if type(manifest["completed"]) is not bool:
            raise ValueError("manifest completed must be a boolean")
        if manifest["completed"] != expected_completed:
            raise ValueError("manifest completed is incoherent")
        return state

    def commit(
        self,
        *,
        batch_id: int,
        submitted_start: int,
        submitted_stop: int,
        accepted: list[dict[str, Any]],
        rejected: list[dict[str, Any]],
        manifest: dict[str, Any],
    ) -> None:
        self._validate_commit(
            batch_id=batch_id,
            submitted_start=submitted_start,
            submitted_stop=submitted_stop,
            accepted=accepted,
            rejected=rejected,
            manifest=manifest,
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.transaction_path.exists():
            raise ValueError("unrecovered transaction journal exists")

        accepted_pre = self._snapshot(self.accepted_path)
        rejected_pre = self._snapshot(self.rejected_path)
        manifest_pre = (
            _read_json(self.manifest_path) if self.manifest_path.exists() else None
        )
        previous_count = (
            manifest_pre.get("processed_count")
            if manifest_pre is not None
            else accepted_pre["row_count"] + rejected_pre["row_count"]
        )
        if not _is_exact_int(previous_count) or previous_count != submitted_start:
            raise ValueError(
                "submitted_start does not match previous processed count"
            )
        if (
            manifest_pre is not None
            and "created_at" in manifest_pre
            and manifest["created_at"] != manifest_pre["created_at"]
        ):
            raise ValueError("manifest created_at must be preserved")

        accepted_bytes = self._build_snapshot_bytes(
            self.accepted_path,
            accepted_pre,
            accepted,
            label="accepted",
        )
        rejected_bytes = self._build_snapshot_bytes(
            self.rejected_path,
            rejected_pre,
            rejected,
            label="rejected",
        )
        committed_manifest = dict(manifest)
        committed_manifest["accepted_sha256"] = _sha256_bytes(accepted_bytes)
        committed_manifest["rejected_sha256"] = _sha256_bytes(rejected_bytes)
        manifest_bytes = _json_bytes(committed_manifest)
        journal = {
            "schema_version": TRANSACTION_SCHEMA_VERSION,
            "batch_id": batch_id,
            "submitted_start": submitted_start,
            "submitted_stop": submitted_stop,
            "accepted": {
                "existed": accepted_pre["existed"],
                "row_count": accepted_pre["row_count"],
                "sha256": accepted_pre["sha256"],
            },
            "rejected": {
                "existed": rejected_pre["existed"],
                "row_count": rejected_pre["row_count"],
                "sha256": rejected_pre["sha256"],
            },
            "manifest": {
                "existed": manifest_pre is not None,
                "payload": manifest_pre,
            },
        }

        owner = f"{batch_id}.{os.getpid()}.{uuid.uuid4().hex}"
        temp_paths: list[Path] = []
        try:
            accepted_temp = self._write_temp(
                self.accepted_path, accepted_bytes, owner
            )
            temp_paths.append(accepted_temp)
            rejected_temp = self._write_temp(
                self.rejected_path, rejected_bytes, owner
            )
            temp_paths.append(rejected_temp)
            manifest_temp = self._write_temp(
                self.manifest_path, manifest_bytes, owner
            )
            temp_paths.append(manifest_temp)
            journal_temp = self._write_temp(
                self.transaction_path, _json_bytes(journal), owner
            )
            temp_paths.append(journal_temp)

            self._replace_file(journal_temp, self.transaction_path)
            temp_paths.remove(journal_temp)
            self._fsync_file(self.transaction_path)
            self._fsync_directory(self.output_dir)

            for temp_path, destination in (
                (accepted_temp, self.accepted_path),
                (rejected_temp, self.rejected_path),
                (manifest_temp, self.manifest_path),
            ):
                self._replace_file(temp_path, destination)
                temp_paths.remove(temp_path)
                self._fsync_file(destination)
                self._fsync_directory(self.output_dir)

            self.transaction_path.unlink()
            self._fsync_directory(self.output_dir)
        finally:
            for temp_path in temp_paths:
                try:
                    temp_path.unlink()
                except FileNotFoundError:
                    pass

    def recover_transaction(self) -> None:
        if not self.transaction_path.exists():
            return
        try:
            journal = _read_json(self.transaction_path)
        except ValueError as exc:
            raise ValueError("invalid transaction journal") from exc
        self._validate_journal(journal)

        accepted_restore = self._prepare_snapshot_restore(
            self.accepted_path,
            journal["accepted"],
            label="accepted snapshot",
        )
        rejected_restore = self._prepare_snapshot_restore(
            self.rejected_path,
            journal["rejected"],
            label="rejected snapshot",
        )
        manifest_state = journal["manifest"]
        manifest_restore = (
            _json_bytes(manifest_state["payload"])
            if manifest_state["existed"]
            else None
        )

        owner = (
            f"recovery.{journal['batch_id']}.{os.getpid()}."
            f"{uuid.uuid4().hex}"
        )
        temp_paths: list[Path] = []
        replacements: list[tuple[Path, Path]] = []
        try:
            for destination, restored in (
                (self.accepted_path, accepted_restore),
                (self.rejected_path, rejected_restore),
                (self.manifest_path, manifest_restore),
            ):
                if restored is not None:
                    temp = self._write_temp(destination, restored, owner)
                    temp_paths.append(temp)
                    replacements.append((temp, destination))

            replacement_by_destination = {
                destination: temp for temp, destination in replacements
            }
            for destination, restored in (
                (self.accepted_path, accepted_restore),
                (self.rejected_path, rejected_restore),
                (self.manifest_path, manifest_restore),
            ):
                if restored is None:
                    try:
                        destination.unlink()
                    except FileNotFoundError:
                        pass
                    self._fsync_directory(self.output_dir)
                    continue
                temp = replacement_by_destination[destination]
                self._replace_file(temp, destination)
                temp_paths.remove(temp)
                self._fsync_file(destination)
                self._fsync_directory(self.output_dir)

            for path, snapshot in (
                (self.accepted_path, journal["accepted"]),
                (self.rejected_path, journal["rejected"]),
            ):
                if snapshot["existed"]:
                    if sha256_file(path) != snapshot["sha256"]:
                        raise ValueError(
                            f"restored {path.name} hash verification failed"
                        )
                elif path.exists():
                    raise ValueError(f"failed to restore absence of {path.name}")
            if manifest_state["existed"]:
                if _read_json(self.manifest_path) != manifest_state["payload"]:
                    raise ValueError("restored manifest verification failed")
            elif self.manifest_path.exists():
                raise ValueError("failed to restore manifest absence")

            self.transaction_path.unlink()
            self._fsync_directory(self.output_dir)
        finally:
            for temp_path in temp_paths:
                try:
                    temp_path.unlink()
                except FileNotFoundError:
                    pass

    def _validate_commit(
        self,
        *,
        batch_id: object,
        submitted_start: object,
        submitted_stop: object,
        accepted: object,
        rejected: object,
        manifest: object,
    ) -> None:
        _validate_exact_nonnegative_int("batch_id", batch_id)
        start = _validate_exact_nonnegative_int(
            "submitted_start", submitted_start
        )
        stop = _validate_exact_nonnegative_int("submitted_stop", submitted_stop)
        if stop < start:
            raise ValueError("submitted range must be half-open and nondecreasing")
        if not isinstance(accepted, list) or not all(
            isinstance(row, dict) for row in accepted
        ):
            raise ValueError("accepted must be a list of objects")
        if not isinstance(rejected, list) or not all(
            isinstance(row, dict) for row in rejected
        ):
            raise ValueError("rejected must be a list of objects")
        if not isinstance(manifest, dict) or set(manifest) != MANIFEST_KEYS:
            raise ValueError("manifest keys mismatch")

        for field in (
            "processed_count",
            "accepted_count",
            "rejected_count",
            "target_accepted_count",
        ):
            if not _is_exact_int(manifest[field]):
                raise ValueError(f"manifest {field} must be an exact integer")
        last_position = manifest["last_committed_position"]
        if last_position is not None and not _is_exact_int(last_position):
            raise ValueError(
                "manifest last_committed_position must be an exact integer or null"
            )
        expected_values = {
            "processed_count": stop,
            "accepted_count": len(accepted),
            "rejected_count": len(rejected),
            "last_committed_position": stop - 1 if stop else None,
        }
        for field, expected in expected_values.items():
            if manifest[field] != expected:
                raise ValueError(f"manifest {field} is incoherent")
        if len(accepted) + len(rejected) != stop:
            raise ValueError("manifest processed_count does not match output rows")
        seen_ids: set[Any] = set()
        for row in (*accepted, *rejected):
            row_id = _row_id(row, label="output")
            if row_id in seen_ids:
                raise ValueError(f"teacher outputs contain duplicate id: {row_id}")
            seen_ids.add(row_id)

        target = manifest["target_accepted_count"]
        _require_positive_int("target_accepted_count", target)
        if len(accepted) > target:
            raise ValueError("accepted_count exceeds target")
        if type(manifest["completed"]) is not bool:
            raise ValueError("manifest completed must be a boolean")
        if manifest["completed"] != (len(accepted) == target):
            raise ValueError("manifest completed is incoherent")
        if manifest["stage"] != STAGE:
            raise ValueError(f"manifest stage must be {STAGE}")

        contract = manifest["generation_contract"]
        if not isinstance(contract, dict) or set(contract) != CONTRACT_KEYS:
            raise ValueError("generation contract keys mismatch")
        if manifest["generation_contract_fingerprint"] != fingerprint_contract(
            contract
        ):
            raise ValueError("generation contract fingerprint mismatch")
        if manifest["source_sha256"] != contract["source_sha256"]:
            raise ValueError("manifest source_sha256 disagrees with contract")
        source_path = Path(manifest["source_path"])
        if not source_path.exists() or sha256_file(source_path) != manifest[
            "source_sha256"
        ]:
            raise ValueError("manifest source_sha256 does not match source file")
        aligned_fields = (
            "schema_version",
            "model_path",
            "topology",
            "batch_size",
            "max_model_len",
            "max_new_tokens",
            "temperature",
            "top_p",
            "seed",
            "enable_thinking",
        )
        for field in aligned_fields:
            if manifest[field] != contract[field]:
                raise ValueError(
                    f"manifest {field} disagrees with generation contract"
                )

    def _snapshot(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {
                "existed": False,
                "row_count": 0,
                "sha256": None,
                "bytes": b"",
                "rows": [],
            }
        data = path.read_bytes()
        rows = _rows_from_jsonl_bytes(data, label=path.name)
        return {
            "existed": True,
            "row_count": len(rows),
            "sha256": _sha256_bytes(data),
            "bytes": data,
            "rows": rows,
        }

    def _build_snapshot_bytes(
        self,
        path: Path,
        snapshot: dict[str, Any],
        rows: list[dict[str, Any]],
        *,
        label: str,
    ) -> bytes:
        previous_rows = snapshot["rows"]
        if len(rows) < len(previous_rows) or rows[: len(previous_rows)] != previous_rows:
            raise ValueError(f"{label} rows do not preserve committed prefix")
        old_bytes = snapshot["bytes"]
        suffix = rows[len(previous_rows) :]
        if suffix and old_bytes and not old_bytes.endswith(b"\n"):
            raise ValueError(f"{path.name} is not newline-terminated")
        return old_bytes + _jsonl_suffix_bytes(suffix)

    def _write_temp(self, destination: Path, data: bytes, owner: str) -> Path:
        temp_path = destination.with_name(
            f".{destination.name}.{owner}.tmp"
        )
        with temp_path.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        return temp_path

    def _validate_journal(self, journal: dict[str, Any]) -> None:
        if set(journal) != JOURNAL_KEYS:
            raise ValueError("transaction journal keys mismatch")
        if journal["schema_version"] != TRANSACTION_SCHEMA_VERSION:
            raise ValueError("transaction journal schema_version mismatch")
        _validate_exact_nonnegative_int("journal batch_id", journal["batch_id"])
        start = _validate_exact_nonnegative_int(
            "journal submitted_start", journal["submitted_start"]
        )
        stop = _validate_exact_nonnegative_int(
            "journal submitted_stop", journal["submitted_stop"]
        )
        if stop < start:
            raise ValueError("transaction journal range is invalid")
        for name in ("accepted", "rejected"):
            snapshot = journal[name]
            if not isinstance(snapshot, dict) or set(snapshot) != SNAPSHOT_KEYS:
                raise ValueError(f"transaction journal {name} snapshot keys mismatch")
            if type(snapshot["existed"]) is not bool:
                raise ValueError(
                    f"transaction journal {name} existed must be boolean"
                )
            _validate_exact_nonnegative_int(
                f"transaction journal {name} row_count",
                snapshot["row_count"],
            )
            digest = snapshot["sha256"]
            if snapshot["existed"]:
                if (
                    not isinstance(digest, str)
                    or len(digest) != 64
                    or any(character not in "0123456789abcdef" for character in digest)
                ):
                    raise ValueError(
                        f"transaction journal {name} sha256 is invalid"
                    )
            elif snapshot["row_count"] != 0 or digest is not None:
                raise ValueError(
                    f"transaction journal absent {name} snapshot is incoherent"
                )
        manifest = journal["manifest"]
        if (
            not isinstance(manifest, dict)
            or set(manifest) != MANIFEST_SNAPSHOT_KEYS
        ):
            raise ValueError("transaction journal manifest snapshot keys mismatch")
        if type(manifest["existed"]) is not bool:
            raise ValueError(
                "transaction journal manifest existed must be boolean"
            )
        if manifest["existed"]:
            if not isinstance(manifest["payload"], dict):
                raise ValueError(
                    "transaction journal manifest payload must be an object"
                )
        elif manifest["payload"] is not None:
            raise ValueError(
                "transaction journal absent manifest payload must be null"
            )
        pre_count = (
            journal["accepted"]["row_count"]
            + journal["rejected"]["row_count"]
        )
        if start != pre_count:
            raise ValueError(
                "transaction journal submitted_start does not match pre-state counts"
            )
        if manifest["existed"]:
            payload = manifest["payload"]
            manifest_checks = {
                "processed_count": start,
                "accepted_count": journal["accepted"]["row_count"],
                "rejected_count": journal["rejected"]["row_count"],
                "accepted_sha256": journal["accepted"]["sha256"],
                "rejected_sha256": journal["rejected"]["sha256"],
            }
            for field, expected in manifest_checks.items():
                if field in payload and payload[field] != expected:
                    raise ValueError(
                        f"transaction journal pre-manifest {field} is incoherent"
                    )

    def _prepare_snapshot_restore(
        self,
        path: Path,
        snapshot: dict[str, Any],
        *,
        label: str,
    ) -> bytes | None:
        if not snapshot["existed"]:
            return None
        if not path.exists():
            raise ValueError(f"{label} must contain at least the pre-state rows")
        return _prefix_with_hash(
            path.read_bytes(),
            row_count=snapshot["row_count"],
            expected_hash=snapshot["sha256"],
            label=label,
        )
