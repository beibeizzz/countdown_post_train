from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable


def _find_repo_root(script_path: Path) -> Path:
    for parent in script_path.resolve().parents:
        if (parent / "post_train").is_dir() and (parent / "post_train_v2").is_dir():
            return parent
    raise RuntimeError(f"could not locate repository root from {script_path}")


REPO_ROOT = _find_repo_root(Path(__file__))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from post_train_v2.src.config.loading import load_yaml
from post_train_v2.src.countdown.validation import (
    serialize_fraction,
    validate_countdown_response,
)
from post_train_v2.src.generation.output_lock import OutputLock
from post_train_v2.src.generation.parallel_vllm import (
    ParallelVLLMEngine,
    PositionedPrompt,
    WorkerReady,
    WorkerSpec,
    split_contiguous,
)
from post_train_v2.src.generation.seeding import derive_request_seed
from post_train_v2.src.generation.teacher_state import (
    TeacherGenerationConfig,
    TeacherStateStore,
    build_generation_contract,
    build_manifest,
)


LOGGER = logging.getLogger(__name__)
PRODUCTION_CONFIG = (
    REPO_ROOT / "post_train_v2/configs/generation/teacher_rollout_2gpu.yaml"
)
SMOKE_CONFIG = (
    REPO_ROOT / "post_train_v2/configs/generation/teacher_rollout_2gpu_smoke.yaml"
)
CONFIG_FIELDS = {
    "model_path",
    "input_path",
    "output_dir",
    "devices",
    "topology",
    "batch_size",
    "worker_timeout_seconds",
    "gpu_memory_utilization",
    "max_model_len",
    "max_new_tokens",
    "temperature",
    "top_p",
    "seed",
    "enable_thinking",
    "stop_after_accepted",
    "cache_root",
    "schema_version",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the V2 dual-GPU teacher accepted pool."
    )
    parser.add_argument("--config", default=str(PRODUCTION_CONFIG))
    parser.add_argument("--recover-stale-lock", action="store_true")
    parser.add_argument("--adopt-legacy-state", action="store_true")
    return parser.parse_args(argv)


def load_teacher_config(
    config_path: str | Path,
    *,
    repo_root: Path = REPO_ROOT,
) -> TeacherGenerationConfig:
    path = _resolve_path(config_path, repo_root)
    raw = load_yaml(path)
    missing = sorted(CONFIG_FIELDS - set(raw))
    extra = sorted(set(raw) - CONFIG_FIELDS)
    if missing or extra:
        raise ValueError(
            f"teacher config fields mismatch; missing={missing}, extra={extra}"
        )

    cache_value = raw["cache_root"]
    if not isinstance(cache_value, (str, Path)):
        raise ValueError("cache_root must be an absolute path")
    cache_root = Path(cache_value)
    if not cache_root.is_absolute() and not str(cache_value).startswith("/"):
        raise ValueError("cache_root must be an absolute path")
    devices = raw["devices"]
    if not isinstance(devices, list):
        raise ValueError("devices must be a list")

    config = TeacherGenerationConfig(
        model_path=_resolve_path(raw["model_path"], repo_root),
        input_path=_resolve_path(raw["input_path"], repo_root),
        output_dir=_resolve_path(raw["output_dir"], repo_root),
        devices=tuple(devices),
        topology=raw["topology"],
        batch_size=raw["batch_size"],
        worker_timeout_seconds=raw["worker_timeout_seconds"],
        gpu_memory_utilization=raw["gpu_memory_utilization"],
        max_model_len=raw["max_model_len"],
        max_new_tokens=raw["max_new_tokens"],
        temperature=raw["temperature"],
        top_p=raw["top_p"],
        seed=raw["seed"],
        enable_thinking=raw["enable_thinking"],
        stop_after_accepted=raw["stop_after_accepted"],
        cache_root=cache_root,
        schema_version=raw["schema_version"],
    ).resolved()
    config.validate()
    return config


def _resolve_path(value: str | Path, repo_root: Path) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (repo_root / candidate).resolve()


def validate_cuda_visibility(
    config: TeacherGenerationConfig,
    value: str | None = None,
) -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES") if value is None else value
    if visible is None or not visible.strip():
        return
    actual = [item.strip() for item in visible.split(",")]
    expected = [str(device) for device in config.devices]
    if (
        any(not item for item in actual)
        or len(set(actual)) != len(actual)
        or actual != expected
    ):
        raise ValueError(
            "CUDA_VISIBLE_DEVICES must be unset/blank or the exact ordered list "
            f"{','.join(expected)}. Reordered, remapped, duplicate, missing, "
            "extra, and UUID masks are rejected because workers set child masks "
            "to configured physical device indices."
        )


def validate_source_rows(rows: list[dict[str, Any]]) -> None:
    seen: set[Any] = set()
    for position, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"source row {position} must be an object")
        row_id = row.get("id")
        if row_id is None or row_id == "":
            raise ValueError(f"source row {position} id must be nonempty")
        try:
            hash(row_id)
        except TypeError as exc:
            raise ValueError(f"source row {position} id must be hashable") from exc
        if row_id in seen:
            raise ValueError(f"source contains duplicate id: {row_id}")
        seen.add(row_id)

        prompt = row.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"source row {position} prompt must be a nonempty string")
        numbers = row.get("numbers")
        if not isinstance(numbers, list):
            raise ValueError(f"source row {position} numbers must be a list")
        if any(type(number) is not int for number in numbers):
            raise ValueError(
                f"source row {position} numbers must contain exact integers"
            )
        if type(row.get("target")) is not int:
            raise ValueError(f"source row {position} target must be an exact integer")


def _parse_jsonl_bytes(data: bytes, path: Path) -> list[dict[str, Any]]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path}: invalid UTF-8") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(io.StringIO(text, newline=None), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: JSONL row must be an object")
        rows.append(row)
    return rows


def _stat_signature(stat_result: os.stat_result) -> tuple[int, ...]:
    return (
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_size,
        stat_result.st_mtime_ns,
        stat_result.st_ctime_ns,
    )


def _read_source_snapshot(path: Path) -> tuple[bytes, tuple[int, ...], str]:
    before = _stat_signature(path.stat())
    data = path.read_bytes()
    after = _stat_signature(path.stat())
    if before != after:
        raise ValueError(f"source file changed while being read: {path}")
    return data, after, hashlib.sha256(data).hexdigest()


def _verify_source_snapshot(
    path: Path,
    expected_stat: tuple[int, ...],
    expected_sha256: str,
) -> None:
    before = _stat_signature(path.stat())
    data = path.read_bytes()
    after = _stat_signature(path.stat())
    digest = hashlib.sha256(data).hexdigest()
    if before != after or after != expected_stat or digest != expected_sha256:
        raise ValueError(f"source file changed during resume validation: {path}")


def build_teacher_payload(
    row: dict[str, Any],
    response: str,
) -> dict[str, Any]:
    text = response.strip()
    result = validate_countdown_response(text, row["numbers"], row["target"])
    return {
        **row,
        "response": text,
        "validation": {
            "ok": result.ok,
            "error": result.error,
            "value": serialize_fraction(result.value),
            "used_numbers": result.used_numbers,
            "expression": result.expression,
        },
        "provenance": {
            "generator": "qwen3-8b-teacher",
            "stage": "teacher",
            "rollout_index": 0,
        },
    }


def _jsonl_sha256(rows: list[dict[str, Any]]) -> str:
    payload = b"".join(
        (json.dumps(row, ensure_ascii=False) + os.linesep).encode("utf-8")
        for row in rows
    )
    return hashlib.sha256(payload).hexdigest()


def _coerce_utc(value: datetime | str) -> datetime:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("now provider must return a timezone-aware timestamp")
    return parsed.astimezone(timezone.utc)


class _IncreasingUtcClock:
    def __init__(
        self,
        now: Callable[[], datetime | str],
        *,
        after: str,
    ) -> None:
        self._now = now
        self._last = _coerce_utc(after)

    def next(self) -> str:
        candidate = _coerce_utc(self._now())
        if candidate <= self._last:
            candidate = self._last + timedelta(microseconds=1)
        self._last = candidate
        return candidate.isoformat()


def _default_now() -> datetime:
    return datetime.now(timezone.utc)


def _snapshot_sha256(
    store: Any,
    path_attribute: str,
    rows: list[dict[str, Any]],
) -> str:
    path = getattr(store, path_attribute, None)
    if path is not None and Path(path).exists():
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return _jsonl_sha256(rows)


def _manifest(
    *,
    config: TeacherGenerationConfig,
    accepted: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    source_sha256: str,
    generation_contract: dict[str, Any],
    created_at: str,
    updated_at: str,
    accepted_sha256: str | None = None,
    rejected_sha256: str | None = None,
) -> dict[str, Any]:
    processed_count = len(accepted) + len(rejected)
    return build_manifest(
        config=config,
        processed_count=processed_count,
        accepted_count=len(accepted),
        rejected_count=len(rejected),
        last_committed_position=processed_count - 1 if processed_count else None,
        completed=len(accepted) == config.stop_after_accepted,
        generation_contract=generation_contract,
        source_sha256=source_sha256,
        accepted_sha256=accepted_sha256 or _jsonl_sha256(accepted),
        rejected_sha256=rejected_sha256 or _jsonl_sha256(rejected),
        created_at=created_at,
        updated_at=updated_at,
    )


def _build_engine(
    config: TeacherGenerationConfig,
    engine_factory: Callable[..., Any],
) -> Any:
    specs = (
        WorkerSpec(0, config.devices[0], str(config.cache_root / "gpu0")),
        WorkerSpec(1, config.devices[1], str(config.cache_root / "gpu1")),
    )
    return engine_factory(
        model_path=str(config.model_path),
        worker_specs=specs,
        gpu_memory_utilization=config.gpu_memory_utilization,
        max_model_len=config.max_model_len,
        seed=config.seed,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        enable_thinking=config.enable_thinking,
        timeout_seconds=config.worker_timeout_seconds,
    )


def _validated_worker_runtime_info(
    engine: Any,
) -> tuple[WorkerReady, WorkerReady]:
    info = engine.worker_runtime_info
    if not isinstance(info, tuple) or len(info) != 2:
        raise ValueError(
            "worker runtime info must be a two-item tuple ordered worker0/worker1"
        )
    for expected_index, worker in enumerate(info):
        if (
            not isinstance(worker, WorkerReady)
            or worker.worker_index != expected_index
            or type(worker.pid) is not int
            or worker.pid <= 0
            or not isinstance(worker.visible_device, str)
            or not worker.visible_device
            or not isinstance(worker.cache_root, str)
            or not worker.cache_root
        ):
            raise ValueError(
                "worker runtime info must contain valid WorkerReady entries "
                "ordered worker0/worker1"
            )
    if info[0].pid == info[1].pid:
        raise ValueError("worker runtime info must contain distinct child PIDs")
    return info


def _validated_worker_latencies(engine: Any) -> tuple[float, float]:
    latencies = engine.last_worker_latencies
    if not isinstance(latencies, tuple) or len(latencies) != 2:
        raise ValueError(
            "worker latencies must be a two-item tuple ordered worker0/worker1"
        )
    normalized: list[float] = []
    for latency in latencies:
        if (
            type(latency) not in (int, float)
            or not math.isfinite(float(latency))
            or latency < 0
        ):
            raise ValueError(
                "worker latencies must be finite nonnegative numbers"
            )
        normalized.append(float(latency))
    return normalized[0], normalized[1]


def _validated_worker_counts(
    values: Any,
    *,
    label: str,
) -> tuple[int, int]:
    if (
        not isinstance(values, tuple)
        or len(values) != 2
        or any(type(value) is not int or value < 0 for value in values)
    ):
        raise ValueError(
            f"{label} must be a two-item tuple of nonnegative exact integers "
            "ordered worker0/worker1"
        )
    return values


def _validated_worker_batch_metadata(
    engine: Any,
    expected_shard_sizes: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int], tuple[float, float]]:
    result_counts = _validated_worker_counts(
        engine.last_worker_result_counts,
        label="worker result counts",
    )
    if result_counts != expected_shard_sizes:
        raise ValueError(
            f"worker result counts {result_counts} do not match expected shard "
            f"sizes {expected_shard_sizes}"
        )
    nonempty_counts = _validated_worker_counts(
        engine.last_worker_nonempty_counts,
        label="worker nonempty counts",
    )
    if any(
        nonempty > result
        for nonempty, result in zip(nonempty_counts, result_counts, strict=True)
    ):
        raise ValueError(
            "worker nonempty counts cannot exceed worker result counts"
        )
    return result_counts, nonempty_counts, _validated_worker_latencies(engine)


def _execute_locked(
    *,
    config: TeacherGenerationConfig,
    adopt_legacy_state: bool,
    engine_factory: Callable[..., Any],
    state_store_factory: Callable[[Path], Any],
    now: Callable[[], datetime | str],
    resources: dict[str, Any],
) -> int:
    source_bytes, source_stat, source_hash = _read_source_snapshot(
        config.input_path
    )
    source_rows = _parse_jsonl_bytes(source_bytes, config.input_path)
    validate_source_rows(source_rows)
    LOGGER.info("source rows=%s", len(source_rows))

    store = state_store_factory(config.output_dir)
    _verify_source_snapshot(config.input_path, source_stat, source_hash)
    state = store.load_resume_state(
        source_rows,
        config,
        adopt_legacy_state=adopt_legacy_state,
    )
    _verify_source_snapshot(config.input_path, source_stat, source_hash)
    accepted = list(state.accepted)
    rejected = list(state.rejected)
    LOGGER.info(
        "resume processed=%s accepted=%s rejected=%s",
        state.processed_count,
        len(accepted),
        len(rejected),
    )
    contract = build_generation_contract(config, source_sha256=source_hash)
    clock = _IncreasingUtcClock(now, after=state.created_at)

    if not store.has_v2_manifest():
        initial_manifest = _manifest(
            config=config,
            accepted=accepted,
            rejected=rejected,
            source_sha256=source_hash,
            generation_contract=contract,
            created_at=state.created_at,
            updated_at=clock.next(),
            accepted_sha256=_snapshot_sha256(
                store, "accepted_path", accepted
            ),
            rejected_sha256=_snapshot_sha256(
                store, "rejected_path", rejected
            ),
        )
        store.commit(
            batch_id=0,
            submitted_start=state.processed_count,
            submitted_stop=state.processed_count,
            accepted=accepted,
            rejected=rejected,
            manifest=initial_manifest,
        )
        LOGGER.info(
            "materialized V2 manifest processed=%s accepted=%s rejected=%s",
            state.processed_count,
            len(accepted),
            len(rejected),
        )

    if len(accepted) >= config.stop_after_accepted:
        LOGGER.info("teacher target already complete: %s", len(accepted))
        return 0
    if state.processed_count >= len(source_rows):
        LOGGER.error(
            "source exhausted: accepted=%s target=%s",
            len(accepted),
            config.stop_after_accepted,
        )
        return 2

    engine = _build_engine(config, engine_factory)
    resources["engine"] = engine
    LOGGER.info("engine starting")
    engine.start()
    runtime_info = _validated_worker_runtime_info(engine)
    resources["runtime_pids"] = tuple(worker.pid for worker in runtime_info)
    LOGGER.info("engine ready")
    for worker in runtime_info:
        LOGGER.info(
            "worker%s runtime pid=%s visible_device=%s cache_root=%s",
            worker.worker_index,
            worker.pid,
            worker.visible_device,
            worker.cache_root,
        )
    processed = state.processed_count
    batch_id = processed // config.batch_size + 1
    while processed < len(source_rows) and len(accepted) < config.stop_after_accepted:
        submitted_start = processed
        batch_rows = source_rows[
            submitted_start : submitted_start + config.batch_size
        ]
        prompts = tuple(
            PositionedPrompt(
                submitted_start + offset,
                row["prompt"],
                seed=derive_request_seed(
                    config.seed,
                    "teacher",
                    str(row["id"]),
                    0,
                ),
            )
            for offset, row in enumerate(batch_rows)
        )
        LOGGER.info(
            "batch=%s global_range=[%s,%s) count=%s",
            batch_id,
            submitted_start,
            submitted_start + len(prompts),
            len(prompts),
        )
        shards = split_contiguous(prompts)
        responses = engine.generate(batch_id, prompts)
        expected_shard_sizes = (len(shards[0]), len(shards[1]))
        result_counts, nonempty_counts, worker_latencies = (
            _validated_worker_batch_metadata(engine, expected_shard_sizes)
        )
        LOGGER.info(
            "batch=%s worker0_shard=%s worker0_results=%s "
            "worker0_nonempty=%s worker0_latency_seconds=%.3f "
            "worker1_shard=%s worker1_results=%s worker1_nonempty=%s "
            "worker1_latency_seconds=%.3f",
            batch_id,
            expected_shard_sizes[0],
            result_counts[0],
            nonempty_counts[0],
            worker_latencies[0],
            expected_shard_sizes[1],
            result_counts[1],
            nonempty_counts[1],
            worker_latencies[1],
        )
        expected_positions = [item.position for item in prompts]
        if len(responses) != len(expected_positions):
            raise ValueError(
                "response count mismatch: "
                f"received {len(responses)}, expected {len(expected_positions)}"
            )
        actual_positions = []
        for item in responses:
            if (
                not isinstance(item, tuple)
                or len(item) != 2
                or type(item[0]) is not int
                or not isinstance(item[1], str)
            ):
                raise ValueError(f"malformed positioned response: {item!r}")
            actual_positions.append(item[0])
        if actual_positions != expected_positions:
            raise ValueError(
                f"response positions {actual_positions} do not match "
                f"expected {expected_positions}"
            )

        new_accepted = list(accepted)
        new_rejected = list(rejected)
        consumed = 0
        for row, (_, response) in zip(batch_rows, responses, strict=True):
            if len(new_accepted) >= config.stop_after_accepted:
                break
            payload = build_teacher_payload(row, response)
            if payload["validation"]["ok"]:
                new_accepted.append(payload)
            else:
                new_rejected.append(payload)
            consumed += 1

        new_processed = submitted_start + consumed
        manifest = _manifest(
            config=config,
            accepted=new_accepted,
            rejected=new_rejected,
            source_sha256=source_hash,
            generation_contract=contract,
            created_at=state.created_at,
            updated_at=clock.next(),
        )
        store.commit(
            batch_id=batch_id,
            submitted_start=submitted_start,
            submitted_stop=new_processed,
            accepted=new_accepted,
            rejected=new_rejected,
            manifest=manifest,
        )
        accepted = new_accepted
        rejected = new_rejected
        processed = new_processed
        LOGGER.info(
            "committed batch=%s processed=%s accepted=%s rejected=%s",
            batch_id,
            processed,
            len(accepted),
            len(rejected),
        )
        batch_id += 1

    if len(accepted) == config.stop_after_accepted:
        return 0
    LOGGER.error(
        "source exhausted: accepted=%s target=%s",
        len(accepted),
        config.stop_after_accepted,
    )
    return 2


def _cleanup_resources(
    engine: Any | None,
    lock: Any,
    runtime_pids: tuple[int, int] | None,
) -> list[BaseException]:
    errors: list[BaseException] = []
    if engine is not None:
        try:
            engine.close()
        except BaseException as exc:
            errors.append(exc)
        else:
            LOGGER.info("workers closed")
        try:
            LOGGER.info(
                "worker shutdown exitcodes=%s runtime_pids=%s",
                engine.worker_exitcodes,
                runtime_pids,
            )
        except BaseException as exc:
            errors.append(exc)
    try:
        lock.release()
    except BaseException as exc:
        errors.append(exc)
    return errors


def _cleanup_failure(errors: list[BaseException]) -> BaseException:
    if len(errors) == 1:
        return errors[0]
    if all(isinstance(error, Exception) for error in errors):
        return ExceptionGroup(
            "teacher coordinator cleanup failures",
            [error for error in errors if isinstance(error, Exception)],
        )
    return BaseExceptionGroup("teacher coordinator cleanup failures", errors)


def run(
    config_path: str | Path = PRODUCTION_CONFIG,
    *,
    recover_stale_lock: bool = False,
    adopt_legacy_state: bool = False,
    engine_factory: Callable[..., Any] = ParallelVLLMEngine,
    state_store_factory: Callable[[Path], Any] = TeacherStateStore,
    lock_factory: Callable[..., Any] = OutputLock,
    now: Callable[[], datetime | str] = _default_now,
    cuda_visible_devices: str | None = None,
) -> int:
    resolved_config_path = _resolve_path(config_path, REPO_ROOT)
    config = load_teacher_config(resolved_config_path)
    validate_cuda_visibility(config, cuda_visible_devices)

    lock = lock_factory(
        path=config.output_dir / ".teacher_pool.lock",
        config_path=resolved_config_path,
        output_dir=config.output_dir,
        topology=config.topology,
    )
    if recover_stale_lock:
        LOGGER.info("stale lock recovery requested")
    LOGGER.info("acquiring output lock: %s", lock.path)
    lock.acquire(recover_stale=recover_stale_lock)
    if lock.recovered_stale:
        LOGGER.info("stale output lock recovered and acquired: %s", lock.path)
    else:
        LOGGER.info("output lock acquired normally: %s", lock.path)
    resources: dict[str, Any] = {"engine": None, "runtime_pids": None}
    primary: BaseException | None = None
    result: int | None = None
    try:
        result = _execute_locked(
            config=config,
            adopt_legacy_state=adopt_legacy_state,
            engine_factory=engine_factory,
            state_store_factory=state_store_factory,
            now=now,
            resources=resources,
        )
    except KeyboardInterrupt as exc:
        primary = exc
        result = 130
        LOGGER.error("teacher generation interrupted")
    except BaseException as exc:
        primary = exc

    cleanup_errors = _cleanup_resources(
        resources["engine"],
        lock,
        resources["runtime_pids"],
    )
    if isinstance(primary, KeyboardInterrupt):
        for cleanup_error in cleanup_errors:
            primary.add_note(f"cleanup failed: {cleanup_error}")
            LOGGER.error("cleanup failed after interrupt: %s", cleanup_error)
        return 130
    if primary is not None:
        if cleanup_errors:
            cleanup_failure = _cleanup_failure(cleanup_errors)
            primary.add_note(f"cleanup failed: {cleanup_failure}")
            raise primary.with_traceback(primary.__traceback__) from cleanup_failure
        raise primary.with_traceback(primary.__traceback__)
    if cleanup_errors:
        raise _cleanup_failure(cleanup_errors)
    assert result is not None
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        return run(
            args.config,
            recover_stale_lock=args.recover_stale_lock,
            adopt_legacy_state=args.adopt_legacy_state,
        )
    except Exception:
        LOGGER.exception("teacher generation failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
