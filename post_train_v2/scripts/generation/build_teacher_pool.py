from __future__ import annotations

import argparse
import hashlib
import json
import logging
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

from post_train.src.countdown.config import load_yaml_config, resolve_path
from post_train.src.countdown.io import read_jsonl
from post_train.src.countdown.output_lock import OutputLock
from post_train.src.countdown.validation import (
    extract_answer_text,
    validate_countdown_response,
)
from post_train_v2.src.generation.parallel_vllm import (
    ParallelVLLMEngine,
    PositionedPrompt,
    WorkerSpec,
)
from post_train_v2.src.generation.teacher_state import (
    TeacherGenerationConfig,
    TeacherStateStore,
    build_generation_contract,
    build_manifest,
    sha256_file,
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
    path = resolve_path(config_path, repo_root).resolve()
    raw = load_yaml_config(path)
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
        model_path=resolve_path(raw["model_path"], repo_root),
        input_path=resolve_path(raw["input_path"], repo_root),
        output_dir=resolve_path(raw["output_dir"], repo_root),
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


def build_teacher_payload(
    row: dict[str, Any],
    response: str,
) -> dict[str, Any]:
    text = response.strip()
    result = validate_countdown_response(text, row["numbers"], row["target"])
    return {
        **row,
        "response": text,
        "teacher_expr": extract_answer_text(text),
        "validation": {
            "ok": result.ok,
            "error": result.error,
            "value": result.value,
        },
    }


def _jsonl_sha256(rows: list[dict[str, Any]]) -> str:
    payload = b"".join(
        (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
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


def _store_has_committed_state(store: Any) -> bool:
    explicit = getattr(store, "state_exists", None)
    if explicit is not None:
        return bool(explicit)
    paths = (
        getattr(store, "accepted_path", None),
        getattr(store, "rejected_path", None),
        getattr(store, "manifest_path", None),
    )
    return any(path is not None and Path(path).exists() for path in paths)


def _manifest(
    *,
    config: TeacherGenerationConfig,
    accepted: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    source_sha256: str,
    generation_contract: dict[str, Any],
    created_at: str,
    updated_at: str,
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
        accepted_sha256=_jsonl_sha256(accepted),
        rejected_sha256=_jsonl_sha256(rejected),
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
    LOGGER.info(
        "workers: gpu0 device=%s cache=%s; gpu1 device=%s cache=%s",
        specs[0].device,
        specs[0].cache_root,
        specs[1].device,
        specs[1].cache_root,
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
    resolved_config_path = resolve_path(config_path, REPO_ROOT).resolve()
    config = load_teacher_config(resolved_config_path)
    validate_cuda_visibility(config, cuda_visible_devices)
    source_rows = read_jsonl(config.input_path)
    validate_source_rows(source_rows)

    lock = lock_factory(
        path=config.output_dir / ".teacher_pool.lock",
        config_path=resolved_config_path,
        output_dir=config.output_dir,
        topology=config.topology,
    )
    lock.acquire(recover_stale=recover_stale_lock)
    engine = None
    try:
        store = state_store_factory(config.output_dir)
        state = store.load_resume_state(
            source_rows,
            config,
            adopt_legacy_state=adopt_legacy_state,
        )
        accepted = list(state.accepted)
        rejected = list(state.rejected)
        source_hash = sha256_file(config.input_path)
        contract = build_generation_contract(config, source_sha256=source_hash)
        clock = _IncreasingUtcClock(now, after=state.created_at)

        if state.processed_count == 0 and not _store_has_committed_state(store):
            initial_manifest = _manifest(
                config=config,
                accepted=accepted,
                rejected=rejected,
                source_sha256=source_hash,
                generation_contract=contract,
                created_at=state.created_at,
                updated_at=clock.next(),
            )
            store.commit(
                batch_id=0,
                submitted_start=0,
                submitted_stop=0,
                accepted=accepted,
                rejected=rejected,
                manifest=initial_manifest,
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
        engine.start()
        processed = state.processed_count
        batch_id = processed // config.batch_size + 1
        while processed < len(source_rows) and len(accepted) < config.stop_after_accepted:
            submitted_start = processed
            batch_rows = source_rows[
                submitted_start : submitted_start + config.batch_size
            ]
            prompts = tuple(
                PositionedPrompt(submitted_start + offset, row["prompt"])
                for offset, row in enumerate(batch_rows)
            )
            LOGGER.info(
                "batch=%s range=[%s,%s) count=%s",
                batch_id,
                submitted_start,
                submitted_start + len(prompts),
                len(prompts),
            )
            responses = engine.generate(batch_id, prompts)
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
    except KeyboardInterrupt:
        LOGGER.error("teacher generation interrupted")
        return 130
    finally:
        try:
            if engine is not None:
                try:
                    engine.close()
                finally:
                    LOGGER.info("workers closed")
        finally:
            lock.release()


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
