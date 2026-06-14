from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from post_train.src.countdown.config import load_yaml_config, resolve_path
from post_train.src.countdown.generation import GenerationConfig, VLLMGenerator
from post_train.src.countdown.io import read_jsonl, write_jsonl, write_manifest
from post_train.src.countdown.output_lock import OutputLock
from post_train.src.countdown.validation import extract_answer_text, validate_countdown_response


DEFAULT_CONFIG = "post_train/configs/teacher_rollout.yaml"
DEFAULT_INPUT = "post_train/data/processed/train_pool.jsonl"
OUTPUT_DIR = "post_train/data/teacher_rollouts"
ACCEPTED_FILENAME = "teacher_accepted_20k.jsonl"
REJECTED_FILENAME = "teacher_rejected.jsonl"
MANIFEST_FILENAME = "manifest.json"
TRANSACTION_FILENAME = ".teacher_pool.transaction.json"
V2_STAGE = "teacher_accepted_pool"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build teacher accepted pool.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--recover-stale-lock", action="store_true")
    return parser.parse_args(argv)


def collect_processed_ids(
    accepted: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
) -> set[Any]:
    validate_unique_ids("teacher rollout outputs", accepted + rejected)
    return {row["id"] for row in accepted + rejected}


def validate_unique_ids(name: str, rows: list[dict[str, Any]]) -> None:
    seen: set[Any] = set()
    for row in rows:
        row_id = row["id"]
        if row_id in seen:
            raise ValueError(f"{name} contains duplicate id: {row_id}")
        seen.add(row_id)


def validate_resume_state(
    accepted: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    target: int,
) -> None:
    validate_unique_ids("teacher rollout outputs", accepted + rejected)
    if len(accepted) > target:
        raise ValueError(
            f"existing accepted rows count {len(accepted)} exceeds target {target}"
        )


def validate_source_rows(source_rows: list[dict[str, Any]]) -> None:
    validate_unique_ids("source input", source_rows)


def build_teacher_payload(row: dict[str, Any], response: str) -> dict[str, Any]:
    text = response.strip()
    result = validate_countdown_response(text, row["numbers"], int(row["target"]))
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


def process_teacher_responses(
    *,
    rows: list[dict[str, Any]],
    responses: list[str],
    accepted: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    processed_ids: set[Any],
    target: int,
) -> None:
    for row, response in zip(rows, responses, strict=True):
        if len(accepted) >= target:
            break

        payload = build_teacher_payload(row, response)
        if payload["validation"]["ok"]:
            accepted.append(payload)
        else:
            rejected.append(payload)
        processed_ids.add(row["id"])


def batched(rows: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    return [rows[index : index + batch_size] for index in range(0, len(rows), batch_size)]


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temp_path = path.with_name(f"{path.name}.tmp")
    write_jsonl(temp_path, rows)
    temp_path.replace(path)


def atomic_write_manifest(path: Path, payload: dict[str, Any]) -> None:
    temp_path = path.with_name(f"{path.name}.tmp")
    write_manifest(temp_path, payload)
    temp_path.replace(path)


def read_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"{path}: manifest must be an object")
    return manifest


def reject_v2_owned_state(output_dir: Path) -> None:
    manifest_path = output_dir / MANIFEST_FILENAME
    if manifest_path.exists():
        manifest = read_manifest(manifest_path)
        has_v2_fingerprint = "generation_contract_fingerprint" in manifest
        has_v2_contract = "generation_contract" in manifest
        has_v2_stage_schema = (
            manifest.get("stage") == V2_STAGE
            and "schema_version" in manifest
        )
        if has_v2_fingerprint or has_v2_contract or has_v2_stage_schema:
            raise RuntimeError(
                "Output directory contains V2 teacher state; archive or remove "
                "the V2 state before using the legacy teacher generator."
            )

    transaction_path = output_dir / TRANSACTION_FILENAME
    if transaction_path.exists():
        raise RuntimeError(
            "Output directory contains a V2 transaction journal; archive or remove "
            "the transaction state before using the legacy teacher generator."
        )


def _execute_locked(
    *,
    cfg: dict[str, Any],
    input_path: Path,
    model_path: Path,
    output_dir: Path,
    generator_factory: Callable[[str], Any],
) -> None:
    reject_v2_owned_state(output_dir)

    accepted_path = output_dir / ACCEPTED_FILENAME
    rejected_path = output_dir / REJECTED_FILENAME

    source_rows = read_jsonl(input_path)
    validate_source_rows(source_rows)
    accepted = read_jsonl(accepted_path) if accepted_path.exists() else []
    rejected = read_jsonl(rejected_path) if rejected_path.exists() else []

    target = int(cfg["stop_after_accepted"])
    validate_resume_state(accepted, rejected, target)
    processed_ids = collect_processed_ids(accepted, rejected)
    generation_config = GenerationConfig(
        max_new_tokens=int(cfg["max_new_tokens"]),
        temperature=float(cfg["temperature"]),
        top_p=float(cfg["top_p"]),
        enable_thinking=bool(cfg.get("enable_thinking", False)),
    )

    remaining_rows = [
        row
        for row in source_rows
        if row["id"] not in processed_ids
    ]

    if len(accepted) < target and remaining_rows:
        generator = generator_factory(str(model_path))
        for batch in batched(remaining_rows, int(cfg["batch_size"])):
            if len(accepted) >= target:
                break

            prompts = [row["prompt"] for row in batch]
            responses = generator.generate(prompts, generation_config)
            process_teacher_responses(
                rows=batch,
                responses=responses,
                accepted=accepted,
                rejected=rejected,
                processed_ids=processed_ids,
                target=target,
            )

            atomic_write_jsonl(accepted_path, accepted[:target])
            atomic_write_jsonl(rejected_path, rejected)

    atomic_write_jsonl(accepted_path, accepted[:target])
    atomic_write_jsonl(rejected_path, rejected)
    atomic_write_manifest(
        output_dir / MANIFEST_FILENAME,
        {
            "name": "teacher_accepted_pool",
            "model": str(model_path),
            "num_accepted": len(accepted[:target]),
            "num_rejected": len(rejected),
            "max_new_tokens": generation_config.max_new_tokens,
            "enable_thinking": generation_config.enable_thinking,
        },
    )


def run(
    config_path: str | Path = DEFAULT_CONFIG,
    input_path: str | Path = DEFAULT_INPUT,
    *,
    recover_stale_lock: bool = False,
    lock_factory: Callable[..., Any] = OutputLock,
    generator_factory: Callable[[str], Any] = VLLMGenerator,
) -> None:
    cfg_path = resolve_path(config_path, REPO_ROOT)
    cfg = load_yaml_config(cfg_path)
    resolved_input_path = resolve_path(input_path, REPO_ROOT)
    model_path = resolve_path(cfg["model_path"], REPO_ROOT)
    output_dir = resolve_path(OUTPUT_DIR, REPO_ROOT)

    lock = lock_factory(
        path=output_dir / ".teacher_pool.lock",
        config_path=cfg_path,
        output_dir=output_dir,
        topology="legacy_single_tp1",
    )
    lock.acquire(recover_stale=recover_stale_lock)

    primary_error: BaseException | None = None
    try:
        _execute_locked(
            cfg=cfg,
            input_path=resolved_input_path,
            model_path=model_path,
            output_dir=output_dir,
            generator_factory=generator_factory,
        )
    except BaseException as exc:
        primary_error = exc
    finally:
        try:
            lock.release()
        except BaseException as release_error:
            if primary_error is not None:
                primary_error.add_note(f"lock release failed: {release_error}")
                raise primary_error.with_traceback(
                    primary_error.__traceback__
                ) from release_error
            raise

    if primary_error is not None:
        raise primary_error.with_traceback(primary_error.__traceback__)


def main() -> None:
    args = parse_args()
    run(
        args.config,
        args.input,
        recover_stale_lock=args.recover_stale_lock,
    )


if __name__ == "__main__":
    main()
