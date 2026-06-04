import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    jsonl_path = Path(path)
    rows: list[dict[str, Any]] = []

    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{jsonl_path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{jsonl_path}:{line_number}: JSONL row must be an object")
            rows.append(row)

    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    jsonl_path = Path(path)
    ensure_parent(jsonl_path)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def write_json(path: str | Path, payload: Any) -> None:
    json_path = Path(path)
    ensure_parent(json_path)
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_manifest(path: str | Path, payload: dict[str, Any]) -> None:
    created_at = payload.get("created_at", datetime.now(timezone.utc).isoformat())
    manifest = {
        "manifest_version": 1,
        "schema": "countdown.post_train.manifest.v1",
        "name": payload.get("name", Path(path).parent.name),
        "stage": payload.get("stage", payload.get("name", Path(path).parent.name)),
        "created_at": created_at,
    }
    manifest.update(payload)
    write_json(path, manifest)
