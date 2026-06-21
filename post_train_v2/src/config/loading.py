from pathlib import Path
from typing import Any, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]


def load_yaml(path: str | Path) -> dict[str, Any]:
    resolved = resolve_repo_path(path)
    value = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML config must be a mapping: {resolved}")
    return value


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return (
        candidate.resolve()
        if candidate.is_absolute()
        else (REPO_ROOT / candidate).resolve()
    )


def require_keys(mapping: Mapping[str, Any], *keys: str) -> None:
    missing = sorted(key for key in keys if key not in mapping)
    if missing:
        raise ValueError(f"missing keys: {', '.join(missing)}")
