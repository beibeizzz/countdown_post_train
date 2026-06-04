from pathlib import Path

from post_train.src.countdown.config import load_yaml_config, resolve_path
from post_train.src.countdown.io import read_jsonl, write_jsonl, write_manifest


def test_load_yaml_config_reads_values(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text("seed: 42\nname: countdown\n", encoding="utf-8")

    cfg = load_yaml_config(path)

    assert cfg["seed"] == 42
    assert cfg["name"] == "countdown"


def test_resolve_path_keeps_absolute(tmp_path: Path):
    absolute = tmp_path / "file.jsonl"

    assert resolve_path(absolute, base_dir=Path("base")) == absolute


def test_resolve_path_joins_relative():
    assert resolve_path("data/file.jsonl", base_dir=Path("/repo")) == Path("/repo/data/file.jsonl")


def test_jsonl_round_trip(tmp_path: Path):
    rows = [{"id": "a", "value": 1}, {"id": "b", "value": 2}]
    path = tmp_path / "rows.jsonl"

    write_jsonl(path, rows)

    assert read_jsonl(path) == rows


def test_write_manifest_adds_standard_envelope(tmp_path: Path):
    path = tmp_path / "manifest.json"

    write_manifest(path, {"name": "stage_name", "num_records": 3})

    manifest = load_yaml_config(path)
    assert manifest["manifest_version"] == 1
    assert manifest["schema"] == "countdown.post_train.manifest.v1"
    assert manifest["name"] == "stage_name"
    assert manifest["stage"] == "stage_name"
    assert manifest["created_at"]
    assert manifest["num_records"] == 3
