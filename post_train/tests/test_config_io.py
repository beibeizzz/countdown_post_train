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


def test_training_configs_include_disabled_wandb_defaults():
    config_paths = [
        Path("post_train/configs/sft_full.yaml"),
        Path("post_train/configs/sft_lora.yaml"),
        Path("post_train/configs/dpo_train.yaml"),
        Path("post_train/configs/grpo.yaml"),
    ]

    for path in config_paths:
        cfg = load_yaml_config(path)
        assert "report_to" in cfg
        assert cfg["report_to"] is None
        assert cfg["wandb_project"] == "countdown-post-train"
        assert cfg["wandb_entity"] is None
        assert cfg["wandb_group"] is None
        assert isinstance(cfg["wandb_tags"], list)
        assert isinstance(cfg["run_name"], str)
        assert cfg["run_name_auto_suffix"] is True
        assert int(cfg["logging_steps"]) > 0

    rft_cfg = load_yaml_config("post_train/configs/rft.yaml")
    train_cfg = rft_cfg["train"]
    assert train_cfg["report_to"] is None
    assert train_cfg["wandb_project"] == "countdown-post-train"
    assert train_cfg["wandb_entity"] is None
    assert train_cfg["wandb_group"] is None
    assert isinstance(train_cfg["wandb_tags"], list)
    assert train_cfg["run_name"] == "rft"
    assert train_cfg["run_name_auto_suffix"] is True
    assert int(train_cfg["logging_steps"]) > 0
