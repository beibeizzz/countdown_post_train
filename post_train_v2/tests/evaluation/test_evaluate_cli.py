from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from post_train_v2.src.artifacts.atomic import publish_jsonl
from post_train_v2.src.artifacts.hashing import sha256_file
from post_train_v2.src.artifacts.manifest import (
    ArtifactFile,
    ManifestV2,
    load_manifest,
    publish_manifest,
)
from post_train_v2.src.countdown.bucketing import assign_bucket
from post_train_v2.src.countdown.prompts import build_solution_prompt
from post_train_v2.src.evaluation import generation
from post_train_v2.src.evaluation.cli import (
    fingerprint_evaluation_model,
    run_evaluation,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "post_train_v2/scripts/eval/evaluate_model.py"
SOURCE_SCHEMA = {
    "id": "string",
    "source_index": "integer",
    "numbers": "array[integer]",
    "target": "integer",
    "gold_expr": "string",
    "prompt": "string",
    "bucket": "object",
}


def row(index: int) -> dict:
    numbers = [1, 2, index + 3]
    target = sum(numbers)
    expression = f"1+2+{index + 3}"
    return {
        "id": f"train-{index:06d}",
        "source_index": index,
        "numbers": numbers,
        "target": target,
        "gold_expr": expression,
        "prompt": build_solution_prompt(numbers, target),
        "bucket": assign_bucket(numbers, expression),
    }


def write_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    model_dir = tmp_path / "model"
    input_dir.mkdir()
    model_dir.mkdir()
    eval_path = input_dir / "eval_50.jsonl"
    rows = [row(1), row(2)]
    publish_jsonl(eval_path, rows)
    validation_manifest = ManifestV2.build(
        artifact_type="dataset",
        stage="build_validation_splits",
        files=[
            ArtifactFile(
                eval_path.name,
                sha256_file(eval_path),
                eval_path.stat().st_size,
                len(rows),
                SOURCE_SCHEMA,
            )
        ],
        parents=[],
        config={"seed": 42},
        stage_metadata={"completed": True},
        git_revision="fixture",
    )
    manifest_path = input_dir / "validation_manifest.json"
    publish_manifest(manifest_path, validation_manifest)
    (model_dir / "config.json").write_text('{"model_type":"qwen3"}', encoding="utf-8")
    config = {
        "eval_data": str(eval_path),
        "eval_manifest": str(manifest_path),
        "output_dir": str(output_dir),
        "max_new_tokens": 256,
        "enable_thinking": False,
        "do_sample": False,
    }
    config_path = tmp_path / "eval.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path, model_dir, output_dir


def test_generation_contract_disables_thinking_and_sampling():
    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 2

        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{"role": "user", "content": "solve"}]
            assert kwargs == {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": False,
            }
            return "rendered"

    assert generation.render_prompt(Tokenizer(), "solve") == "rendered"
    assert generation.generation_kwargs(256, pad_token_id=0) == {
        "do_sample": False,
        "max_new_tokens": 256,
        "pad_token_id": 0,
    }
    with pytest.raises(ValueError, match="256"):
        generation.generation_kwargs(257, pad_token_id=0)


def test_run_evaluation_writes_samples_metrics_and_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config_path, model_dir, output_dir = write_fixture(tmp_path)

    monkeypatch.setattr(
        "post_train_v2.src.evaluation.cli.load_model_and_tokenizer",
        lambda *args, **kwargs: (object(), object()),
    )

    def fake_evaluate(rows, tokenizer, model, *, max_new_tokens):
        assert max_new_tokens == 256
        return [
            {
                "id": item["id"],
                "prompt": item["prompt"],
                "raw_generation": f"<answer>{item['gold_expr']}</answer>",
                "extracted_expr": item["gold_expr"],
                "format_ok": True,
                "valid_expression": True,
                "correct": True,
                "error": None,
                "value": f"{item['target']}/1",
                "generated_tokens": 8,
                "truncated": False,
            }
            for item in rows
        ]

    monkeypatch.setattr(
        "post_train_v2.src.evaluation.cli.evaluate_rows",
        fake_evaluate,
    )

    manifest = run_evaluation(config_path, model_dir)

    samples = [
        json.loads(line)
        for line in (output_dir / "samples.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert len(samples) == 2
    assert metrics["accuracy"] == 1.0
    assert manifest == load_manifest(output_dir / "manifest.json")
    assert manifest.stage == "evaluate_model"
    assert manifest.model_path == str(model_dir.resolve())
    assert len(manifest.model_fingerprint) == 64
    assert [item.relative_path for item in manifest.files] == [
        "samples.jsonl",
        "metrics.json",
    ]


def test_adapter_fingerprint_includes_base_model(tmp_path: Path):
    adapter = tmp_path / "adapter"
    base = tmp_path / "base"
    adapter.mkdir()
    base.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": str(base)}),
        encoding="utf-8",
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    (base / "model.safetensors").write_bytes(b"base-one")

    first = fingerprint_evaluation_model(adapter)
    (base / "model.safetensors").write_bytes(b"base-two")
    second = fingerprint_evaluation_model(adapter)

    assert first != second


def test_model_change_during_evaluation_prevents_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config_path, model_dir, output_dir = write_fixture(tmp_path)
    model_file = model_dir / "config.json"
    monkeypatch.setattr(
        "post_train_v2.src.evaluation.cli.load_model_and_tokenizer",
        lambda *args, **kwargs: (object(), object()),
    )

    def mutate_model(rows, tokenizer, model, *, max_new_tokens):
        model_file.write_text('{"model_type":"changed"}', encoding="utf-8")
        return []

    monkeypatch.setattr(
        "post_train_v2.src.evaluation.cli.evaluate_rows",
        mutate_model,
    )

    with pytest.raises(ValueError, match="model changed"):
        run_evaluation(config_path, model_dir)

    assert not (output_dir / "manifest.json").exists()


def test_evaluation_output_lock_prevents_concurrent_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config_path, model_dir, output_dir = write_fixture(tmp_path)
    output_dir.mkdir()
    lock_path = output_dir / ".evaluate_model.lock"
    lock_path.write_text('{"owner_token":"other"}\n', encoding="utf-8")
    loaded = False

    def fail_if_loaded(*args, **kwargs):
        nonlocal loaded
        loaded = True
        raise AssertionError("model must not load while output is locked")

    monkeypatch.setattr(
        "post_train_v2.src.evaluation.cli.load_model_and_tokenizer",
        fail_if_loaded,
    )

    with pytest.raises(RuntimeError, match="output lock"):
        run_evaluation(config_path, model_dir)

    assert loaded is False
    assert lock_path.exists()


def test_evaluation_releases_output_lock_after_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config_path, model_dir, output_dir = write_fixture(tmp_path)
    monkeypatch.setattr(
        "post_train_v2.src.evaluation.cli.load_model_and_tokenizer",
        lambda *args, **kwargs: (object(), object()),
    )
    monkeypatch.setattr(
        "post_train_v2.src.evaluation.cli.evaluate_rows",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("generation failed")
        ),
    )

    with pytest.raises(RuntimeError, match="generation failed"):
        run_evaluation(config_path, model_dir)

    assert not (output_dir / ".evaluate_model.lock").exists()


def test_cli_help_from_arbitrary_cwd(tmp_path: Path):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--model-path" in result.stdout
    assert "--base-model-path" in result.stdout
