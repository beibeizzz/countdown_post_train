from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from post_train_v2.src.training.model_selection import (
    EvaluationResult,
    export_best_checkpoint,
    export_final_model,
    publish_model_export,
    select_best,
)


def result(step: int, *, accuracy: float, format_rate: float) -> EvaluationResult:
    return EvaluationResult(
        step=step,
        metrics={"accuracy": accuracy, "format_rate": format_rate},
    )


@dataclass
class FakePublisher:
    files: tuple[str, ...]

    def save_pretrained(self, path):
        for filename in self.files:
            (path / filename).write_text(filename, encoding="utf-8")


def test_select_best_orders_by_accuracy_format_then_earliest_step():
    assert select_best(
        [
            result(100, accuracy=0.5, format_rate=0.9),
            result(200, accuracy=0.5, format_rate=0.95),
            result(300, accuracy=0.5, format_rate=0.95),
        ]
    ).step == 200


def test_publish_full_model_export_requires_direct_load_before_manifest(tmp_path):
    checked = []

    manifest = publish_model_export(
        model=FakePublisher(("config.json",)),
        tokenizer=FakePublisher(("tokenizer.json",)),
        output_dir=tmp_path,
        export_name="best",
        export_kind="full_model",
        direct_load_check=lambda path: checked.append(path),
    )

    export_dir = tmp_path / "best"
    assert checked == [export_dir]
    assert manifest["export_kind"] == "full_model"
    assert manifest["direct_loadable"] is True
    assert json.loads((export_dir / "export_manifest.json").read_text()) == manifest


def test_publish_full_model_export_does_not_publish_manifest_when_check_fails(tmp_path):
    with pytest.raises(RuntimeError, match="cannot load"):
        publish_model_export(
            model=FakePublisher(("config.json",)),
            tokenizer=FakePublisher(("tokenizer.json",)),
            output_dir=tmp_path,
            export_name="best",
            export_kind="full_model",
            direct_load_check=lambda path: (_ for _ in ()).throw(
                RuntimeError("cannot load")
            ),
        )

    assert not (tmp_path / "best" / "export_manifest.json").exists()


def test_publish_lora_best_is_adapter_not_direct_loadable(tmp_path):
    manifest = publish_model_export(
        model=FakePublisher(("adapter_config.json",)),
        tokenizer=FakePublisher(("tokenizer.json",)),
        output_dir=tmp_path,
        export_name="best",
        export_kind="lora_adapter",
    )

    assert manifest["export_kind"] == "lora_adapter"
    assert manifest["direct_loadable"] is False


def test_export_final_model_uses_trainer_save_model_then_tokenizer(tmp_path):
    events = []
    trainer = type(
        "Trainer",
        (),
        {"save_model": lambda self, path: events.append(("model", path))},
    )()
    tokenizer = FakePublisher(("tokenizer.json",))

    manifest = export_final_model(
        trainer=trainer,
        tokenizer=tokenizer,
        output_dir=tmp_path,
        export_kind="full_model",
        direct_load_check=lambda path: events.append(("check", path)),
    )

    assert events == [
        ("model", tmp_path / "final"),
        ("check", tmp_path / "final"),
    ]
    assert manifest["export_name"] == "final"
    assert (tmp_path / "final" / "tokenizer.json").is_file()


def test_export_best_checkpoint_copies_selected_existing_checkpoint(tmp_path):
    ledger = tmp_path / "eval" / "ledger.jsonl"
    ledger.parent.mkdir()
    ledger.write_text(
        '{"step": 100, "metrics": {"accuracy": 0.5, "format_rate": 0.9}}\n'
        '{"step": 200, "metrics": {"accuracy": 0.6, "format_rate": 0.8}}\n',
        encoding="utf-8",
    )
    checkpoint = tmp_path / "checkpoint-200"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")

    manifest = export_best_checkpoint(
        tokenizer=FakePublisher(("tokenizer.json",)),
        output_dir=tmp_path,
        export_kind="full_model",
        direct_load_check=lambda path: None,
    )

    assert manifest is not None
    assert manifest["export_name"] == "best"
    assert (tmp_path / "best" / "config.json").is_file()
    assert (tmp_path / "best" / "tokenizer.json").is_file()
