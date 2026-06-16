from __future__ import annotations

import json
from types import SimpleNamespace

from post_train_v2.src.training import fixed_eval
from post_train_v2.src.training.fixed_eval import FixedEvaluationCallback


class FakeModel:
    def __init__(self):
        self.training = True
        self.events = []

    def eval(self):
        self.training = False
        self.events.append("eval")

    def train(self):
        self.training = True
        self.events.append("train")


class FakeAccelerator:
    def __init__(self, unwrapped):
        self.unwrapped = unwrapped
        self.calls = 0

    def unwrap_model(self, model):
        self.calls += 1
        return self.unwrapped


def _state(step: int, max_steps: int = 250):
    return SimpleNamespace(global_step=step, max_steps=max_steps)


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_fixed_eval_runs_on_boundaries_and_final_non_boundary(tmp_path, monkeypatch):
    calls = []
    model = FakeModel()
    accelerator = FakeAccelerator(model)

    def fake_evaluate_rows(rows, tokenizer, evaluated_model, *, max_new_tokens):
        calls.append((evaluated_model, max_new_tokens))
        return [
            {
                "id": "row-1",
                "prompt": rows[0]["prompt"],
                "raw_generation": "<answer>1+1</answer>",
                "extracted_expr": "1+1",
                "format_ok": True,
                "valid_expression": True,
                "correct": True,
                "error": None,
                "value": "2",
                "generated_tokens": 3,
                "truncated": False,
            }
        ]

    monkeypatch.setattr(fixed_eval, "evaluate_rows", fake_evaluate_rows)
    callback = FixedEvaluationCallback(
        eval_rows=[{"id": "row-1", "prompt": "q", "numbers": [1, 1], "target": 2}],
        tokenizer=object(),
        output_dir=tmp_path,
        eval_every_steps=100,
        max_new_tokens=128,
    )

    for step in (99, 100, 150, 200, 250):
        callback.on_step_end(
            args=None,
            state=_state(step),
            control=SimpleNamespace(),
            model=SimpleNamespace(events=["wrapper-forward-not-called"]),
            accelerator=accelerator,
        )

    assert len(calls) == 3
    assert [call[0] for call in calls] == [model, model, model]
    assert [call[1] for call in calls] == [128, 128, 128]
    assert accelerator.calls == 3
    assert model.training is True
    assert model.events == ["eval", "train", "eval", "train", "eval", "train"]
    assert _read_json(tmp_path / "step_100" / "metrics.json")["accuracy"] == 1.0
    assert _read_json(tmp_path / "step_200" / "metrics.json")["format_rate"] == 1.0
    ledger_lines = (tmp_path / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["step"] for line in ledger_lines] == [100, 200, 250]


def test_fixed_eval_restores_eval_mode(tmp_path, monkeypatch):
    calls = []
    model = FakeModel()
    model.training = False
    monkeypatch.setattr(
        fixed_eval,
        "evaluate_rows",
        lambda rows, tokenizer, evaluated_model, *, max_new_tokens: calls.append(1)
        or [],
    )
    callback = FixedEvaluationCallback(
        eval_rows=[{"id": "row-1", "prompt": "q", "numbers": [1], "target": 1}],
        tokenizer=object(),
        output_dir=tmp_path,
    )

    callback.on_step_end(
        args=None,
        state=_state(100),
        control=SimpleNamespace(),
        model=model,
    )

    assert calls == [1]
    assert model.training is False
    assert model.events == ["eval"]
