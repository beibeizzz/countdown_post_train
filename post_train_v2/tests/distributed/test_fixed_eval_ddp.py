from __future__ import annotations

from types import SimpleNamespace

from post_train_v2.src.training import fixed_eval
from post_train_v2.src.training.fixed_eval import FixedEvaluationCallback


def test_nonzero_rank_enters_main_section_but_does_not_generate(tmp_path, monkeypatch):
    events = []

    def fake_main_rank_section(fn):
        events.append("entered-section")
        return None

    monkeypatch.setattr(fixed_eval, "main_rank_section", fake_main_rank_section)
    monkeypatch.setattr(
        fixed_eval,
        "evaluate_rows",
        lambda *args, **kwargs: events.append("generated") or [],
    )
    callback = FixedEvaluationCallback(
        eval_rows=[{"id": "row-1", "prompt": "q", "numbers": [1], "target": 1}],
        tokenizer=object(),
        output_dir=tmp_path,
    )

    callback.on_step_end(
        args=None,
        state=SimpleNamespace(global_step=100, max_steps=100),
        control=SimpleNamespace(),
        model=object(),
        accelerator=SimpleNamespace(unwrap_model=lambda model: model),
    )

    assert events == ["entered-section"]
