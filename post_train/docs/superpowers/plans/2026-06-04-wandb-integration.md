# Weights And Biases Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional wandb monitoring to SFT, LoRA, RFT, DPO, and GRPO training while keeping offline evaluation scripts local-only.

**Architecture:** Hugging Face Trainer based paths will use `TrainingArguments` / `DPOConfig` `report_to`, `run_name`, and `logging_steps`. The hand-written GRPO loop will use a small local wandb helper that initializes only when enabled, logs existing metric rows and periodic fixed-eval metrics, and always preserves local JSONL/JSON outputs.

**Tech Stack:** Python 3.12, Transformers Trainer, TRL DPOTrainer/DPOConfig, PEFT LoRA, wandb, pytest.

---

## Scope

In scope:

- Full-parameter SFT wandb logging.
- LoRA SFT wandb logging.
- RFT training wandb logging through the reused full-SFT trainer.
- DPO training wandb logging.
- GRPO training wandb logging for rollout/training metrics and fixed 50-example eval metrics.
- Config-driven enabling/disabling.
- Documentation of remote-machine setup.

Out of scope:

- `post_train/scripts/eval/evaluate_model.py` uploading anything to wandb.
- Data construction scripts uploading anything to wandb.
- Logging sample-level generations as wandb tables in the first implementation.
- Changing DPO rejected category names.
- Changing accepted-pool sampling behavior.

Default behavior:

- wandb must be disabled unless the config explicitly enables it.
- Existing local logs must remain the source of truth:
  - Trainer logs/checkpoints under `outputs/...`
  - GRPO `metrics.jsonl`
  - fixed eval `eval/step_x/eval_samples.jsonl`
  - fixed eval `eval/step_x/eval_metrics.json`

---

## File Structure

Create:

- `post_train/src/countdown/wandb_utils.py`
  - Central config parsing for wandb.
  - Optional import of `wandb`.
  - Helpers for `report_to`, `run_name`, `wandb.init`, `wandb.log`, `wandb.finish`, and eval metric prefixing.

- `post_train/tests/test_wandb_utils.py`
  - Pure unit tests for disabled/default behavior, enabled config, environment setup, metric prefixing, and fake wandb run lifecycle.

Modify:

- `post_train/scripts/sft/train_full.py`
  - Use `wandb_utils` in `build_training_arguments`.
  - Log fixed-eval aggregate metrics to wandb from `build_eval_callback` only when enabled.
  - Preserve local eval files.

- `post_train/scripts/dpo/train_dpo.py`
  - Use `wandb_utils` in `build_dpo_training_arguments`.
  - DPO fixed eval already reuses SFT eval callback, so no separate eval logging path is needed.

- `post_train/scripts/grpo/train_grpo.py`
  - Initialize wandb for hand-written GRPO loop.
  - Log every metric row written to `metrics.jsonl`.
  - Log fixed-eval aggregate metrics from `run_fixed_eval`.
  - Finish the wandb run at the end, including exception paths.

- `post_train/configs/sft_full.yaml`
- `post_train/configs/sft_lora.yaml`
- `post_train/configs/rft.yaml`
- `post_train/configs/dpo_train.yaml`
- `post_train/configs/grpo.yaml`
  - Add conservative disabled-by-default wandb fields.

- `post_train/README.md`
- `post_train/configs/README.md`
- `post_train/scripts/sft/README.md`
- `post_train/scripts/dpo/README.md`
- `post_train/scripts/grpo/README.md`
  - Document how to enable wandb and which scripts log to it.

Do not modify:

- `post_train/scripts/eval/evaluate_model.py`
- `post_train/scripts/data/*.py`
- `post_train/scripts/sft/build_rft_data.py`
- `post_train/scripts/dpo/build_dpo_data.py`

---

## Config Schema

Add these fields to `sft_full.yaml`, `sft_lora.yaml`, `dpo_train.yaml`, and `grpo.yaml`:

```yaml
report_to: null
wandb_project: countdown-post-train
wandb_entity: null
wandb_group: null
wandb_tags: []
run_name: sft_full
logging_steps: 10
```

For `rft.yaml`, add the training fields under `train:` because `train_full.normalize_sft_config()` maps `rft.yaml.train` to the full-SFT config:

```yaml
train:
  report_to: null
  wandb_project: countdown-post-train
  wandb_entity: null
  wandb_group: null
  wandb_tags: []
  run_name: rft
  logging_steps: 10
```

Enable wandb by setting:

```yaml
report_to: wandb
```

or, for RFT:

```yaml
train:
  report_to: wandb
```

Environment setup on the training machine:

```bash
pip install wandb
wandb login
```

or:

```bash
export WANDB_API_KEY=...
```

---

## Task 1: Add Shared wandb Utility

**Files:**

- Create: `post_train/src/countdown/wandb_utils.py`
- Create: `post_train/tests/test_wandb_utils.py`

- [ ] **Step 1: Write failing utility tests**

Create `post_train/tests/test_wandb_utils.py`:

```python
import sys
import types

from post_train.src.countdown.wandb_utils import (
    build_wandb_init_kwargs,
    is_wandb_enabled,
    prefixed_metrics,
    trainer_report_to,
    wandb_run_name,
)


def test_wandb_disabled_by_default():
    cfg = {}

    assert is_wandb_enabled(cfg) is False
    assert trainer_report_to(cfg) == []
    assert wandb_run_name(cfg) is None


def test_wandb_enabled_by_report_to_string():
    cfg = {"report_to": "wandb", "run_name": "sft_full"}

    assert is_wandb_enabled(cfg) is True
    assert trainer_report_to(cfg) == ["wandb"]
    assert wandb_run_name(cfg) == "sft_full"


def test_wandb_enabled_by_report_to_list():
    cfg = {"report_to": ["tensorboard", "wandb"]}

    assert is_wandb_enabled(cfg) is True
    assert trainer_report_to(cfg) == ["tensorboard", "wandb"]


def test_build_wandb_init_kwargs_omits_empty_optional_values():
    cfg = {
        "wandb_project": "countdown-post-train",
        "wandb_entity": None,
        "wandb_group": "",
        "wandb_tags": ["sft", "full"],
        "run_name": "sft_full",
        "learning_rate": 1e-5,
    }

    assert build_wandb_init_kwargs(cfg, default_name="fallback") == {
        "project": "countdown-post-train",
        "name": "sft_full",
        "tags": ["sft", "full"],
        "config": cfg,
    }


def test_prefixed_metrics_filters_non_numeric_values():
    metrics = {
        "accuracy": 0.5,
        "format_rate": 1.0,
        "note": "skip",
        "entropy": None,
        "truncated_count": 2,
    }

    assert prefixed_metrics("eval", metrics) == {
        "eval/accuracy": 0.5,
        "eval/format_rate": 1.0,
        "eval/truncated_count": 2,
    }
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
python -m pytest post_train\tests\test_wandb_utils.py -q
```

Expected: FAIL because `post_train.src.countdown.wandb_utils` does not exist.

- [ ] **Step 3: Implement `wandb_utils.py`**

Create `post_train/src/countdown/wandb_utils.py`:

```python
from __future__ import annotations

from typing import Any


def _report_to_value(cfg: dict[str, Any]) -> str | list[str] | None:
    return cfg.get("report_to")


def trainer_report_to(cfg: dict[str, Any]) -> list[str]:
    value = _report_to_value(cfg)
    if value is None or value is False or value == "":
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    raise TypeError("report_to must be null, a string, or a list of strings")


def is_wandb_enabled(cfg: dict[str, Any]) -> bool:
    return "wandb" in trainer_report_to(cfg)


def wandb_run_name(cfg: dict[str, Any]) -> str | None:
    value = cfg.get("run_name")
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def build_wandb_init_kwargs(cfg: dict[str, Any], default_name: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "project": str(cfg.get("wandb_project") or "countdown-post-train"),
        "name": wandb_run_name(cfg) or default_name,
        "config": cfg,
    }
    entity = cfg.get("wandb_entity")
    if entity:
        kwargs["entity"] = str(entity)
    group = cfg.get("wandb_group")
    if group:
        kwargs["group"] = str(group)
    tags = cfg.get("wandb_tags") or []
    if tags:
        kwargs["tags"] = [str(tag) for tag in tags]
    return kwargs


def init_wandb_if_enabled(cfg: dict[str, Any], default_name: str):
    if not is_wandb_enabled(cfg):
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("wandb logging is enabled, but the 'wandb' package is not installed") from exc
    return wandb.init(**build_wandb_init_kwargs(cfg, default_name=default_name))


def log_wandb_metrics(run, metrics: dict[str, Any], step: int | None = None) -> None:
    if run is None:
        return
    if step is None:
        run.log(metrics)
    else:
        run.log(metrics, step=step)


def finish_wandb(run) -> None:
    if run is not None:
        run.finish()


def prefixed_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, float | int]:
    output: dict[str, float | int] = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            output[f"{prefix}/{key}"] = value
    return output
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```powershell
python -m pytest post_train\tests\test_wandb_utils.py -q
```

Expected: PASS.

---

## Task 2: Add Trainer wandb Arguments For SFT, LoRA, And RFT

**Files:**

- Modify: `post_train/scripts/sft/train_full.py`
- Modify: `post_train/tests/test_train_full.py`

- [ ] **Step 1: Add failing tests for Trainer argument config**

Append to `post_train/tests/test_train_full.py`:

```python
def test_normalize_sft_config_preserves_rft_wandb_train_fields():
    cfg = normalize_sft_config(
        {
            "accepted_output": "post_train/data/sft/rft_accepted.jsonl",
            "output_dir": "post_train/outputs/sft/rft",
            "train": {
                "max_seq_len": 256,
                "learning_rate": 1e-5,
                "warmup_ratio": 0.03,
                "scheduler": "cosine",
                "epochs": 2,
                "per_device_train_batch_size": 4,
                "gradient_accumulation_steps": 4,
                "bf16": True,
                "gradient_checkpointing": True,
                "report_to": "wandb",
                "wandb_project": "countdown-post-train",
                "run_name": "rft",
                "logging_steps": 5,
            },
        }
    )

    assert cfg["report_to"] == "wandb"
    assert cfg["wandb_project"] == "countdown-post-train"
    assert cfg["run_name"] == "rft"
    assert cfg["logging_steps"] == 5
```

Create a lightweight fake `TrainingArguments` test by monkeypatching `transformers`:

```python
def test_build_training_arguments_uses_wandb_config(monkeypatch, tmp_path):
    from post_train.scripts.sft import train_full

    captured = {}

    class FakeTrainingArguments:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(
        __import__("sys").modules,
        "transformers",
        type("FakeTransformers", (), {"TrainingArguments": FakeTrainingArguments}),
    )

    train_full.build_training_arguments(
        {
            "epochs": 1,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 1e-5,
            "weight_decay": 0.0,
            "warmup_ratio": 0.03,
            "scheduler": "cosine",
            "bf16": False,
            "gradient_checkpointing": False,
            "save_every_steps": 100,
            "report_to": "wandb",
            "run_name": "sft_full",
            "logging_steps": 7,
        },
        tmp_path,
        max_steps=2,
    )

    assert captured["report_to"] == ["wandb"]
    assert captured["run_name"] == "sft_full"
    assert captured["logging_steps"] == 7
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
python -m pytest post_train\tests\test_train_full.py -q
```

Expected: FAIL because `build_training_arguments()` still hardcodes `report_to=[]`, lacks `run_name`, and hardcodes `logging_steps=10`.

- [ ] **Step 3: Update `train_full.py`**

Modify imports:

```python
from post_train.src.countdown.wandb_utils import trainer_report_to, wandb_run_name
```

Modify `build_training_arguments()`:

```python
return TrainingArguments(
    output_dir=str(output_dir),
    overwrite_output_dir=False,
    num_train_epochs=float(cfg["epochs"]),
    max_steps=max_steps if max_steps is not None else -1,
    per_device_train_batch_size=int(cfg["per_device_train_batch_size"]),
    gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
    learning_rate=float(cfg["learning_rate"]),
    weight_decay=float(cfg["weight_decay"]),
    warmup_ratio=float(cfg["warmup_ratio"]),
    lr_scheduler_type=str(cfg.get("scheduler", "cosine")),
    bf16=bool(cfg.get("bf16", False)),
    gradient_checkpointing=bool(cfg.get("gradient_checkpointing", False)),
    save_strategy="steps",
    save_steps=int(cfg["save_every_steps"]),
    logging_steps=int(cfg.get("logging_steps", 10)),
    report_to=trainer_report_to(cfg),
    run_name=wandb_run_name(cfg),
    remove_unused_columns=False,
)
```

No change is needed in `train_lora.py` or `train_rft.py` for training arguments because both reuse `train_full.py`.

- [ ] **Step 4: Run tests to verify pass**

Run:

```powershell
python -m pytest post_train\tests\test_train_full.py post_train\tests\test_wandb_utils.py -q
```

Expected: PASS.

---

## Task 3: Log Fixed Eval Metrics To wandb For Trainer-Based Training

**Files:**

- Modify: `post_train/scripts/sft/train_full.py`
- Modify: `post_train/tests/test_train_full.py`

- [ ] **Step 1: Add failing test for eval metric prefixing path**

Add a pure helper to `train_full.py` first only as a tested design target in the test:

```python
def test_build_eval_wandb_metrics_prefixes_numeric_values():
    from post_train.scripts.sft.train_full import build_eval_wandb_metrics

    assert build_eval_wandb_metrics({"accuracy": 0.4, "note": "skip", "truncated_count": 1}) == {
        "eval/accuracy": 0.4,
        "eval/truncated_count": 1,
    }
```

- [ ] **Step 2: Run test to verify failure**

Run:

```powershell
python -m pytest post_train\tests\test_train_full.py::test_build_eval_wandb_metrics_prefixes_numeric_values -q
```

Expected: FAIL because `build_eval_wandb_metrics` does not exist.

- [ ] **Step 3: Implement eval wandb helper and callback logging**

Modify imports in `train_full.py`:

```python
from post_train.src.countdown.wandb_utils import (
    log_wandb_metrics,
    prefixed_metrics,
    trainer_report_to,
    wandb_run_name,
)
```

Add helper:

```python
def build_eval_wandb_metrics(metrics: dict[str, Any]) -> dict[str, float | int]:
    return prefixed_metrics("eval", metrics)
```

Modify `build_eval_callback()` after local `write_json()`:

```python
            run = getattr(args, "_wandb_run", None)
            log_wandb_metrics(
                run,
                build_eval_wandb_metrics(metrics),
                step=int(state.global_step),
            )
```

If `TrainingArguments` cannot carry `_wandb_run`, do not attach dynamic attributes. Instead rely on Transformers' built-in wandb callback for train loss and skip fixed-eval wandb logging in Task 3. In that case, remove Task 3 implementation and keep fixed eval local-only for Trainer paths. GRPO still gets fixed-eval wandb logging because it owns the run. This fallback must be documented in the final report.

- [ ] **Step 4: Run tests**

Run:

```powershell
python -m pytest post_train\tests\test_train_full.py -q
```

Expected: PASS.

---

## Task 4: Add DPO wandb Training Arguments

**Files:**

- Modify: `post_train/scripts/dpo/train_dpo.py`
- Modify: `post_train/tests/test_train_dpo.py`

- [ ] **Step 1: Add failing test for DPO config passthrough**

Append to `post_train/tests/test_train_dpo.py`:

```python
def test_build_dpo_training_arguments_uses_wandb_config(monkeypatch, tmp_path):
    import sys
    from post_train.scripts.dpo import train_dpo

    captured = {}

    class FakeDPOConfig:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "trl", type("FakeTRL", (), {"DPOConfig": FakeDPOConfig}))

    train_dpo.build_dpo_training_arguments(
        {
            "epochs": 1,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 5e-7,
            "weight_decay": 0.0,
            "warmup_ratio": 0.03,
            "scheduler": "cosine",
            "bf16": False,
            "gradient_checkpointing": False,
            "save_every_steps": 100,
            "eval_every_steps": 100,
            "beta": 0.05,
            "max_seq_len": 256,
            "report_to": "wandb",
            "run_name": "dpo",
            "logging_steps": 8,
        },
        tmp_path,
        max_steps=2,
    )

    assert captured["report_to"] == ["wandb"]
    assert captured["run_name"] == "dpo"
    assert captured["logging_steps"] == 8
```

- [ ] **Step 2: Run test to verify failure**

Run:

```powershell
python -m pytest post_train\tests\test_train_dpo.py -q
```

Expected: FAIL because DPO training args hardcode `report_to=[]`, `logging_steps=10`, and do not pass `run_name`.

- [ ] **Step 3: Update `train_dpo.py`**

Add import:

```python
from post_train.src.countdown.wandb_utils import trainer_report_to, wandb_run_name
```

Update kwargs in `build_dpo_training_arguments()`:

```python
"logging_steps": int(cfg.get("logging_steps", 10)),
"report_to": trainer_report_to(cfg),
"run_name": wandb_run_name(cfg),
```

- [ ] **Step 4: Run DPO tests**

Run:

```powershell
python -m pytest post_train\tests\test_train_dpo.py post_train\tests\test_wandb_utils.py -q
```

Expected: PASS.

---

## Task 5: Add GRPO Manual wandb Logging

**Files:**

- Modify: `post_train/scripts/grpo/train_grpo.py`
- Modify: `post_train/tests/test_grpo_metrics.py`

- [ ] **Step 1: Add failing tests for GRPO wandb helper behavior**

Append to `post_train/tests/test_grpo_metrics.py`:

```python
def test_build_grpo_wandb_metrics_prefixes_training_metric_row():
    from post_train.scripts.grpo.train_grpo import build_grpo_wandb_metrics

    metric = {
        "step": 3,
        "loss": 0.5,
        "mean_reward": 0.2,
        "entropy": None,
        "note": "skip",
    }

    assert build_grpo_wandb_metrics(metric) == {
        "train/loss": 0.5,
        "train/mean_reward": 0.2,
    }


def test_build_grpo_eval_wandb_metrics_prefixes_eval_metrics():
    from post_train.scripts.grpo.train_grpo import build_grpo_eval_wandb_metrics

    assert build_grpo_eval_wandb_metrics({"accuracy": 1.0, "truncated_count": 0}) == {
        "eval/accuracy": 1.0,
        "eval/truncated_count": 0,
    }
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
python -m pytest post_train\tests\test_grpo_metrics.py -q
```

Expected: FAIL because the two helper functions do not exist.

- [ ] **Step 3: Implement GRPO metric helpers**

Add imports:

```python
from post_train.src.countdown.wandb_utils import (
    finish_wandb,
    init_wandb_if_enabled,
    log_wandb_metrics,
    prefixed_metrics,
)
```

Add helpers:

```python
def build_grpo_wandb_metrics(metric: dict[str, Any]) -> dict[str, float | int]:
    return prefixed_metrics(
        "train",
        {key: value for key, value in metric.items() if key != "step"},
    )


def build_grpo_eval_wandb_metrics(metrics: dict[str, Any]) -> dict[str, float | int]:
    return prefixed_metrics("eval", metrics)
```

- [ ] **Step 4: Wire wandb run into `train_grpo()`**

At the beginning of `train_grpo()` after `output_dir.mkdir(...)`:

```python
    wandb_run = init_wandb_if_enabled(cfg, default_name="grpo")
```

Wrap the training loop in `try/finally`:

```python
    try:
        while global_step < max_steps:
            ...
    finally:
        finish_wandb(wandb_run)
```

After `write_metric(output_dir, metric)`:

```python
            log_wandb_metrics(
                wandb_run,
                build_grpo_wandb_metrics(metric),
                step=global_step,
            )
```

Modify `run_fixed_eval()` signature:

```python
def run_fixed_eval(..., wandb_run=None) -> None:
```

After writing `eval_metrics.json`:

```python
    log_wandb_metrics(
        wandb_run,
        build_grpo_eval_wandb_metrics(metrics),
        step=step,
    )
```

Update call site:

```python
run_fixed_eval(model, tokenizer, output_dir, global_step, eval_rows, eval_cfg, wandb_run=wandb_run)
```

- [ ] **Step 5: Run GRPO tests**

Run:

```powershell
python -m pytest post_train\tests\test_grpo_metrics.py post_train\tests\test_wandb_utils.py -q
```

Expected: PASS.

---

## Task 6: Add Config Defaults

**Files:**

- Modify: `post_train/configs/sft_full.yaml`
- Modify: `post_train/configs/sft_lora.yaml`
- Modify: `post_train/configs/rft.yaml`
- Modify: `post_train/configs/dpo_train.yaml`
- Modify: `post_train/configs/grpo.yaml`
- Test: `post_train/tests/test_config_io.py`

- [ ] **Step 1: Add config schema test**

Append to `post_train/tests/test_config_io.py`:

```python
from pathlib import Path

from post_train.src.countdown.config import load_yaml_config


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
        assert isinstance(cfg["run_name"], str)
        assert int(cfg["logging_steps"]) > 0

    rft_cfg = load_yaml_config("post_train/configs/rft.yaml")
    train_cfg = rft_cfg["train"]
    assert train_cfg["report_to"] is None
    assert train_cfg["wandb_project"] == "countdown-post-train"
    assert train_cfg["run_name"] == "rft"
    assert int(train_cfg["logging_steps"]) > 0
```

- [ ] **Step 2: Run test to verify failure**

Run:

```powershell
python -m pytest post_train\tests\test_config_io.py::test_training_configs_include_disabled_wandb_defaults -q
```

Expected: FAIL because config defaults do not exist.

- [ ] **Step 3: Update config files**

Add to `post_train/configs/sft_full.yaml`:

```yaml
report_to: null
wandb_project: countdown-post-train
wandb_entity: null
wandb_group: null
wandb_tags:
  - sft
  - full
run_name: sft_full
logging_steps: 10
```

Add to `post_train/configs/sft_lora.yaml`:

```yaml
report_to: null
wandb_project: countdown-post-train
wandb_entity: null
wandb_group: null
wandb_tags:
  - sft
  - lora
run_name: sft_lora
logging_steps: 10
```

Add under `train:` in `post_train/configs/rft.yaml`:

```yaml
  report_to: null
  wandb_project: countdown-post-train
  wandb_entity: null
  wandb_group: null
  wandb_tags:
    - sft
    - rft
  run_name: rft
  logging_steps: 10
```

Add to `post_train/configs/dpo_train.yaml`:

```yaml
report_to: null
wandb_project: countdown-post-train
wandb_entity: null
wandb_group: null
wandb_tags:
  - dpo
run_name: dpo
logging_steps: 10
```

Add to `post_train/configs/grpo.yaml`:

```yaml
report_to: null
wandb_project: countdown-post-train
wandb_entity: null
wandb_group: null
wandb_tags:
  - grpo
run_name: grpo
logging_steps: 10
```

- [ ] **Step 4: Run config tests**

Run:

```powershell
python -m pytest post_train\tests\test_config_io.py -q
```

Expected: PASS.

---

## Task 7: Update Documentation

**Files:**

- Modify: `post_train/README.md`
- Modify: `post_train/configs/README.md`
- Modify: `post_train/scripts/sft/README.md`
- Modify: `post_train/scripts/dpo/README.md`
- Modify: `post_train/scripts/grpo/README.md`
- Modify: `post_train/docs/superpowers/specs/2026-06-03-countdown-post-training-design.md`

- [ ] **Step 1: Update root README**

Add section:

```markdown
## Optional wandb Monitoring

Training scripts support optional wandb logging. It is disabled by default with `report_to: null`.

Enable it in the relevant training config:

```yaml
report_to: wandb
wandb_project: countdown-post-train
run_name: sft_full
```

On the training machine:

```bash
pip install wandb
wandb login
```

The offline evaluator does not upload to wandb:

```bash
python post_train/scripts/eval/evaluate_model.py --config post_train/configs/eval.yaml --model-path post_train/outputs/sft/full/final --output-dir post_train/data/eval/sft_full
```
```

- [ ] **Step 2: Update config README**

Add:

```markdown
Training configs include disabled-by-default wandb fields:

- `report_to`: set to `wandb` to enable.
- `wandb_project`: default project name.
- `wandb_entity`: optional team/user.
- `wandb_group`: optional run grouping.
- `wandb_tags`: optional tags.
- `run_name`: run display name.
- `logging_steps`: Trainer/loop logging cadence.
```

- [ ] **Step 3: Update script READMEs**

In SFT, DPO, and GRPO README files, add a one-paragraph note:

```markdown
wandb logging is optional and disabled by default. Set `report_to: wandb` in the corresponding config to enable it. SFT/LoRA/RFT/DPO use Trainer integration; GRPO logs its local metric rows manually.
```

- [ ] **Step 4: Update design spec implemented clarifications**

Add:

```markdown
- wandb monitoring is optional and config-driven. Trainer-based SFT/LoRA/RFT/DPO use `report_to`; GRPO logs metric rows manually. The standalone evaluator remains local-only and does not upload to wandb.
```

- [ ] **Step 5: Verify docs mention evaluator local-only**

Run:

```powershell
rg -n "wandb|evaluator does not upload|local-only" post_train\README.md post_train\configs\README.md post_train\scripts post_train\docs\superpowers\specs
```

Expected: wandb docs appear in training docs, and evaluator local-only is explicitly stated.

---

## Task 8: Final Verification And Review

**Files:**

- All modified files.

- [ ] **Step 1: Run full unit tests**

Run:

```powershell
python -m pytest post_train\tests -q --basetemp post_train\.pytest_tmp
```

Expected: all tests pass.

- [ ] **Step 2: Run py_compile for modified scripts**

Run:

```powershell
python -m py_compile post_train\src\countdown\wandb_utils.py post_train\scripts\sft\train_full.py post_train\scripts\sft\train_lora.py post_train\scripts\sft\train_rft.py post_train\scripts\dpo\train_dpo.py post_train\scripts\grpo\train_grpo.py
```

Expected: no syntax errors.

- [ ] **Step 3: Run no-wandb smoke checks for config parsing**

Run:

```powershell
python -m pytest post_train\tests\test_wandb_utils.py post_train\tests\test_train_full.py post_train\tests\test_train_dpo.py post_train\tests\test_grpo_metrics.py -q
```

Expected: all tests pass with wandb disabled by default.

- [ ] **Step 4: Record GPU-dependent checks not run locally**

Record in final report:

- `python post_train/scripts/sft/train_full.py --config post_train/configs/sft_full.yaml --max-steps 2` with `report_to: wandb` must be run on the training host.
- `python post_train/scripts/dpo/train_dpo.py --config post_train/configs/dpo_train.yaml --max-steps 2` with `report_to: wandb` must be run on the training host.
- `python post_train/scripts/grpo/train_grpo.py --config post_train/configs/grpo.yaml --max-steps 2` with `report_to: wandb` must be run on the training host.

- [ ] **Step 5: Perform post-implementation review**

Review checklist:

- wandb disabled by default.
- Enabling wandb does not affect data construction.
- `evaluate_model.py` remains local-only.
- Trainer-based scripts use config-driven `report_to`, `run_name`, and `logging_steps`.
- GRPO logs all existing training metric rows to wandb when enabled.
- GRPO fixed-eval aggregate metrics are logged to wandb when enabled.
- Local JSONL/JSON logs are still written.
- No wandb import occurs unless wandb is enabled for manual GRPO path.
- Tests cover config parsing and metric prefixing.

---

## Plan Self-Review

Spec coverage:

- SFT full wandb: Task 2 and Task 6.
- LoRA wandb: Task 2 and Task 6 through shared SFT trainer.
- RFT wandb: Task 2 and Task 6 through `normalize_sft_config`.
- DPO wandb: Task 4 and Task 6.
- GRPO wandb: Task 5 and Task 6.
- Fixed eval metrics: Task 3 for Trainer path if feasible, Task 5 for GRPO.
- Evaluator local-only: explicitly out of scope and documented in Task 7.
- Docs and final review: Task 7 and Task 8.

Placeholder scan:

- No TBD/TODO placeholders.
- Each task has concrete file paths, code snippets, commands, and expected results.

Type consistency:

- Shared utility names are consistent:
  - `trainer_report_to`
  - `wandb_run_name`
  - `init_wandb_if_enabled`
  - `log_wandb_metrics`
  - `finish_wandb`
  - `prefixed_metrics`
- Config field names are consistent across SFT, LoRA, RFT, DPO, and GRPO.

