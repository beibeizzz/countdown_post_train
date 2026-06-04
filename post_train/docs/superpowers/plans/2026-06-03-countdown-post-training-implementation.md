# Countdown Post-Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first end-to-end Countdown post-training project under `post_train/`, covering data warehouse construction, teacher-filtered SFT data, SFT/RFT/DPO/GRPO script scaffolding, and common evaluation.

**Architecture:** Keep the old `sft/` and `rlhf/` folders as references and put all new code in `post_train/`. Shared Countdown logic lives in `post_train/src/countdown/`; stage scripts in `post_train/scripts/` call that shared logic and are driven by per-stage YAML configs in `post_train/configs/`.

**Tech Stack:** Python 3.12, PyTorch 2.8.0, Transformers, TRL, PEFT, vLLM, pandas, PyYAML, pytest, fractions.Fraction for exact arithmetic validation.

---

## Scope And Execution Notes

This plan implements the first complete project version. It is intentionally split into small tasks so later execution can stop after any verified checkpoint.

The current workspace root is not a git repository. Commit steps are included for execution environments that do have `.git`; if `git status --short` returns `fatal: not a git repository`, skip the commit step and record that in the final report.

The vLLM and training tasks require the target Ubuntu environment with local models at:

- `post_train/model/qwen/qwen3-0.6b`
- `post_train/model/qwen/qwen3-8b`

Local Windows verification should focus on pure Python tests and data-building dry runs.

## File Structure

Create or modify these files:

- Create `post_train/configs/data_build.yaml`: source, split, and sampling defaults.
- Create `post_train/configs/teacher_rollout.yaml`: teacher vLLM generation defaults.
- Create `post_train/configs/sft_full.yaml`: full-parameter SFT defaults.
- Create `post_train/configs/sft_lora.yaml`: LoRA SFT defaults.
- Create `post_train/configs/rft.yaml`: rejection-sampling SFT defaults.
- Create `post_train/configs/dpo_data.yaml`: DPO rejected generation defaults.
- Create `post_train/configs/dpo_train.yaml`: DPO training defaults.
- Create `post_train/configs/grpo.yaml`: GRPO defaults.
- Create `post_train/configs/eval.yaml`: common evaluation defaults.
- Create `post_train/src/countdown/__init__.py`: package exports.
- Create `post_train/src/countdown/config.py`: YAML loading and path resolution.
- Create `post_train/src/countdown/solver.py`: exact Countdown solver and expression metadata.
- Create `post_train/src/countdown/validation.py`: answer extraction and Fraction-based AST validation.
- Create `post_train/src/countdown/bucketing.py`: `num_count + complexity` bucket assignment.
- Create `post_train/src/countdown/prompts.py`: all prompt builders.
- Create `post_train/src/countdown/sampling.py`: stratified sampling utilities.
- Create `post_train/src/countdown/io.py`: JSONL, JSON, parquet helpers and manifests.
- Create `post_train/src/countdown/generation.py`: Transformers and vLLM generation wrappers.
- Create `post_train/src/countdown/eval.py`: sample-level and aggregate evaluator.
- Create `post_train/scripts/data/build_source.py`: build solver-backed source, validation, eval subset, and test files.
- Create `post_train/scripts/data/build_teacher_pool.py`: teacher rollout and accepted-pool construction.
- Create `post_train/scripts/data/build_sft_splits.py`: 8k SFT and 4k GRPO stratified samples.
- Create `post_train/scripts/sft/train_full.py`: full-parameter SFT.
- Create `post_train/scripts/sft/train_lora.py`: LoRA SFT.
- Create `post_train/scripts/sft/build_rft_data.py`: base-model rejection sampling data.
- Create `post_train/scripts/dpo/build_dpo_data.py`: DPO pair construction and filtering.
- Create `post_train/scripts/dpo/train_dpo.py`: DPO training.
- Create `post_train/scripts/grpo/train_grpo.py`: GRPO training entrypoint.
- Create `post_train/scripts/eval/evaluate_model.py`: standalone model evaluation.
- Create `post_train/tests/test_validation.py`: expression validator tests.
- Create `post_train/tests/test_bucketing_sampling.py`: bucketing and stratified sampling tests.
- Create `post_train/tests/test_prompts.py`: prompt construction tests.
- Create `post_train/tests/test_eval_metrics.py`: evaluator tests.
- Create `post_train/README.md`: workflow commands.

---

### Task 1: Project Skeleton And Config Files

**Files:**
- Create: `post_train/configs/data_build.yaml`
- Create: `post_train/configs/teacher_rollout.yaml`
- Create: `post_train/configs/sft_full.yaml`
- Create: `post_train/configs/sft_lora.yaml`
- Create: `post_train/configs/rft.yaml`
- Create: `post_train/configs/dpo_data.yaml`
- Create: `post_train/configs/dpo_train.yaml`
- Create: `post_train/configs/grpo.yaml`
- Create: `post_train/configs/eval.yaml`
- Create: `post_train/src/countdown/__init__.py`
- Create: `post_train/scripts/data/.gitkeep`
- Create: `post_train/scripts/sft/.gitkeep`
- Create: `post_train/scripts/dpo/.gitkeep`
- Create: `post_train/scripts/grpo/.gitkeep`
- Create: `post_train/scripts/eval/.gitkeep`
- Create: `post_train/tests/.gitkeep`

- [ ] **Step 1: Create directories**

Run:

```powershell
New-Item -ItemType Directory -Force -Path post_train\configs,post_train\src\countdown,post_train\scripts\data,post_train\scripts\sft,post_train\scripts\dpo,post_train\scripts\grpo,post_train\scripts\eval,post_train\tests,post_train\data\raw,post_train\data\processed,post_train\data\teacher_rollouts,post_train\data\sft,post_train\data\dpo,post_train\data\grpo,post_train\data\eval,post_train\outputs\sft\full,post_train\outputs\sft\lora,post_train\outputs\sft\rft,post_train\outputs\dpo,post_train\outputs\grpo
```

Expected: directories exist under `post_train/`.

- [ ] **Step 2: Add `post_train/src/countdown/__init__.py`**

Create:

```python
"""Shared Countdown post-training utilities."""
```

- [ ] **Step 3: Add config YAML files**

Use the exact defaults from the design spec:

```yaml
# post_train/configs/data_build.yaml
seed: 42
train_input: datasets/raw_train.parquet
test_input: datasets/raw_test.json
output_dir: post_train/data/processed
val_size: 200
eval_subset_size: 50
accepted_pool_target: 20000
sft_train_target: 8000
grpo_train_target: 4000
```

```yaml
# post_train/configs/teacher_rollout.yaml
model_path: post_train/model/qwen/qwen3-8b
backend: vllm
batch_size: 64
max_new_tokens: 256
temperature: 0.2
top_p: 0.95
enable_thinking: false
stop_after_accepted: 20000
```

```yaml
# post_train/configs/sft_full.yaml
model_path: post_train/model/qwen/qwen3-0.6b
train_data: post_train/data/sft/sft_train_8k.jsonl
val_data: post_train/data/processed/val_200.jsonl
output_dir: post_train/outputs/sft/full
max_seq_len: 256
learning_rate: 1.0e-5
weight_decay: 0.0
warmup_ratio: 0.03
scheduler: cosine
epochs: 3
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
bf16: true
gradient_checkpointing: true
eval_every_steps: 100
save_every_steps: 100
```

```yaml
# post_train/configs/sft_lora.yaml
model_path: post_train/model/qwen/qwen3-0.6b
train_data: post_train/data/sft/sft_train_8k.jsonl
val_data: post_train/data/processed/val_200.jsonl
output_dir: post_train/outputs/sft/lora
max_seq_len: 256
learning_rate: 2.0e-5
weight_decay: 0.0
warmup_ratio: 0.03
scheduler: cosine
epochs: 3
per_device_train_batch_size: 8
gradient_accumulation_steps: 4
bf16: true
gradient_checkpointing: true
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target_modules: auto
eval_every_steps: 100
save_every_steps: 100
```

```yaml
# post_train/configs/rft.yaml
base_model_path: post_train/model/qwen/qwen3-8b
train_prompts: post_train/data/sft/sft_train_8k.jsonl
accepted_output: post_train/data/sft/rft_accepted.jsonl
output_dir: post_train/outputs/sft/rft
num_samples_per_prompt: 4
batch_size: 32
max_new_tokens: 256
temperature: 0.7
top_p: 0.95
enable_thinking: false
train:
  max_seq_len: 256
  learning_rate: 1.0e-5
  warmup_ratio: 0.03
  scheduler: cosine
  epochs: 2
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  bf16: true
  gradient_checkpointing: true
```

```yaml
# post_train/configs/dpo_data.yaml
model_path: post_train/model/qwen/qwen3-8b
chosen_data: post_train/data/sft/sft_train_8k.jsonl
output_dir: post_train/data/dpo
target_pairs: 6000
forced_wrong_fraction: 0.5
high_temp_fraction: 0.5
max_new_tokens: 256
forced_wrong_temperature: 0.3
high_temp_temperature: 1.0
top_p: 0.95
batch_size: 64
enable_thinking: false
malformed_cap_fraction: 0.10
preferred_wrong_value_min_fraction: 0.70
```

```yaml
# post_train/configs/dpo_train.yaml
model_path: post_train/outputs/sft/full/final
train_data: post_train/data/dpo/dpo_train.jsonl
val_data: post_train/data/processed/val_200.jsonl
output_dir: post_train/outputs/dpo
max_seq_len: 256
learning_rate: 5.0e-7
weight_decay: 0.0
warmup_ratio: 0.03
scheduler: cosine
epochs: 1
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
bf16: true
gradient_checkpointing: true
beta: 0.05
eval_every_steps: 100
save_every_steps: 100
```

```yaml
# post_train/configs/grpo.yaml
model_path: post_train/outputs/sft/full/final
train_data: post_train/data/grpo/grpo_train_4k.jsonl
val_data: post_train/data/processed/val_200.jsonl
output_dir: post_train/outputs/grpo
max_prompt_len: 256
max_new_tokens: 256
batch_size: 4
group_size: 4
policy_updates_per_rollout: 2
sync_every_steps: 20
learning_rate: 3.0e-7
weight_decay: 0.0
warmup_ratio: 0.03
scheduler: cosine
max_steps: 500
clip_eps: 0.2
kl_coeff: 0.0
format_reward: 0.2
answer_reward: 1.0
temperature: 1.0
top_p: 0.95
bf16: true
gradient_checkpointing: true
compute_entropy: false
eval_every_steps: 100
save_every_steps: 20
```

```yaml
# post_train/configs/eval.yaml
val_data: post_train/data/processed/val_200.jsonl
eval_subset: post_train/data/processed/val_eval_50.jsonl
test_data: post_train/data/processed/test_with_solver_answers.jsonl
max_new_tokens: 256
temperature: 0.0
top_p: 1.0
batch_size: 32
enable_thinking: false
```

- [ ] **Step 4: Verify config files exist**

Run:

```powershell
Get-ChildItem post_train\configs
```

Expected: all nine YAML files are listed.

- [ ] **Step 5: Commit if git is available**

Run:

```powershell
git status --short
```

Expected in a git repo: newly created config and skeleton files are shown. Then run:

```powershell
git add post_train/configs post_train/src/countdown/__init__.py post_train/scripts post_train/tests
git commit -m "chore: scaffold countdown post-training project"
```

If expected output is `fatal: not a git repository`, skip this commit.

---

### Task 2: Config And IO Utilities

**Files:**
- Create: `post_train/src/countdown/config.py`
- Create: `post_train/src/countdown/io.py`
- Test: `post_train/tests/test_config_io.py`

- [ ] **Step 1: Write config and IO tests**

Create `post_train/tests/test_config_io.py`:

```python
from pathlib import Path

from post_train.src.countdown.config import load_yaml_config, resolve_path
from post_train.src.countdown.io import read_jsonl, write_jsonl


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
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
python -m pytest post_train\tests\test_config_io.py -v
```

Expected: FAIL because `post_train.src.countdown.config` and `post_train.src.countdown.io` do not exist.

- [ ] **Step 3: Implement `config.py`**

Create `post_train/src/countdown/config.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return data


def resolve_path(path: str | Path, base_dir: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path(base_dir) / candidate
```

- [ ] **Step 4: Implement `io.py`**

Create `post_train/src/countdown/io.py`:

```python
from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
            rows.append(row)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    ensure_parent(path)
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
```

- [ ] **Step 5: Run tests and verify pass**

Run:

```powershell
python -m pytest post_train\tests\test_config_io.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit if git is available**

Run:

```powershell
git add post_train/src/countdown/config.py post_train/src/countdown/io.py post_train/tests/test_config_io.py
git commit -m "feat: add config and jsonl utilities"
```

Skip if the workspace is not a git repository.

---

### Task 3: Fraction-Based Validation

**Files:**
- Create: `post_train/src/countdown/validation.py`
- Test: `post_train/tests/test_validation.py`

- [ ] **Step 1: Write validator tests**

Create `post_train/tests/test_validation.py`:

```python
from post_train.src.countdown.validation import (
    extract_answer_text,
    validate_countdown_expression,
)


def test_extract_answer_text_returns_last_answer():
    text = "draft <answer> 1+1 </answer>\nfinal <answer> (7-3)*(8-2) </answer>"

    assert extract_answer_text(text) == "(7-3)*(8-2)"


def test_validation_accepts_correct_expression():
    result = validate_countdown_expression("(7-3)*(8-2)", [7, 3, 8, 2], 24)

    assert result.ok is True
    assert result.value == 24
    assert result.error is None


def test_validation_allows_intermediate_fraction():
    result = validate_countdown_expression("6/(1+1)", [6, 1, 1], 3)

    assert result.ok is True
    assert result.value == 3


def test_validation_rejects_number_mismatch():
    result = validate_countdown_expression("(7-3)*6", [7, 3, 8, 2], 24)

    assert result.ok is False
    assert result.error == "number_mismatch"


def test_validation_rejects_wrong_value():
    result = validate_countdown_expression("(7-3)*(8-2)", [7, 3, 8, 2], 25)

    assert result.ok is False
    assert result.error == "wrong_value"


def test_validation_rejects_unsupported_syntax():
    result = validate_countdown_expression("__import__('os').system('echo bad')", [1], 1)

    assert result.ok is False
    assert result.error == "invalid_expression"
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
python -m pytest post_train\tests\test_validation.py -v
```

Expected: FAIL because `validation.py` does not exist.

- [ ] **Step 3: Implement validator**

Create `post_train/src/countdown/validation.py`:

```python
from __future__ import annotations

import ast
import re
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction


ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", flags=re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    value: int | None
    used_numbers: list[int]
    expression: str | None
    error: str | None


def extract_answer_text(text: str) -> str | None:
    if not text:
        return None
    matches = ANSWER_RE.findall(text)
    if not matches:
        return None
    return matches[-1].strip()


def has_complete_answer_tag(text: str) -> bool:
    return extract_answer_text(text) is not None


def validate_countdown_response(text: str, numbers: list[int], target: int) -> ValidationResult:
    expr = extract_answer_text(text)
    if expr is None:
        return ValidationResult(False, None, [], None, "missing_answer_tag")
    return validate_countdown_expression(expr, numbers, target)


def validate_countdown_expression(expr: str, numbers: list[int], target: int) -> ValidationResult:
    cleaned = expr.strip()
    if not cleaned:
        return ValidationResult(False, None, [], cleaned, "invalid_expression")
    try:
        parsed = ast.parse(cleaned, mode="eval")
        value, used_numbers = _eval_node(parsed.body)
    except Exception:
        return ValidationResult(False, None, [], cleaned, "invalid_expression")

    if Counter(used_numbers) != Counter(int(x) for x in numbers):
        return ValidationResult(False, _to_int_or_none(value), used_numbers, cleaned, "number_mismatch")
    if value != Fraction(int(target), 1):
        return ValidationResult(False, _to_int_or_none(value), used_numbers, cleaned, "wrong_value")
    return ValidationResult(True, int(value), used_numbers, cleaned, None)


def _eval_node(node: ast.AST) -> tuple[Fraction, list[int]]:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        value = int(node.value)
        return Fraction(value, 1), [value]
    if not isinstance(node, ast.BinOp):
        raise ValueError("unsupported node")

    left_value, left_numbers = _eval_node(node.left)
    right_value, right_numbers = _eval_node(node.right)

    if isinstance(node.op, ast.Add):
        value = left_value + right_value
    elif isinstance(node.op, ast.Sub):
        value = left_value - right_value
    elif isinstance(node.op, ast.Mult):
        value = left_value * right_value
    elif isinstance(node.op, ast.Div):
        if right_value == 0:
            raise ValueError("division by zero")
        value = left_value / right_value
    else:
        raise ValueError("unsupported operator")

    return value, left_numbers + right_numbers


def _to_int_or_none(value: Fraction) -> int | None:
    if value.denominator == 1:
        return int(value)
    return None
```

- [ ] **Step 4: Run validator tests**

Run:

```powershell
python -m pytest post_train\tests\test_validation.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit if git is available**

Run:

```powershell
git add post_train/src/countdown/validation.py post_train/tests/test_validation.py
git commit -m "feat: add countdown response validation"
```

Skip if the workspace is not a git repository.

---

### Task 4: Solver And Bucketing

**Files:**
- Create: `post_train/src/countdown/solver.py`
- Create: `post_train/src/countdown/bucketing.py`
- Test: `post_train/tests/test_bucketing_sampling.py`

- [ ] **Step 1: Write solver and bucketing tests**

Create `post_train/tests/test_bucketing_sampling.py`:

```python
from post_train.src.countdown.bucketing import assign_bucket
from post_train.src.countdown.solver import expression_metadata, solve_countdown
from post_train.src.countdown.validation import validate_countdown_expression


def test_solve_countdown_finds_valid_expression():
    expr = solve_countdown([7, 3, 8, 2], 24)

    assert expr is not None
    assert validate_countdown_expression(expr, [7, 3, 8, 2], 24).ok is True


def test_expression_metadata_detects_division_and_depth():
    meta = expression_metadata("6/(1+1)", num_count=3)

    assert meta["has_division"] is True
    assert meta["expr_depth"] >= 2
    assert meta["expr_len"] == len("6/(1+1)")


def test_assign_bucket_marks_five_number_division_hard():
    bucket = assign_bucket(numbers=[100, 75, 23, 15, 6], expr="(100+75)/(23-15+6)")

    assert bucket["num_count"] == 5
    assert bucket["complexity"] == "hard"
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
python -m pytest post_train\tests\test_bucketing_sampling.py -v
```

Expected: FAIL because `solver.py` and `bucketing.py` do not exist.

- [ ] **Step 3: Implement solver**

Create `post_train/src/countdown/solver.py`:

```python
from __future__ import annotations

import ast
from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class ExprNode:
    value: Fraction
    expr: str


def solve_countdown(numbers: list[int], target: int) -> str | None:
    nodes = [ExprNode(Fraction(int(value), 1), str(int(value))) for value in numbers]
    target_value = Fraction(int(target), 1)
    failed: set[tuple[Fraction, ...]] = set()

    def search(current: list[ExprNode]) -> ExprNode | None:
        key = tuple(sorted(node.value for node in current))
        if key in failed:
            return None
        if len(current) == 1:
            if current[0].value == target_value:
                return current[0]
            failed.add(key)
            return None

        size = len(current)
        for left_idx in range(size):
            for right_idx in range(left_idx + 1, size):
                left = current[left_idx]
                right = current[right_idx]
                rest = [current[idx] for idx in range(size) if idx not in (left_idx, right_idx)]
                candidates = [
                    ExprNode(left.value + right.value, f"({left.expr}+{right.expr})"),
                    ExprNode(left.value * right.value, f"({left.expr}*{right.expr})"),
                    ExprNode(left.value - right.value, f"({left.expr}-{right.expr})"),
                    ExprNode(right.value - left.value, f"({right.expr}-{left.expr})"),
                ]
                if right.value != 0:
                    candidates.append(ExprNode(left.value / right.value, f"({left.expr}/{right.expr})"))
                if left.value != 0:
                    candidates.append(ExprNode(right.value / left.value, f"({right.expr}/{left.expr})"))

                seen: set[tuple[Fraction, str]] = set()
                for candidate in candidates:
                    signature = (candidate.value, candidate.expr)
                    if signature in seen:
                        continue
                    seen.add(signature)
                    found = search(rest + [candidate])
                    if found is not None:
                        return found

        failed.add(key)
        return None

    result = search(nodes)
    return result.expr if result is not None else None


def expression_metadata(expr: str, num_count: int) -> dict:
    parsed = ast.parse(expr.strip(), mode="eval")
    depth = _depth(parsed.body)
    return {
        "num_count": int(num_count),
        "expr_depth": depth,
        "expr_len": len(expr.strip()),
        "has_division": "/" in expr,
        "has_subtraction": "-" in expr,
    }


def _depth(node: ast.AST) -> int:
    if isinstance(node, ast.Constant):
        return 1
    if isinstance(node, ast.BinOp):
        return 1 + max(_depth(node.left), _depth(node.right))
    return 1
```

- [ ] **Step 4: Implement bucketing**

Create `post_train/src/countdown/bucketing.py`:

```python
from __future__ import annotations

from post_train.src.countdown.solver import expression_metadata


def assign_bucket(numbers: list[int], expr: str) -> dict:
    meta = expression_metadata(expr, num_count=len(numbers))
    score = 0
    if meta["num_count"] >= 4:
        score += 1
    if meta["num_count"] >= 5:
        score += 2
    if meta["has_subtraction"]:
        score += 1
    if meta["has_division"]:
        score += 2
    if meta["expr_depth"] >= 4:
        score += 1
    if meta["expr_len"] >= 18:
        score += 1

    if score <= 1:
        complexity = "easy"
    elif score <= 3:
        complexity = "medium"
    else:
        complexity = "hard"

    return {**meta, "complexity": complexity, "bucket_key": f"{meta['num_count']}_{complexity}"}
```

- [ ] **Step 5: Run tests**

Run:

```powershell
python -m pytest post_train\tests\test_bucketing_sampling.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit if git is available**

Run:

```powershell
git add post_train/src/countdown/solver.py post_train/src/countdown/bucketing.py post_train/tests/test_bucketing_sampling.py
git commit -m "feat: add countdown solver and buckets"
```

Skip if the workspace is not a git repository.

---

### Task 5: Prompt Builders

**Files:**
- Create: `post_train/src/countdown/prompts.py`
- Test: `post_train/tests/test_prompts.py`

- [ ] **Step 1: Write prompt tests**

Create `post_train/tests/test_prompts.py`:

```python
from post_train.src.countdown.prompts import build_dpo_forced_wrong_prompt, build_solution_prompt


def test_solution_prompt_contains_required_rules_without_exact_division():
    prompt = build_solution_prompt([1, 1, 1, 1], 4)

    assert "Using the numbers [1, 1, 1, 1]" in prompt
    assert "Use each number exactly once" in prompt
    assert "<answer>" in prompt
    assert "Division must be exact" not in prompt


def test_forced_wrong_prompt_requests_wrong_answer():
    prompt = build_dpo_forced_wrong_prompt([7, 3, 8, 2], 24, "short <answer> (7-3)*(8-2) </answer>")

    assert "wrong" in prompt.lower()
    assert "complete <answer>" in prompt
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
python -m pytest post_train\tests\test_prompts.py -v
```

Expected: FAIL because `prompts.py` does not exist.

- [ ] **Step 3: Implement prompt builders**

Create `post_train/src/countdown/prompts.py`:

```python
from __future__ import annotations


def build_solution_prompt(numbers: list[int], target: int) -> str:
    return (
        f"Using the numbers {numbers}, create an equation that equals {int(target)}.\n"
        "Use each number exactly once. Only use +, -, *, / and parentheses.\n"
        "Do not use any other numbers. Keep the response concise.\n"
        "Finally return <answer> equation </answer>."
    )


def build_dpo_forced_wrong_prompt(numbers: list[int], target: int, chosen_response: str) -> str:
    return (
        f"Using the numbers {numbers}, create an equation that equals {int(target)}.\n"
        "You must write a plausible but mathematically wrong answer for preference training.\n"
        "Keep a complete <answer> expression </answer> block when possible.\n"
        "Use the same numbers exactly once if possible, but make the expression evaluate to a wrong value.\n\n"
        f"Correct response to imitate in style:\n{chosen_response}\n\n"
        "Now write the rejected response."
    )


def build_chat_messages(prompt: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": prompt}]
```

- [ ] **Step 4: Run tests**

Run:

```powershell
python -m pytest post_train\tests\test_prompts.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit if git is available**

Run:

```powershell
git add post_train/src/countdown/prompts.py post_train/tests/test_prompts.py
git commit -m "feat: add countdown prompt builders"
```

Skip if the workspace is not a git repository.

---

### Task 6: Stratified Sampling

**Files:**
- Create: `post_train/src/countdown/sampling.py`
- Modify: `post_train/tests/test_bucketing_sampling.py`

- [ ] **Step 1: Add sampling tests**

Append to `post_train/tests/test_bucketing_sampling.py`:

```python
from post_train.src.countdown.sampling import stratified_sample


def test_stratified_sample_is_reproducible():
    rows = [
        {"id": "a1", "bucket": {"bucket_key": "3_easy"}},
        {"id": "a2", "bucket": {"bucket_key": "3_easy"}},
        {"id": "b1", "bucket": {"bucket_key": "4_medium"}},
        {"id": "b2", "bucket": {"bucket_key": "4_medium"}},
        {"id": "c1", "bucket": {"bucket_key": "5_hard"}},
        {"id": "c2", "bucket": {"bucket_key": "5_hard"}},
    ]

    first = stratified_sample(rows, size=3, seed=42)
    second = stratified_sample(rows, size=3, seed=42)

    assert first == second
    assert len(first) == 3
    assert {row["bucket"]["bucket_key"] for row in first} == {"3_easy", "4_medium", "5_hard"}
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
python -m pytest post_train\tests\test_bucketing_sampling.py -v
```

Expected: FAIL because `sampling.py` does not exist.

- [ ] **Step 3: Implement sampling**

Create `post_train/src/countdown/sampling.py`:

```python
from __future__ import annotations

import random
from collections import defaultdict
from typing import Any


def stratified_sample(rows: list[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    if size <= 0:
        return []
    if size >= len(rows):
        return list(rows)

    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = row.get("bucket", {}).get("bucket_key")
        if key is None:
            raise ValueError(f"Missing bucket.bucket_key for row {row.get('id')}")
        buckets[str(key)].append(row)

    bucket_keys = sorted(buckets)
    selected: list[dict[str, Any]] = []
    quotas = _balanced_quotas(bucket_keys, buckets, size)
    for key in bucket_keys:
        candidates = list(buckets[key])
        rng.shuffle(candidates)
        selected.extend(candidates[: quotas[key]])

    if len(selected) < size:
        selected_ids = {row["id"] for row in selected}
        leftovers = [row for row in rows if row["id"] not in selected_ids]
        rng.shuffle(leftovers)
        selected.extend(leftovers[: size - len(selected)])

    rng.shuffle(selected)
    return selected[:size]


def _balanced_quotas(
    bucket_keys: list[str],
    buckets: dict[str, list[dict[str, Any]]],
    size: int,
) -> dict[str, int]:
    base = size // len(bucket_keys)
    remainder = size % len(bucket_keys)
    quotas = {key: min(base, len(buckets[key])) for key in bucket_keys}
    for key in bucket_keys:
        if remainder <= 0:
            break
        if quotas[key] < len(buckets[key]):
            quotas[key] += 1
            remainder -= 1
    return quotas
```

- [ ] **Step 4: Run tests**

Run:

```powershell
python -m pytest post_train\tests\test_bucketing_sampling.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit if git is available**

Run:

```powershell
git add post_train/src/countdown/sampling.py post_train/tests/test_bucketing_sampling.py
git commit -m "feat: add stratified sampling"
```

Skip if the workspace is not a git repository.

---

### Task 7: Data Warehouse Builder

**Files:**
- Create: `post_train/scripts/data/build_source.py`
- Modify: `post_train/src/countdown/io.py`
- Test: use script dry-run with `--limit 500`

- [ ] **Step 1: Extend IO with manifest helper**

Append to `post_train/src/countdown/io.py`:

```python
from datetime import datetime, timezone


def write_manifest(path: str | Path, payload: dict[str, Any]) -> None:
    manifest = dict(payload)
    manifest["created_at"] = datetime.now(timezone.utc).isoformat()
    write_json(path, manifest)
```

- [ ] **Step 2: Create data builder script**

Create `post_train/scripts/data/build_source.py`:

```python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from post_train.src.countdown.bucketing import assign_bucket
from post_train.src.countdown.config import load_yaml_config
from post_train.src.countdown.io import write_jsonl, write_manifest
from post_train.src.countdown.prompts import build_solution_prompt
from post_train.src.countdown.sampling import stratified_sample
from post_train.src.countdown.solver import solve_countdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build solver-backed Countdown source data.")
    parser.add_argument("--config", default="post_train/configs/data_build.yaml")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet(cfg["train_input"])
    if args.limit > 0:
        train_df = train_df.head(args.limit)

    source_rows = []
    unsolved_rows = []
    for source_index, row in enumerate(train_df.itertuples(index=False), start=1):
        numbers = [int(value) for value in row.nums]
        target = int(row.target)
        expr = solve_countdown(numbers, target)
        if expr is None:
            unsolved_rows.append({"source_index": source_index, "numbers": numbers, "target": target})
            continue
        bucket = assign_bucket(numbers, expr)
        source_rows.append(
            {
                "id": f"train-{source_index:06d}",
                "source_index": source_index,
                "numbers": numbers,
                "target": target,
                "gold_expr": expr,
                "prompt": build_solution_prompt(numbers, target),
                "bucket": bucket,
            }
        )

    val_rows = stratified_sample(source_rows, size=int(cfg["val_size"]), seed=int(cfg["seed"]))
    val_ids = {row["id"] for row in val_rows}
    train_pool = [row for row in source_rows if row["id"] not in val_ids]
    eval_subset = stratified_sample(val_rows, size=int(cfg["eval_subset_size"]), seed=int(cfg["seed"]) + 1)

    test_rows = []
    with Path(cfg["test_input"]).open("r", encoding="utf-8") as handle:
        raw_test = json.load(handle)
    for row in raw_test:
        numbers = [int(value) for value in row["nums"]]
        target = int(row["target"])
        expr = solve_countdown(numbers, target)
        if expr is None:
            raise ValueError(f"Unsolved test row id={row.get('id')}")
        test_rows.append(
            {
                "id": f"test-{int(row['id']):06d}",
                "source_index": int(row["id"]),
                "numbers": numbers,
                "target": target,
                "gold_expr": expr,
                "prompt": build_solution_prompt(numbers, target),
                "bucket": assign_bucket(numbers, expr),
            }
        )

    write_jsonl(output_dir / "source_all.jsonl", source_rows)
    write_jsonl(output_dir / "train_pool.jsonl", train_pool)
    write_jsonl(output_dir / "val_200.jsonl", val_rows)
    write_jsonl(output_dir / "val_eval_50.jsonl", eval_subset)
    write_jsonl(output_dir / "test_with_solver_answers.jsonl", test_rows)
    write_jsonl(output_dir / "unsolved_train.jsonl", unsolved_rows)
    write_manifest(
        output_dir / "manifest.json",
        {
            "name": "data_build",
            "num_source": len(source_rows),
            "num_train_pool": len(train_pool),
            "num_val": len(val_rows),
            "num_eval_subset": len(eval_subset),
            "num_test": len(test_rows),
            "num_unsolved": len(unsolved_rows),
            "seed": cfg["seed"],
        },
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run dry-run data build**

Run:

```powershell
python post_train\scripts\data\build_source.py --limit 500
```

Expected: `post_train/data/processed/source_all.jsonl`, `val_200.jsonl`, `val_eval_50.jsonl`, and `test_with_solver_answers.jsonl` are created. If sandbox blocks Python parquet access, rerun in the Ubuntu target environment.

- [ ] **Step 4: Inspect manifest**

Run:

```powershell
Get-Content -LiteralPath post_train\data\processed\manifest.json
```

Expected: JSON contains `num_val: 200` and `num_eval_subset: 50`.

- [ ] **Step 5: Commit if git is available**

Run:

```powershell
git add post_train/src/countdown/io.py post_train/scripts/data/build_source.py
git commit -m "feat: build solver-backed countdown datasets"
```

Skip if the workspace is not a git repository.

---

### Task 8: Generation Wrapper

**Files:**
- Create: `post_train/src/countdown/generation.py`

- [ ] **Step 1: Implement generation interfaces**

Create `post_train/src/countdown/generation.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from post_train.src.countdown.prompts import build_chat_messages


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int
    temperature: float
    top_p: float
    enable_thinking: bool = False


class TextGenerator(Protocol):
    def generate(self, prompts: list[str], config: GenerationConfig) -> list[str]:
        ...


def apply_chat_template(tokenizer, prompt: str, enable_thinking: bool) -> str:
    messages = build_chat_messages(prompt)
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=enable_thinking, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


class VLLMGenerator:
    def __init__(self, model_path: str, tensor_parallel_size: int = 1):
        from vllm import LLM

        self.llm = LLM(model=model_path, tensor_parallel_size=tensor_parallel_size, trust_remote_code=True)

    def generate(self, prompts: list[str], config: GenerationConfig) -> list[str]:
        from vllm import SamplingParams

        sampling = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_new_tokens,
        )
        outputs = self.llm.generate(prompts, sampling)
        return [item.outputs[0].text for item in outputs]
```

- [ ] **Step 2: Syntax check**

Run:

```powershell
python -m py_compile post_train\src\countdown\generation.py
```

Expected: no output and exit code 0.

- [ ] **Step 3: Commit if git is available**

Run:

```powershell
git add post_train/src/countdown/generation.py
git commit -m "feat: add generation wrapper"
```

Skip if the workspace is not a git repository.

---

### Task 9: Teacher Accepted Pool Builder

**Files:**
- Create: `post_train/scripts/data/build_teacher_pool.py`

- [ ] **Step 1: Create teacher pool script**

Create `post_train/scripts/data/build_teacher_pool.py`:

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from post_train.src.countdown.config import load_yaml_config
from post_train.src.countdown.generation import GenerationConfig, VLLMGenerator
from post_train.src.countdown.io import read_jsonl, write_jsonl, write_manifest
from post_train.src.countdown.validation import extract_answer_text, validate_countdown_response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build teacher accepted pool.")
    parser.add_argument("--config", default="post_train/configs/teacher_rollout.yaml")
    parser.add_argument("--input", default="post_train/data/processed/train_pool.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    output_dir = Path("post_train/data/teacher_rollouts")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_rows = read_jsonl(args.input)
    accepted = []
    rejected = []
    processed_ids = set()
    accepted_path = output_dir / "teacher_accepted_20k.jsonl"
    rejected_path = output_dir / "teacher_rejected.jsonl"

    if accepted_path.exists():
        accepted = read_jsonl(accepted_path)
        processed_ids.update(row["id"] for row in accepted)
    if rejected_path.exists():
        rejected = read_jsonl(rejected_path)
        processed_ids.update(row["id"] for row in rejected)

    generator = VLLMGenerator(model_path=cfg["model_path"])
    gen_cfg = GenerationConfig(
        max_new_tokens=int(cfg["max_new_tokens"]),
        temperature=float(cfg["temperature"]),
        top_p=float(cfg["top_p"]),
        enable_thinking=bool(cfg["enable_thinking"]),
    )

    target = int(cfg["stop_after_accepted"])
    batch_size = int(cfg["batch_size"])
    remaining = [row for row in source_rows if row["id"] not in processed_ids]
    for start in range(0, len(remaining), batch_size):
        if len(accepted) >= target:
            break
        batch = remaining[start : start + batch_size]
        outputs = generator.generate([row["prompt"] for row in batch], gen_cfg)
        for row, text in zip(batch, outputs):
            result = validate_countdown_response(text, row["numbers"], int(row["target"]))
            payload = {
                **row,
                "response": text.strip(),
                "teacher_expr": extract_answer_text(text),
                "validation": {
                    "ok": result.ok,
                    "error": result.error,
                    "value": result.value,
                },
            }
            if result.ok:
                accepted.append(payload)
            else:
                rejected.append(payload)
        write_jsonl(accepted_path, accepted[:target])
        write_jsonl(rejected_path, rejected)

    write_manifest(
        output_dir / "manifest.json",
        {
            "name": "teacher_accepted_pool",
            "model": cfg["model_path"],
            "num_accepted": len(accepted[:target]),
            "num_rejected": len(rejected),
            "max_new_tokens": cfg["max_new_tokens"],
            "enable_thinking": cfg["enable_thinking"],
        },
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Syntax check**

Run:

```powershell
python -m py_compile post_train\scripts\data\build_teacher_pool.py
```

Expected: no syntax errors. Full execution requires vLLM and Qwen3-8B on Ubuntu.

- [ ] **Step 3: Commit if git is available**

Run:

```powershell
git add post_train/scripts/data/build_teacher_pool.py
git commit -m "feat: add teacher accepted pool builder"
```

Skip if the workspace is not a git repository.

---

### Task 10: SFT And GRPO Split Builder

**Files:**
- Create: `post_train/scripts/data/build_sft_splits.py`

- [ ] **Step 1: Create split builder**

Create `post_train/scripts/data/build_sft_splits.py`:

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from post_train.src.countdown.config import load_yaml_config
from post_train.src.countdown.io import read_jsonl, write_jsonl, write_manifest
from post_train.src.countdown.sampling import stratified_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SFT and GRPO train splits.")
    parser.add_argument("--config", default="post_train/configs/data_build.yaml")
    parser.add_argument("--accepted", default="post_train/data/teacher_rollouts/teacher_accepted_20k.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    accepted = read_jsonl(args.accepted)

    sft_rows = stratified_sample(accepted, size=int(cfg["sft_train_target"]), seed=int(cfg["seed"]) + 10)
    grpo_rows = stratified_sample(accepted, size=int(cfg["grpo_train_target"]), seed=int(cfg["seed"]) + 20)

    sft_dir = Path("post_train/data/sft")
    grpo_dir = Path("post_train/data/grpo")
    sft_dir.mkdir(parents=True, exist_ok=True)
    grpo_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(sft_dir / "sft_train_8k.jsonl", sft_rows)
    write_jsonl(grpo_dir / "grpo_train_4k.jsonl", grpo_rows)
    write_manifest(
        sft_dir / "manifest.json",
        {
            "name": "sft_and_grpo_splits",
            "num_accepted_pool": len(accepted),
            "num_sft": len(sft_rows),
            "num_grpo": len(grpo_rows),
            "seed": cfg["seed"],
        },
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Syntax check**

Run:

```powershell
python -m py_compile post_train\scripts\data\build_sft_splits.py
```

Expected: no syntax errors.

- [ ] **Step 3: Commit if git is available**

Run:

```powershell
git add post_train/scripts/data/build_sft_splits.py
git commit -m "feat: add sft and grpo split builder"
```

Skip if the workspace is not a git repository.

---

### Task 11: Common Evaluator

**Files:**
- Create: `post_train/src/countdown/eval.py`
- Create: `post_train/tests/test_eval_metrics.py`
- Create: `post_train/scripts/eval/evaluate_model.py`

- [ ] **Step 1: Write evaluator tests**

Create `post_train/tests/test_eval_metrics.py`:

```python
from post_train.src.countdown.eval import aggregate_eval_rows, score_generation


def test_score_generation_correct():
    row = {"numbers": [7, 3, 8, 2], "target": 24}

    scored = score_generation(row, "short\n<answer> (7-3)*(8-2) </answer>", generated_tokens=12, truncated=False)

    assert scored["format_ok"] is True
    assert scored["correct"] is True
    assert scored["valid"] is True


def test_aggregate_eval_rows():
    rows = [
        {"format_ok": True, "valid": True, "correct": True, "generated_tokens": 10, "truncated": False},
        {"format_ok": False, "valid": False, "correct": False, "generated_tokens": 20, "truncated": True},
    ]

    metrics = aggregate_eval_rows(rows)

    assert metrics["accuracy"] == 0.5
    assert metrics["format_rate"] == 0.5
    assert metrics["truncated_count"] == 1
```

- [ ] **Step 2: Implement evaluator**

Create `post_train/src/countdown/eval.py`:

```python
from __future__ import annotations

from typing import Any

from post_train.src.countdown.validation import extract_answer_text, validate_countdown_response


def score_generation(
    row: dict[str, Any],
    raw_generation: str,
    generated_tokens: int,
    truncated: bool,
) -> dict[str, Any]:
    result = validate_countdown_response(raw_generation, row["numbers"], int(row["target"]))
    return {
        "id": row.get("id"),
        "prompt": row.get("prompt"),
        "raw_generation": raw_generation,
        "extracted_expr": extract_answer_text(raw_generation),
        "format_ok": extract_answer_text(raw_generation) is not None,
        "valid": result.ok,
        "correct": result.ok,
        "error": result.error,
        "generated_tokens": int(generated_tokens),
        "truncated": bool(truncated),
    }


def aggregate_eval_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    if total == 0:
        return {
            "accuracy": 0.0,
            "format_rate": 0.0,
            "valid_expression_rate": 0.0,
            "avg_generated_tokens": 0.0,
            "max_generated_tokens": 0,
            "truncated_count": 0,
        }
    return {
        "accuracy": sum(1 for row in rows if row["correct"]) / total,
        "format_rate": sum(1 for row in rows if row["format_ok"]) / total,
        "valid_expression_rate": sum(1 for row in rows if row["valid"]) / total,
        "avg_generated_tokens": sum(int(row["generated_tokens"]) for row in rows) / total,
        "max_generated_tokens": max(int(row["generated_tokens"]) for row in rows),
        "truncated_count": sum(1 for row in rows if row["truncated"]),
    }
```

- [ ] **Step 3: Run evaluator tests**

Run:

```powershell
python -m pytest post_train\tests\test_eval_metrics.py -v
```

Expected: PASS.

- [ ] **Step 4: Add standalone evaluator script**

Create `post_train/scripts/eval/evaluate_model.py` as a Transformers-based evaluator that loads `eval.yaml`, runs generation on `val_eval_50.jsonl`, scores rows with `score_generation`, writes `eval_samples.jsonl`, and writes `eval_metrics.json`. Use `post_train/src/countdown/generation.py` for chat template construction.

The script must expose:

```powershell
python post_train\scripts\eval\evaluate_model.py --model-path post_train\outputs\sft\full\final --config post_train\configs\eval.yaml --output-dir post_train\data\eval\sft_full
```

Expected output files:

- `post_train/data/eval/sft_full/eval_samples.jsonl`
- `post_train/data/eval/sft_full/eval_metrics.json`

- [ ] **Step 5: Commit if git is available**

Run:

```powershell
git add post_train/src/countdown/eval.py post_train/tests/test_eval_metrics.py post_train/scripts/eval/evaluate_model.py
git commit -m "feat: add common countdown evaluator"
```

Skip if the workspace is not a git repository.

---

### Task 12: Full-Parameter SFT Script

**Files:**
- Create: `post_train/scripts/sft/train_full.py`

- [ ] **Step 1: Implement full SFT training entrypoint**

Create a script with these responsibilities:

- Load `post_train/configs/sft_full.yaml`.
- Read `post_train/data/sft/sft_train_8k.jsonl`.
- Format each sample as chat:

```python
[
    {"role": "user", "content": row["prompt"]},
    {"role": "assistant", "content": row["response"]},
]
```

- Tokenize to `max_seq_len=256`.
- Mask prompt tokens with `-100`.
- Train Qwen3-0.6B full parameters with bf16 and gradient checkpointing.
- Use cosine scheduler with warmup ratio 0.03.
- Every 100 steps, call the common evaluator on the fixed `val_eval_50.jsonl`.
- Save checkpoints every 100 steps and final model at `post_train/outputs/sft/full/final`.

Include these functions in the script:

```python
def encode_prompt_response(tokenizer, prompt: str, response: str, max_seq_len: int) -> dict | None:
    prompt_text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
        tokenize=False,
        add_generation_prompt=False,
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if len(full_ids) > max_seq_len:
        full_ids = full_ids[:max_seq_len]
    labels = list(full_ids)
    prompt_len = min(len(prompt_ids), len(labels))
    labels[:prompt_len] = [-100] * prompt_len
    if all(label == -100 for label in labels):
        return None
    return {"input_ids": full_ids, "labels": labels, "attention_mask": [1] * len(full_ids)}
```

- [ ] **Step 2: Syntax check**

Run:

```powershell
python -m py_compile post_train\scripts\sft\train_full.py
```

Expected: no syntax errors.

- [ ] **Step 3: Ubuntu smoke run**

Run in the Ubuntu environment after data exists:

```bash
python post_train/scripts/sft/train_full.py --config post_train/configs/sft_full.yaml --max-steps 2
```

Expected: two training steps complete, one output directory is created, and no OOM occurs.

- [ ] **Step 4: Commit if git is available**

Run:

```powershell
git add post_train/scripts/sft/train_full.py
git commit -m "feat: add full parameter sft training"
```

Skip if the workspace is not a git repository.

---

### Task 13: LoRA SFT Script

**Files:**
- Create: `post_train/scripts/sft/train_lora.py`

- [ ] **Step 1: Implement LoRA SFT**

Implement the same dataset formatting as Task 12, but load PEFT LoRA adapters with:

```python
LoraConfig(
    r=cfg["lora_r"],
    lora_alpha=cfg["lora_alpha"],
    lora_dropout=cfg["lora_dropout"],
    target_modules=resolved_target_modules,
    bias="none",
    task_type="CAUSAL_LM",
)
```

If `lora_target_modules: auto`, detect linear projection suffixes in this order:

```python
["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
```

Save adapters under `post_train/outputs/sft/lora/final`.

- [ ] **Step 2: Syntax check**

Run:

```powershell
python -m py_compile post_train\scripts\sft\train_lora.py
```

Expected: no syntax errors.

- [ ] **Step 3: Ubuntu smoke run**

Run:

```bash
python post_train/scripts/sft/train_lora.py --config post_train/configs/sft_lora.yaml --max-steps 2
```

Expected: LoRA trainable parameter count is printed and final adapter directory is created.

- [ ] **Step 4: Commit if git is available**

Run:

```powershell
git add post_train/scripts/sft/train_lora.py
git commit -m "feat: add lora sft training"
```

Skip if the workspace is not a git repository.

---

### Task 14: RFT Data Builder

**Files:**
- Create: `post_train/scripts/sft/build_rft_data.py`

- [ ] **Step 1: Implement RFT data generation**

Create a vLLM-based script that:

- Loads `rft.yaml`.
- Reads SFT prompts from `post_train/data/sft/sft_train_8k.jsonl`.
- Generates `num_samples_per_prompt` responses per prompt.
- Validates each response with `validate_countdown_response`.
- Writes accepted rows to `post_train/data/sft/rft_accepted.jsonl`.

Each accepted row must contain:

```json
{
  "id": "source-id-rft-0",
  "prompt": "original prompt",
  "response": "accepted sampled response",
  "numbers": [1, 1, 1, 1],
  "target": 4,
  "source": "rft"
}
```

- [ ] **Step 2: Syntax check**

Run:

```powershell
python -m py_compile post_train\scripts\sft\build_rft_data.py
```

Expected: no syntax errors.

- [ ] **Step 3: Ubuntu smoke run**

Run:

```bash
python post_train/scripts/sft/build_rft_data.py --config post_train/configs/rft.yaml --limit 16
```

Expected: script writes accepted and rejected summary files without crashing.

- [ ] **Step 4: Commit if git is available**

Run:

```powershell
git add post_train/scripts/sft/build_rft_data.py
git commit -m "feat: add rft data builder"
```

Skip if the workspace is not a git repository.

---

### Task 15: DPO Data Builder

**Files:**
- Create: `post_train/scripts/dpo/build_dpo_data.py`

- [ ] **Step 1: Implement DPO rejected classifier**

Inside `build_dpo_data.py`, define:

```python
def classify_rejected(text: str, numbers: list[int], target: int, truncated: bool) -> str:
    if truncated:
        return "truncated"
    expr = extract_answer_text(text)
    if expr is None:
        return "missing_answer_tag"
    result = validate_countdown_expression(expr, numbers, target)
    if result.ok:
        return "unexpected_correct"
    if result.error == "number_mismatch":
        return "number_mismatch"
    if result.error == "wrong_value":
        return "wrong_value"
    return "invalid_expression"
```

- [ ] **Step 2: Implement pair generation**

The script must:

- Load Qwen3-8B from `dpo_data.yaml`.
- Generate half rejected candidates using `build_dpo_forced_wrong_prompt`.
- Generate half rejected candidates using the original solution prompt at high temperature.
- Drop `unexpected_correct`.
- Prefer `wrong_value`.
- Keep malformed categories under `malformed_cap_fraction`.
- Write `post_train/data/dpo/dpo_train.jsonl`.
- Write manifest category counts.

- [ ] **Step 3: Syntax check**

Run:

```powershell
python -m py_compile post_train\scripts\dpo\build_dpo_data.py
```

Expected: no syntax errors.

- [ ] **Step 4: Ubuntu smoke run**

Run:

```bash
python post_train/scripts/dpo/build_dpo_data.py --config post_train/configs/dpo_data.yaml --limit 64
```

Expected: DPO pairs are generated and manifest shows category counts.

- [ ] **Step 5: Commit if git is available**

Run:

```powershell
git add post_train/scripts/dpo/build_dpo_data.py
git commit -m "feat: add dpo data builder"
```

Skip if the workspace is not a git repository.

---

### Task 16: DPO Training Script

**Files:**
- Create: `post_train/scripts/dpo/train_dpo.py`

- [ ] **Step 1: Implement DPO training with TRL**

Use TRL `DPOTrainer` if available in the target environment. The script must:

- Load `dpo_train.yaml`.
- Load model from `post_train/outputs/sft/full/final`.
- Use DPO beta `0.05`.
- Train on `post_train/data/dpo/dpo_train.jsonl`.
- Evaluate every 100 steps on fixed `val_eval_50.jsonl` with the common evaluator.
- Save final output under `post_train/outputs/dpo/final`.

Prepare records with keys:

```python
{"prompt": row["prompt"], "chosen": row["chosen"], "rejected": row["rejected"]}
```

- [ ] **Step 2: Syntax check**

Run:

```powershell
python -m py_compile post_train\scripts\dpo\train_dpo.py
```

Expected: no syntax errors.

- [ ] **Step 3: Ubuntu smoke run**

Run:

```bash
python post_train/scripts/dpo/train_dpo.py --config post_train/configs/dpo_train.yaml --max-steps 2
```

Expected: two DPO steps complete and output directory is created.

- [ ] **Step 4: Commit if git is available**

Run:

```powershell
git add post_train/scripts/dpo/train_dpo.py
git commit -m "feat: add dpo training"
```

Skip if the workspace is not a git repository.

---

### Task 17: GRPO Training Script

**Files:**
- Create: `post_train/scripts/grpo/train_grpo.py`

- [ ] **Step 1: Implement GRPO metric helpers**

Create helper functions in `train_grpo.py`:

```python
def compute_rewards(rows: list[dict], completions: list[str], format_reward: float, answer_reward: float) -> list[dict]:
    reward_rows = []
    for row, completion in zip(rows, completions):
        result = validate_countdown_response(completion, row["numbers"], int(row["target"]))
        has_format = extract_answer_text(completion) is not None
        reward = (format_reward if has_format else 0.0) + (answer_reward if result.ok else 0.0)
        reward_rows.append({"reward": reward, "format_ok": has_format, "correct": result.ok})
    return reward_rows
```

```python
def grpo_metric_summary(rewards: list[float], group_size: int) -> dict:
    import statistics

    reward_std = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
    group_stds = []
    zero_std_groups = 0
    for start in range(0, len(rewards), group_size):
        group = rewards[start : start + group_size]
        std = statistics.pstdev(group) if len(group) > 1 else 0.0
        group_stds.append(std)
        if std == 0.0:
            zero_std_groups += 1
    return {
        "reward_std": reward_std,
        "group_reward_std": sum(group_stds) / max(1, len(group_stds)),
        "frac_reward_zero_std": zero_std_groups / max(1, len(group_stds)),
    }
```

- [ ] **Step 2: Implement rollout and training structure**

The script must:

- Load SFT-trained Qwen3-0.6B from `post_train/outputs/sft/full/final`.
- Start vLLM rollout model from the same path.
- Maintain separate Transformers training model.
- Roll out `batch_size * group_size` completions.
- Compute group-relative advantages.
- Run `policy_updates_per_rollout=2`.
- Use cosine scheduler.
- Set KL coefficient to 0.0.
- Save and sync weights every 20 steps.
- Log `loss`, `mean_reward`, `reward_std`, `group_reward_std`, `frac_reward_zero_std`, `accuracy`, `format_rate`, `approx_kl`, `entropy`, `avg_gen_tokens`, `max_gen_tokens`, `truncated_count`, `rollout_count`, and `learning_rate`.

- [ ] **Step 3: Syntax check**

Run:

```powershell
python -m py_compile post_train\scripts\grpo\train_grpo.py
```

Expected: no syntax errors.

- [ ] **Step 4: Ubuntu smoke run**

Run:

```bash
python post_train/scripts/grpo/train_grpo.py --config post_train/configs/grpo.yaml --max-steps 2
```

Expected: two rollout/update iterations complete and metrics JSONL contains the required fields.

- [ ] **Step 5: Commit if git is available**

Run:

```powershell
git add post_train/scripts/grpo/train_grpo.py
git commit -m "feat: add grpo training"
```

Skip if the workspace is not a git repository.

---

### Task 18: README Workflow

**Files:**
- Create: `post_train/README.md`

- [ ] **Step 1: Write workflow README**

Create `post_train/README.md` with these command sections:

```markdown
# Countdown Post-Training

## 1. Build Solver-Backed Data

```bash
python post_train/scripts/data/build_source.py --config post_train/configs/data_build.yaml
```

## 2. Build Teacher Accepted Pool

```bash
python post_train/scripts/data/build_teacher_pool.py --config post_train/configs/teacher_rollout.yaml
```

## 3. Build SFT And GRPO Splits

```bash
python post_train/scripts/data/build_sft_splits.py --config post_train/configs/data_build.yaml
```

## 4. Train SFT

```bash
python post_train/scripts/sft/train_full.py --config post_train/configs/sft_full.yaml
python post_train/scripts/sft/train_lora.py --config post_train/configs/sft_lora.yaml
```

## 5. Build RFT Data

```bash
python post_train/scripts/sft/build_rft_data.py --config post_train/configs/rft.yaml
```

## 6. Build And Train DPO

```bash
python post_train/scripts/dpo/build_dpo_data.py --config post_train/configs/dpo_data.yaml
python post_train/scripts/dpo/train_dpo.py --config post_train/configs/dpo_train.yaml
```

## 7. Train GRPO

```bash
python post_train/scripts/grpo/train_grpo.py --config post_train/configs/grpo.yaml
```

## 8. Evaluate

```bash
python post_train/scripts/eval/evaluate_model.py --config post_train/configs/eval.yaml --model-path post_train/outputs/sft/full/final --output-dir post_train/data/eval/sft_full
```
```

- [ ] **Step 2: Verify README renders as markdown**

Run:

```powershell
Get-Content -LiteralPath post_train\README.md
```

Expected: command sections are visible and fenced code blocks are balanced.

- [ ] **Step 3: Commit if git is available**

Run:

```powershell
git add post_train/README.md
git commit -m "docs: add countdown workflow"
```

Skip if the workspace is not a git repository.

---

## Final Verification

- [ ] **Step 1: Run all pure Python tests**

Run:

```powershell
python -m pytest post_train\tests -v
```

Expected: PASS for config, validation, bucketing, sampling, prompts, and eval metric tests.

- [ ] **Step 2: Run syntax checks for scripts**

Run:

```powershell
python -m py_compile post_train\scripts\data\build_source.py post_train\scripts\data\build_teacher_pool.py post_train\scripts\data\build_sft_splits.py post_train\scripts\sft\train_full.py post_train\scripts\sft\train_lora.py post_train\scripts\sft\build_rft_data.py post_train\scripts\dpo\build_dpo_data.py post_train\scripts\dpo\train_dpo.py post_train\scripts\grpo\train_grpo.py post_train\scripts\eval\evaluate_model.py
```

Expected: no syntax errors.

- [ ] **Step 3: Run limited source build**

Run:

```powershell
python post_train\scripts\data\build_source.py --config post_train\configs\data_build.yaml --limit 500
```

Expected: source, train pool, validation, eval subset, test, unsolved, and manifest files are produced under `post_train/data/processed`.

- [ ] **Step 4: Record unavailable GPU checks**

If not in the Ubuntu GPU environment, record that vLLM, SFT, DPO, and GRPO smoke runs were not executed locally. Run them later in the target environment with the smoke commands in Tasks 9, 12, 13, 14, 15, 16, and 17.

## Plan Self-Review

Spec coverage:

- Data warehouse, validation split, test solver answers, and buckets are covered by Tasks 2, 4, 6, and 7.
- Prompt and generation centralization are covered by Tasks 5 and 8.
- Teacher accepted pool is covered by Task 9.
- SFT, LoRA SFT, and RFT are covered by Tasks 12, 13, and 14.
- DPO data construction and training are covered by Tasks 15 and 16.
- GRPO rollout/training and required metrics are covered by Task 17.
- Fixed 50-example eval subset and common metrics are covered by Tasks 7 and 11.
- Per-stage config files are covered by Task 1.

Placeholder scan:

- The plan uses no `TBD` markers.
- GPU-dependent training tasks specify smoke-run commands and expected results.

Type consistency:

- Dataset rows consistently use `id`, `prompt`, `response`, `numbers`, `target`, `gold_expr`, and `bucket`.
- Validation functions consistently return `ValidationResult` with `ok`, `value`, `used_numbers`, `expression`, and `error`.
