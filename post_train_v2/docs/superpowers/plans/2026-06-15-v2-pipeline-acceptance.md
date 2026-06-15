# V2 Pipeline and Acceptance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect all V2 stages into a resumable manifest-driven pipeline, execute the final evaluation matrix, and provide complete local/remote verification and operating documentation.

**Architecture:** A small orchestrator evaluates artifact manifests and launches existing stage CLIs as subprocesses; it does not duplicate stage logic. Stage completion is based on validated manifests and file hashes, not file existence. Final evaluation consumes published best/final artifacts through the common evaluator.

**Tech Stack:** Python subprocess orchestration, Manifest V2, YAML, Transformers evaluation, pytest.

---

## File Map

Create:

- `post_train_v2/src/artifacts/lineage.py`
- `post_train_v2/src/pipeline/model.py`
- `post_train_v2/src/pipeline/runner.py`
- `post_train_v2/scripts/pipeline/run_pipeline.py`
- `post_train_v2/scripts/eval/evaluate_matrix.py`
- `post_train_v2/configs/pipeline/smoke.yaml`
- `post_train_v2/configs/pipeline/production.yaml`
- `post_train_v2/configs/eval/final_matrix.yaml`
- integration and GPU test entrypoints.
- final runbooks and directory README files.

## Task 1: Artifact Lineage Validation

**Files:**

- Create: `post_train_v2/src/artifacts/lineage.py`
- Create: `post_train_v2/tests/integration/test_artifact_lineage.py`

- [ ] **Step 1: Write failing lineage tests**

Assert a child artifact validates all parent IDs/hashes, configuration hash,
schema version, and output hashes. Changed data must mark the stage stale.

- [ ] **Step 2: Implement**

```python
@dataclass(frozen=True)
class ArtifactStatus:
    state: Literal["missing", "complete", "stale", "failed"]
    reason: str
    manifest_path: Path | None
```

Do not accept a partial Teacher manifest as a complete parent.

- [ ] **Step 3: Verify and commit**

```bash
python -m pytest -q post_train_v2/tests/integration/test_artifact_lineage.py
git add post_train_v2/src/artifacts/lineage.py post_train_v2/tests/integration/test_artifact_lineage.py
git commit -m "feat: validate v2 artifact lineage"
```

## Task 2: Pipeline DAG Model

**Files:**

- Create: `post_train_v2/src/pipeline/{__init__.py,model.py,README.md}`
- Create: `post_train_v2/tests/integration/test_pipeline_model.py`

- [ ] **Step 1: Write failing DAG tests**

Freeze this order:

```text
build_source
validation_split
teacher_pool
accepted_splits
full_sft
lora_sft
rft_data
rft_train
dpo_data
dpo_train
grpo_convert
grpo_train
grpo_export
final_eval
```

Assert dependency closure, cycle rejection, `--from-stage`, and
`--through-stage` selection.

- [ ] **Step 2: Implement**

```python
@dataclass(frozen=True)
class StageSpec:
    name: str
    command: tuple[str, ...]
    dependencies: tuple[str, ...]
    manifest_path: Path
    resources: Literal["cpu", "gpu2_vllm", "gpu2_ddp", "gpu2_verl"]
```

- [ ] **Step 3: Verify and commit**

```bash
python -m pytest -q post_train_v2/tests/integration/test_pipeline_model.py
git add post_train_v2/src/pipeline post_train_v2/tests/integration/test_pipeline_model.py
git commit -m "feat: define v2 pipeline dag"
```

## Task 3: Resumable Pipeline Runner

**Files:**

- Create: `post_train_v2/src/pipeline/runner.py`
- Create: `post_train_v2/scripts/pipeline/{run_pipeline.py,README.md}`
- Create: `post_train_v2/configs/pipeline/smoke.yaml`
- Create: `post_train_v2/configs/pipeline/production.yaml`
- Create: `post_train_v2/tests/integration/test_pipeline_runner.py`

- [ ] **Step 1: Write failing runner tests**

Assert complete stages skip, stale stages stop with a diagnostic unless
`--rebuild-stage` names them, failed subprocesses stop downstream execution,
and `--dry-run` loads no ML libraries.

- [ ] **Step 2: Implement**

The CLI supports:

```text
--config
--from-stage
--through-stage
--rebuild-stage (repeatable)
--dry-run
```

Write `pipeline_events.jsonl` atomically per event with command, start/end
time, exit code, input manifest hashes, and output manifest hash.

- [ ] **Step 3: Verify and commit**

```bash
python -m pytest -q post_train_v2/tests/integration/test_pipeline_runner.py
git add post_train_v2/src/pipeline/runner.py post_train_v2/scripts/pipeline post_train_v2/configs/pipeline post_train_v2/tests/integration/test_pipeline_runner.py
git commit -m "feat: add resumable v2 pipeline runner"
```

## Task 4: Final Evaluation Matrix

**Files:**

- Create: `post_train_v2/scripts/eval/evaluate_matrix.py`
- Create: `post_train_v2/configs/eval/final_matrix.yaml`
- Create: `post_train_v2/tests/evaluation/test_evaluation_matrix.py`

- [ ] **Step 1: Write failing matrix tests**

Assert configured models include base, Full SFT best, LoRA best adapter, RFT
best, DPO best, GRPO best, and optional Teacher. Each runs on `val_200` and
solved test; final artifacts receive only fixed-50 direct-load evaluation by
default.

- [ ] **Step 2: Implement**

Write per-model/per-dataset sample and metric files plus:

```json
{
  "models": {},
  "datasets": {},
  "results": [],
  "ranking": {
    "primary": "test_accuracy",
    "secondary": "test_format_rate"
  }
}
```

Do not suppress model-load or evaluation failures; record the failed entry and
exit nonzero after processing the remaining independent models.

- [ ] **Step 3: Verify and commit**

```bash
python -m pytest -q post_train_v2/tests/evaluation/test_evaluation_matrix.py
git add post_train_v2/scripts/eval/evaluate_matrix.py post_train_v2/configs/eval/final_matrix.yaml post_train_v2/tests/evaluation/test_evaluation_matrix.py
git commit -m "feat: add final model evaluation matrix"
```

## Task 5: Recovery Integration Tests

**Files:**

- Create: `post_train_v2/tests/integration/test_recovery_workflows.py`

- [ ] **Step 1: Implement subprocess fakes and recovery cases**

Cover:

- interrupted Teacher resumes from committed source position;
- Trainer stage forwards its latest complete checkpoint;
- GRPO resumes from native checkpoint;
- post-training GRPO export reruns without training;
- changed input/config refuses resume;
- atomic temporary files are not accepted as complete artifacts.

- [ ] **Step 2: Run**

```bash
python -m pytest -q post_train_v2/tests/integration/test_recovery_workflows.py
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add post_train_v2/tests/integration/test_recovery_workflows.py
git commit -m "test: cover v2 recovery workflows"
```

## Task 6: GPU Smoke Entry Points

**Files:**

- Create: `post_train_v2/tests/gpu/README.md`
- Create: `post_train_v2/scripts/env/smoke_v2_training.py`
- Create: `post_train_v2/tests/gpu/test_smoke_entrypoints.py`

- [ ] **Step 1: Write CPU command-construction tests**

Assert smoke commands use two visible GPUs, `torchrun` for Trainer/TRL stages,
plain Python for dual vLLM generation, and the V2 verl launcher for GRPO.

- [ ] **Step 2: Implement orchestrated smoke**

Support:

```bash
python post_train_v2/scripts/env/smoke_v2_training.py \
  --through-stage grpo_export \
  --work-dir /tmp/post_train_v2_smoke
```

The script uses fixture-sized data and one/two-step configs; it never points
to production output directories.

- [ ] **Step 3: Verify and commit**

```bash
python -m pytest -q post_train_v2/tests/gpu/test_smoke_entrypoints.py
git add post_train_v2/scripts/env/smoke_v2_training.py post_train_v2/tests/gpu
git commit -m "feat: add v2 gpu smoke orchestrator"
```

## Task 7: Documentation Completion

**Files:**

- Create: `post_train_v2/docs/runbooks/full_pipeline.md`
- Create: `post_train_v2/docs/runbooks/recovery.md`
- Create: `post_train_v2/docs/runbooks/final_evaluation.md`
- Modify: `post_train_v2/README.md`
- Add concise README files to every functional directory lacking one.

- [ ] **Step 1: Document production prerequisites**

Include environment activation, Level 1 gates, model locations, raw dataset
locations, W&B login/config, disk-space check, and output isolation.

- [ ] **Step 2: Document exact full workflow**

Include one command per stage, resume commands, expected manifests, best/final
artifact locations, and test commands.

- [ ] **Step 3: Verify README coverage**

```bash
python -m pytest -q post_train_v2/tests/env/test_repository_hygiene.py
```

Expected: PASS with every functional directory documented.

- [ ] **Step 4: Commit**

```bash
git add post_train_v2/docs/runbooks post_train_v2/README.md post_train_v2
git commit -m "docs: complete v2 operating runbooks"
```

## Task 8: Final Acceptance Gate

- [ ] **Step 1: Run the complete local suite**

```bash
python -m pytest -q post_train_v2/tests
git diff --check
```

Expected: PASS.

- [ ] **Step 2: Validate dry-run DAG**

```bash
python post_train_v2/scripts/pipeline/run_pipeline.py \
  --config post_train_v2/configs/pipeline/smoke.yaml \
  --dry-run
```

Expected: all 14 stages print in dependency order with resolved commands and
no model imports.

- [ ] **Step 3: Run remote GPU smoke**

```bash
CUDA_VISIBLE_DEVICES=0,1 python post_train_v2/scripts/env/smoke_v2_training.py \
  --through-stage grpo_export \
  --work-dir /tmp/post_train_v2_smoke
```

Expected: all stage smoke manifests are complete and model artifacts pass
direct-load checks.

- [ ] **Step 4: Run small end-to-end acceptance**

```bash
CUDA_VISIBLE_DEVICES=0,1 python post_train_v2/scripts/pipeline/run_pipeline.py \
  --config post_train_v2/configs/pipeline/smoke.yaml \
  --through-stage final_eval
```

Expected: final evaluation summary, complete lineage, isolated outputs, and
no stage writes below `post_train/`.

- [ ] **Step 5: Commit acceptance evidence**

```bash
git add post_train_v2/docs post_train_v2/README.md
git commit -m "docs: record v2 acceptance workflow"
```

