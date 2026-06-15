# Post-Train V2 Core Development Master Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the complete `post_train_v2` Countdown pipeline from normalized raw data through two-GPU SFT, RFT, DPO, verl GRPO, recovery, and final evaluation.

**Architecture:** Development is split into five sequential plans with independently testable outputs. V2 owns shared Countdown, artifact, data, evaluation, tracking, and distributed utilities; Transformers/TRL stages use two-rank `torchrun` DDP, while GRPO delegates actor rollout and optimization to stock verl 0.6.0.

**Tech Stack:** Python 3.11.15, PyTorch 2.7.0 cu128, Transformers 4.53.2, TRL 0.19.1, PEFT 0.15.2, vLLM 0.9.1, verl 0.6.0, FSDP2, Flash Attention 2.7.4.post1, Ray 2.48.0, W&B 0.21.4, pytest 8.3.5.

---

## Authoritative Inputs

- Design: `post_train_v2/docs/superpowers/specs/2026-06-15-post-train-v2-core-development-design.md`
- Runtime lock: `post_train_v2/pyproject.toml`
- Environment gate: `post_train_v2/docs/environment_setup.md`
- Existing Teacher implementation:
  - `post_train_v2/src/generation/parallel_vllm.py`
  - `post_train_v2/src/generation/teacher_state.py`
  - `post_train_v2/scripts/generation/build_teacher_pool.py`

## Phase Plans

1. `2026-06-15-v2-foundations-data-evaluation.md`
   - V2-owned Countdown core.
   - Config and path contracts.
   - Manifest V2 and atomic artifacts.
   - Raw source normalization, fixed `val_200`, fixed `eval_50`, solved test.
   - SFT 8k and GRPO 4k stratified splits.
   - Common evaluation and W&B helpers.
   - Existing Teacher integration with Manifest V2.

2. `2026-06-15-v2-supervised-ddp-rft.md`
   - Shared supervised tokenization and Trainer core.
   - Two-rank DDP Full SFT and LoRA SFT.
   - Fixed-50 rank-0 evaluation every 100 optimizer steps.
   - Checkpoint selection and model export.
   - Dual-GPU RFT rollout, normalization, deduplication, and earliest-two rule.
   - Two-rank DDP RFT training.

3. `2026-06-15-v2-dpo-generation-training.md`
   - Metadata-capable dual-worker generation.
   - Forced-wrong and high-temperature rejected candidates.
   - Stable five-category selection and approximately 6k pairs.
   - Two-rank DDP TRL DPO.

4. `2026-06-15-v2-verl-grpo.md`
   - JSONL to verl Parquet conversion.
   - Framework-neutral reward and thin verl adapter.
   - Stock verl 0.6.0 FSDP2 plus vLLM launch configuration.
   - Validation dumps, custom metrics, continuation checkpoints.
   - Best/final selection and `verl.model_merger` export.

5. `2026-06-15-v2-pipeline-acceptance.md`
   - Stage orchestration and artifact lineage.
   - Resume and failure recovery.
   - Final evaluation matrix.
   - GPU smoke and end-to-end acceptance runbooks.
   - README coverage for every functional directory.

## Dependency Graph

```text
Phase 1 foundations
  ├── Phase 2 supervised + RFT
  │     ├── Phase 3 DPO
  │     └── Phase 4 GRPO
  └──────────────────────────┐
                             v
                    Phase 5 orchestration
```

Phase 3 requires Full SFT `best/` and the SFT 8k data contract from Phases 1
and 2. Phase 4 requires Full SFT `best/`, the GRPO 4k split, and common
evaluation from Phases 1 and 2. Phase 5 begins only after Phases 1-4 pass
their CPU and applicable distributed tests.

## Execution Rules

- Execute one phase plan at a time.
- Use TDD for every behavior change.
- Do not modify `post_train` unless a phase plan names a narrowly scoped
  compatibility change.
- Run commands from the repository root unless the command explicitly starts
  with `cd post_train_v2`.
- Keep all V2 outputs under `post_train_v2/data/` and
  `post_train_v2/outputs/`.
- Commit after each task using the commit message stated in the phase plan.
- Before beginning a later phase, run:

```bash
python -m pytest -q post_train_v2/tests
git status --short
```

Expected: all implemented V2 tests pass; the worktree contains only changes
belonging to the current task.

## Phase Completion Gates

### Phase 1

```bash
python -m pytest -q \
  post_train_v2/tests/unit \
  post_train_v2/tests/data \
  post_train_v2/tests/evaluation \
  post_train_v2/tests/generation
```

Expected: PASS, including deterministic split hashes and Teacher manifest
compatibility.

### Phase 2

```bash
python -m pytest -q \
  post_train_v2/tests/training \
  post_train_v2/tests/distributed \
  post_train_v2/tests/generation/test_rft_pipeline.py
```

Expected: PASS. On the remote host, the Full SFT, LoRA, and RFT two-rank smoke
commands each complete two optimizer steps and publish loadable artifacts.

### Phase 3

```bash
python -m pytest -q \
  post_train_v2/tests/dpo \
  post_train_v2/tests/distributed/test_dpo_ddp.py
```

Expected: PASS. The remote DPO smoke command completes two optimizer steps
without duplicate W&B runs.

### Phase 4

```bash
python -m pytest -q \
  post_train_v2/tests/verl \
  post_train_v2/tests/rewards \
  post_train_v2/tests/evaluation/test_grpo_selection.py
```

Expected: PASS. The remote GRPO smoke run completes one trainer iteration,
saves a native checkpoint, writes fixed-50 validation output in the production
configuration test, and merges a directly loadable actor export.

### Phase 5

```bash
python -m pytest -q post_train_v2/tests
python post_train_v2/scripts/pipeline/run_pipeline.py \
  --config post_train_v2/configs/pipeline/smoke.yaml \
  --dry-run
```

Expected: all tests pass; dry-run prints the ordered stage DAG, resolved
inputs, outputs, and resume decisions without loading a model.

## Final Remote Acceptance Order

```bash
cd post_train_v2
source .venv/bin/activate

python scripts/env/check_runtime.py \
  --manifest configs/environment/runtime-cu128.json

CUDA_VISIBLE_DEVICES=0,1 python scripts/env/smoke_nccl.py

python scripts/pipeline/run_pipeline.py \
  --config configs/pipeline/production.yaml \
  --from-stage build_source \
  --through-stage final_eval
```

The production command is added only in Phase 5. It must refuse to run when
Level 1 runtime gates are not recorded as passed in the configured runtime
acceptance file.

