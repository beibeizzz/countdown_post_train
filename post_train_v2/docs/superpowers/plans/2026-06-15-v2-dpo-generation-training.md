# V2 DPO Generation and Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build stable, category-controlled DPO pairs from Qwen3-8B rejected generations and train Full SFT Qwen3-0.6B with two-rank TRL DPO.

**Architecture:** The existing dual-worker engine gains an optional metadata result protocol used by DPO without changing Teacher string generation. A framework-neutral builder classifies candidates, applies exact quotas and priority fallback, publishes approximately 6k pairs, then a thin DPO runner reuses distributed evaluation, model loading, tracking, and export foundations.

**Tech Stack:** vLLM 0.9.1, TRL 0.19.1 `DPOTrainer`, PyTorch DDP, Flash Attention 2, pytest.

---

## File Map

Create:

- `post_train_v2/src/generation/metadata.py`
- `post_train_v2/src/generation/dpo.py`
- `post_train_v2/src/training/dpo.py`
- `post_train_v2/scripts/dpo/build_dpo_data.py`
- `post_train_v2/scripts/dpo/train_dpo.py`
- DPO configs, tests, README, and runbook.

Modify:

- `post_train_v2/src/generation/parallel_vllm.py`
- `post_train_v2/src/countdown/prompts.py`
- shared Trainer argument builder only to expose DPO-specific fields.

## Task 1: Metadata-Capable Worker Protocol

**Files:**

- Create: `post_train_v2/src/generation/metadata.py`
- Modify: `post_train_v2/src/generation/parallel_vllm.py`
- Create: `post_train_v2/tests/generation/test_generation_metadata.py`

- [ ] **Step 1: Write failing protocol tests**

```python
record = GenerationRecord(
    text="<answer>1+1</answer>",
    finish_reason="stop",
    token_count=8,
    stop_reason=None,
)
assert classify_truncation(record, max_new_tokens=256).truncated is False
```

Assert `finish_reason="length"` is truncated, token count at the limit is
used only when finish metadata is absent, and Teacher `generate()` still
returns `list[str]`.

- [ ] **Step 2: Implement**

Add `generate_with_metadata()` to the worker-side generator and
`ParallelVLLMEngine.generate(batch_id, items, include_metadata=True)`. Use a tagged
`WorkerGeneration` dataclass so string and metadata responses cannot be
mixed silently.

- [ ] **Step 3: Verify and commit**

```bash
python -m pytest -q post_train_v2/tests/generation/test_generation_metadata.py post_train_v2/tests/generation/test_parallel_vllm.py
git add post_train_v2/src/generation post_train_v2/tests/generation
git commit -m "feat: add dual-worker generation metadata"
```

## Task 2: DPO Candidate Classification

**Files:**

- Create: `post_train_v2/src/generation/dpo.py`
- Create: `post_train_v2/tests/dpo/test_candidate_classification.py`

- [ ] **Step 1: Write failing category tests**

Cover exactly:

```text
wrong_value
number_mismatch
invalid_expression
missing_answer_tag
truncated
unexpected_correct
```

`truncated` takes precedence when metadata proves truncation.
`unexpected_correct` is never eligible.

- [ ] **Step 2: Implement**

Return:

```python
@dataclass(frozen=True)
class DPOCandidate:
    source_id: str
    candidate_id: str
    generation_route: Literal["forced_wrong", "high_temp"]
    rejected: str
    rejected_category: str
    validation: dict[str, Any]
    rollout_index: int
```

- [ ] **Step 3: Verify and commit**

```bash
python -m pytest -q post_train_v2/tests/dpo/test_candidate_classification.py
git add post_train_v2/src/generation/dpo.py post_train_v2/tests/dpo
git commit -m "feat: classify dpo rejected candidates"
```

## Task 3: Stable Quotas and Pair Selection

**Files:**

- Modify: `post_train_v2/src/generation/dpo.py`
- Create: `post_train_v2/tests/dpo/test_pair_selection.py`

- [ ] **Step 1: Write failing quota tests**

For target 6000 assert integer quotas:

```python
{
    "wrong_value": 4200,
    "number_mismatch": 900,
    "invalid_expression": 600,
    "missing_answer_tag": 180,
    "truncated": 120,
}
```

Assert one pair per question, stable hash ordering, route balance when supply
permits, and deficit fallback follows the strict category priority.

- [ ] **Step 2: Implement**

Order candidates using:

```python
key = sha256_text(f"{seed}|{source_id}|{candidate_id}")
```

First fill category quotas, then fill shortfalls from remaining candidates in
global priority order. Never relax validation or include
`unexpected_correct`.

- [ ] **Step 3: Verify and commit**

```bash
python -m pytest -q post_train_v2/tests/dpo/test_pair_selection.py
git add post_train_v2/src/generation/dpo.py post_train_v2/tests/dpo/test_pair_selection.py
git commit -m "feat: add stable dpo pair selection"
```

## Task 4: Dual-GPU DPO Data CLI

**Files:**

- Create: `post_train_v2/scripts/dpo/{build_dpo_data.py,README.md}`
- Create: `post_train_v2/configs/dpo/build.yaml`
- Create: `post_train_v2/configs/dpo/build_smoke.yaml`
- Create: `post_train_v2/tests/dpo/test_build_dpo_data.py`

- [ ] **Step 1: Write failing coordinator tests**

Assert half of candidate requests use the forced-wrong prompt and half use
the original prompt, all request seeds derive from stage/source/rollout, and
outputs preserve the five category names. Each chosen response must pass the
exact Countdown validator for its own numbers and target before any rejected
request is generated.

- [ ] **Step 2: Implement**

Use Qwen3-8B, forced temperature 0.3, high temperature 1.0, top-p 0.95,
thinking disabled, max 256. Publish candidate JSONL, pair JSONL, and Manifest
V2 with category/route counts and shortfall.

- [ ] **Step 3: Verify and commit**

```bash
python -m pytest -q post_train_v2/tests/dpo/test_build_dpo_data.py
git add post_train_v2/scripts/dpo post_train_v2/configs/dpo/build*.yaml post_train_v2/tests/dpo
git commit -m "feat: add dual-gpu dpo data builder"
```

## Task 5: DPO Dataset and Trainer Construction

**Files:**

- Create: `post_train_v2/src/training/dpo.py`
- Create: `post_train_v2/tests/dpo/test_dpo_trainer.py`

- [ ] **Step 1: Write failing TRL construction tests**

Assert Full SFT `best/` is loaded with BF16/FA2, no explicit trainable
reference model is passed, `beta=0.05`, max length 256, learning rate `5e-7`,
one epoch, micro batch 4, accumulation 2, and logging every optimizer step.

- [ ] **Step 2: Implement**

Build a Hugging Face Dataset with `prompt`, `chosen`, and `rejected`. Use
TRL's implicit frozen reference copy. Reuse fixed-50 callback and
best/final export.

- [ ] **Step 3: Verify and commit**

```bash
python -m pytest -q post_train_v2/tests/dpo/test_dpo_trainer.py
git add post_train_v2/src/training/dpo.py post_train_v2/tests/dpo/test_dpo_trainer.py
git commit -m "feat: add v2 dpo trainer core"
```

## Task 6: Public DPO Training Entry

**Files:**

- Create: `post_train_v2/scripts/dpo/train_dpo.py`
- Create: `post_train_v2/configs/dpo/train.yaml`
- Create: `post_train_v2/configs/dpo/train_smoke.yaml`
- Create: `post_train_v2/tests/dpo/test_train_dpo_cli.py`
- Create: `post_train_v2/tests/distributed/test_dpo_ddp.py`

- [ ] **Step 1: Write failing CLI and rank tests**

Assert `--config`, `--max-steps`, and `--resume-from-checkpoint` are
forwarded. Two CPU ranks must create one evaluation artifact and one logical
tracking run.

- [ ] **Step 2: Implement and verify**

```bash
python -m pytest -q post_train_v2/tests/dpo post_train_v2/tests/distributed/test_dpo_ddp.py
```

- [ ] **Step 3: Commit**

```bash
git add post_train_v2/scripts/dpo post_train_v2/configs/dpo/train*.yaml post_train_v2/tests/dpo post_train_v2/tests/distributed/test_dpo_ddp.py
git commit -m "feat: add two-gpu dpo training entrypoint"
```

## Task 7: Phase Gate and Runbook

**Files:**

- Create: `post_train_v2/docs/runbooks/dpo.md`
- Update DPO and root README files.

- [ ] **Step 1: Run tests**

```bash
python -m pytest -q post_train_v2/tests/dpo post_train_v2/tests/distributed/test_dpo_ddp.py
git diff --check
```

- [ ] **Step 2: Document remote commands**

```bash
CUDA_VISIBLE_DEVICES=0,1 python post_train_v2/scripts/dpo/build_dpo_data.py \
  --config post_train_v2/configs/dpo/build_smoke.yaml --limit 8

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  post_train_v2/scripts/dpo/train_dpo.py \
  --config post_train_v2/configs/dpo/train_smoke.yaml --max-steps 2
```

- [ ] **Step 3: Commit**

```bash
git add post_train_v2/docs/runbooks/dpo.md post_train_v2/README.md post_train_v2/scripts/dpo/README.md
git commit -m "docs: add dpo runbook"
```
