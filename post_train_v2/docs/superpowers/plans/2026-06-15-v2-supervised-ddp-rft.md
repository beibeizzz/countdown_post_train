# V2 Supervised DDP and RFT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement two-GPU DDP Full SFT, LoRA SFT, and RFT with shared response-only supervision, fixed-50 evaluation, resumable checkpoints, best/final exports, and dual-GPU RFT data generation.

**Architecture:** One supervised core owns tokenization, collators, TrainingArguments, callbacks, checkpoint selection, and export. Thin stage entrypoints supply model adaptation and data paths. RFT rollout reuses the existing two-worker vLLM engine and feeds accepted responses back into the same full-parameter supervised core.

**Tech Stack:** PyTorch DDP, `torchrun`, Transformers 4.53.2 Trainer, PEFT 0.15.2, Flash Attention 2, vLLM 0.9.1, W&B, pytest.

---

## File Map

Create:

- `post_train_v2/src/distributed/runtime.py`: rank/world-size helpers, barriers, failure propagation.
- `post_train_v2/src/training/supervised_data.py`: chat rendering, response-only labels, collator.
- `post_train_v2/src/training/model_loading.py`: BF16/FA2 model and tokenizer loading.
- `post_train_v2/src/training/trainer_args.py`: shared TrainingArguments builder.
- `post_train_v2/src/training/fixed_eval.py`: rank-0 unwrapped fixed-50 callback.
- `post_train_v2/src/training/model_selection.py`: best-step ledger and export.
- `post_train_v2/src/training/supervised.py`: shared Full SFT/RFT Trainer runner.
- `post_train_v2/src/training/lora.py`: PEFT configuration and merge.
- `post_train_v2/src/generation/rft.py`: rollout expansion, normalization, deduplication, earliest-two selection.
- `post_train_v2/scripts/sft/train_full.py`
- `post_train_v2/scripts/sft/train_lora.py`
- `post_train_v2/scripts/sft/build_rft_data.py`
- `post_train_v2/scripts/sft/train_rft.py`
- `post_train_v2/scripts/sft/merge_lora.py`
- conservative stage configs under `post_train_v2/configs/sft/`.
- tests under `post_train_v2/tests/training/`, `distributed/`, and `generation/`.

Modify:

- `post_train_v2/src/generation/parallel_vllm.py`: expose stable per-request
  seed and rollout index input without changing Teacher defaults.
- `post_train_v2/src/evaluation/generation.py`: expose inference-mode
  generation used by the fixed evaluation callback.
- `post_train_v2/src/tracking/wandb.py`: expose rank-zero metric logging used
  by Trainer callbacks.

## Task 1: Distributed Runtime Contract

**Files:**

- Create: `post_train_v2/src/distributed/{__init__.py,runtime.py,README.md}`
- Create: `post_train_v2/tests/distributed/test_runtime.py`

- [ ] **Step 1: Write failing Gloo tests**

Spawn two CPU processes and assert:

```python
assert context.world_size == 2
assert context.is_main_process is (context.rank == 0)
assert shared_side_effect_count.value == 1
```

Add a test where rank 0 raises during the protected section and rank 1
receives a propagated failure instead of hanging at the second barrier.

- [ ] **Step 2: Verify RED**

```bash
python -m pytest -q post_train_v2/tests/distributed/test_runtime.py
```

- [ ] **Step 3: Implement**

```python
@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
```

Implement `main_rank_section(fn)` with a broadcasted success/error envelope so
all ranks either continue or raise.

- [ ] **Step 4: Verify and commit**

```bash
python -m pytest -q post_train_v2/tests/distributed/test_runtime.py
git add post_train_v2/src/distributed post_train_v2/tests/distributed/test_runtime.py
git commit -m "feat: add v2 distributed runtime contract"
```

## Task 2: Response-Only Supervised Dataset

**Files:**

- Create: `post_train_v2/src/training/{__init__.py,supervised_data.py,README.md}`
- Create: `post_train_v2/tests/training/test_supervised_data.py`

- [ ] **Step 1: Write failing tokenization tests**

```python
encoded = encode_prompt_response(
    tokenizer=fake_qwen_tokenizer(),
    prompt="question",
    response="reasoning\n<answer>1+1</answer>",
    max_seq_len=256,
)
assert encoded.labels[: encoded.prompt_length] == [-100] * encoded.prompt_length
assert any(label != -100 for label in encoded.labels)
assert encoded.supervised_text.endswith("<answer>1+1</answer>")
```

Assert `enable_thinking=False` is passed to both prompt-only and full
conversation templates, the whole response is supervised, and records with
no remaining response token after truncation are rejected.

- [ ] **Step 2: Verify RED**

```bash
python -m pytest -q post_train_v2/tests/training/test_supervised_data.py
```

- [ ] **Step 3: Implement**

Use a dataclass:

```python
@dataclass(frozen=True)
class EncodedSupervisedExample:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    prompt_length: int
    supervised_text: str
```

Right-pad `input_ids` and attention masks; use `-100` for prompt and padding
labels.

- [ ] **Step 4: Verify and commit**

```bash
python -m pytest -q post_train_v2/tests/training/test_supervised_data.py
git add post_train_v2/src/training post_train_v2/tests/training/test_supervised_data.py
git commit -m "feat: add response-only supervised dataset"
```

## Task 3: Shared Model Loading and Training Arguments

**Files:**

- Create: `post_train_v2/src/training/model_loading.py`
- Create: `post_train_v2/src/training/trainer_args.py`
- Create: `post_train_v2/tests/training/test_training_config.py`

- [ ] **Step 1: Write failing loader and argument tests**

Assert model loading receives:

```python
{
    "trust_remote_code": True,
    "attn_implementation": "flash_attention_2",
    "torch_dtype": torch.bfloat16,
}
```

Assert TrainingArguments contain cosine scheduling, 3% warmup, BF16,
gradient checkpointing, `logging_strategy="steps"`, `logging_steps=1`,
`logging_first_step=True`, save every 100 steps, and
`save_total_limit=2`.

- [ ] **Step 2: Implement**

Do not pass `device_map`. Set `use_cache=False` when gradient checkpointing is
enabled. Expose `--max-steps` by overriding `max_steps` only when supplied.

- [ ] **Step 3: Verify and commit**

```bash
python -m pytest -q post_train_v2/tests/training/test_training_config.py
git add post_train_v2/src/training/model_loading.py post_train_v2/src/training/trainer_args.py post_train_v2/tests/training/test_training_config.py
git commit -m "feat: add shared supervised training configuration"
```

## Task 4: Fixed-50 Distributed Evaluation Callback

**Files:**

- Create: `post_train_v2/src/training/fixed_eval.py`
- Create: `post_train_v2/tests/training/test_fixed_eval.py`
- Create: `post_train_v2/tests/distributed/test_fixed_eval_ddp.py`

- [ ] **Step 1: Write failing callback tests**

Assert steps 100, 200, and the final non-boundary step evaluate exactly once.
Assert rank 1 never calls generation. Assert rank 0 calls
`accelerator.unwrap_model(model)` and not the DDP wrapper's `forward`.

- [ ] **Step 2: Verify RED**

```bash
python -m pytest -q \
  post_train_v2/tests/training/test_fixed_eval.py \
  post_train_v2/tests/distributed/test_fixed_eval_ddp.py
```

- [ ] **Step 3: Implement**

The callback writes:

```text
post_train_v2/outputs/sft/full/eval/step_100/samples.jsonl
post_train_v2/outputs/sft/full/eval/step_100/metrics.json
post_train_v2/outputs/sft/full/eval/ledger.jsonl
```

Restore the model's prior train/eval state in `finally`. Broadcast evaluation
failure before leaving the synchronized section.

- [ ] **Step 4: Verify and commit**

```bash
python -m pytest -q post_train_v2/tests/training/test_fixed_eval.py post_train_v2/tests/distributed/test_fixed_eval_ddp.py
git add post_train_v2/src/training/fixed_eval.py post_train_v2/tests/training/test_fixed_eval.py post_train_v2/tests/distributed/test_fixed_eval_ddp.py
git commit -m "feat: add synchronized fixed evaluation"
```

## Task 5: Model Selection and Export

**Files:**

- Create: `post_train_v2/src/training/model_selection.py`
- Create: `post_train_v2/tests/training/test_model_selection.py`

- [ ] **Step 1: Write failing selection tests**

```python
assert select_best([
    result(100, accuracy=.5, format_rate=.9),
    result(200, accuracy=.5, format_rate=.95),
    result(300, accuracy=.5, format_rate=.95),
]).step == 200
```

Assert full-model `best/` and `final/` are direct-loadable publication units,
while LoRA `best/` remains an adapter and merge writes a separate full model.

- [ ] **Step 2: Implement**

Use ordering `(-accuracy, -format_rate, step)`. Publish export manifests only
after tokenizer files and direct-load checks succeed.

- [ ] **Step 3: Verify and commit**

```bash
python -m pytest -q post_train_v2/tests/training/test_model_selection.py
git add post_train_v2/src/training/model_selection.py post_train_v2/tests/training/test_model_selection.py
git commit -m "feat: add trainer model selection and export"
```

## Task 6: Full SFT Runner

**Files:**

- Create: `post_train_v2/src/training/supervised.py`
- Create: `post_train_v2/scripts/sft/{train_full.py,README.md}`
- Create: `post_train_v2/configs/sft/full.yaml`
- Create: `post_train_v2/configs/sft/full_smoke.yaml`
- Create: `post_train_v2/tests/training/test_train_full.py`

- [ ] **Step 1: Write failing runner tests**

Assert the public CLI accepts `--config`, `--max-steps`, and
`--resume-from-checkpoint`; calls
`trainer.train(resume_from_checkpoint=resume_path)`;
and uses micro batch 4, accumulation 2, world size 2, global batch 16.

- [ ] **Step 2: Implement**

The config freezes learning rate `1e-5`, three epochs, max sequence 256,
fixed-50 evaluation every 100, and output under
`post_train_v2/outputs/sft/full`.

- [ ] **Step 3: Verify and commit**

```bash
python -m pytest -q post_train_v2/tests/training/test_train_full.py
git add post_train_v2/src/training/supervised.py post_train_v2/scripts/sft post_train_v2/configs/sft/full*.yaml post_train_v2/tests/training/test_train_full.py
git commit -m "feat: add two-gpu full sft entrypoint"
```

## Task 7: LoRA SFT and Merge

**Files:**

- Create: `post_train_v2/src/training/lora.py`
- Create: `post_train_v2/scripts/sft/train_lora.py`
- Create: `post_train_v2/scripts/sft/merge_lora.py`
- Create: `post_train_v2/configs/sft/lora.yaml`
- Create: `post_train_v2/configs/sft/lora_smoke.yaml`
- Create: `post_train_v2/tests/training/test_train_lora.py`

- [ ] **Step 1: Write failing PEFT tests**

Assert automatic target modules resolve Qwen projection suffixes, adapter
parameters alone are trainable, input gradients are enabled under gradient
checkpointing, and merge loads the adapter over its base then calls
`merge_and_unload()`.

- [ ] **Step 2: Implement**

Freeze LoRA defaults `r=16`, alpha 32, dropout 0.05, learning rate `2e-5`,
micro batch 8, accumulation 2, global batch 32.

- [ ] **Step 3: Verify and commit**

```bash
python -m pytest -q post_train_v2/tests/training/test_train_lora.py
git add post_train_v2/src/training/lora.py post_train_v2/scripts/sft post_train_v2/configs/sft/lora*.yaml post_train_v2/tests/training/test_train_lora.py
git commit -m "feat: add two-gpu lora sft"
```

## Task 8: Dual-GPU RFT Data Construction

**Files:**

- Create: `post_train_v2/src/generation/rft.py`
- Create: `post_train_v2/scripts/sft/build_rft_data.py`
- Create: `post_train_v2/configs/sft/rft_rollout.yaml`
- Create: `post_train_v2/configs/sft/rft_rollout_smoke.yaml`
- Create: `post_train_v2/tests/generation/test_rft_pipeline.py`

- [ ] **Step 1: Write failing selection tests**

For each source row create four `(source_index, rollout_index)` requests.
Normalize only line endings and outer whitespace. Assert exact response
duplicates are removed, expression-equivalent but text-distinct responses
remain, and more than two correct responses retain the earliest two
`rollout_index` values.

- [ ] **Step 2: Implement**

Use Qwen3-8B, temperature 0.7, top-p 0.95, max 256, thinking disabled, and
stable per-request seeds. Publish accepted/rejected JSONL plus actual question
coverage and accepted count in Manifest V2.

- [ ] **Step 3: Verify and commit**

```bash
python -m pytest -q post_train_v2/tests/generation/test_rft_pipeline.py
git add post_train_v2/src/generation/rft.py post_train_v2/scripts/sft/build_rft_data.py post_train_v2/configs/sft/rft_rollout*.yaml post_train_v2/tests/generation/test_rft_pipeline.py
git commit -m "feat: add dual-gpu rft data generation"
```

## Task 9: RFT Training Entry

**Files:**

- Create: `post_train_v2/scripts/sft/train_rft.py`
- Create: `post_train_v2/configs/sft/rft_train.yaml`
- Create: `post_train_v2/configs/sft/rft_train_smoke.yaml`
- Create: `post_train_v2/tests/training/test_train_rft.py`

- [ ] **Step 1: Write failing entrypoint test**

Assert RFT uses Qwen3-0.6B base, the accepted RFT data, full-parameter
training, micro batch 4, accumulation 2, and the shared supervised runner.

- [ ] **Step 2: Implement and verify**

```bash
python -m pytest -q post_train_v2/tests/training/test_train_rft.py
```

- [ ] **Step 3: Commit**

```bash
git add post_train_v2/scripts/sft/train_rft.py post_train_v2/configs/sft/rft_train*.yaml post_train_v2/tests/training/test_train_rft.py
git commit -m "feat: add two-gpu rft training"
```

## Task 10: Phase Gate and Remote Runbook

**Files:**

- Create: `post_train_v2/docs/runbooks/supervised_and_rft.md`
- Update relevant README files.

- [ ] **Step 1: Run CPU gates**

```bash
python -m pytest -q \
  post_train_v2/tests/training \
  post_train_v2/tests/distributed \
  post_train_v2/tests/generation/test_rft_pipeline.py
git diff --check
```

- [ ] **Step 2: Record remote smoke commands**

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  post_train_v2/scripts/sft/train_full.py \
  --config post_train_v2/configs/sft/full_smoke.yaml --max-steps 2

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  post_train_v2/scripts/sft/train_lora.py \
  --config post_train_v2/configs/sft/lora_smoke.yaml --max-steps 2

CUDA_VISIBLE_DEVICES=0,1 python post_train_v2/scripts/sft/build_rft_data.py \
  --config post_train_v2/configs/sft/rft_rollout_smoke.yaml --limit 4

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  post_train_v2/scripts/sft/train_rft.py \
  --config post_train_v2/configs/sft/rft_train_smoke.yaml --max-steps 2
```

- [ ] **Step 3: Commit**

```bash
git add post_train_v2/docs/runbooks post_train_v2/README.md post_train_v2
git commit -m "docs: add supervised and rft runbook"
```
