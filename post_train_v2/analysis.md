# Existing Project Analysis

## 1. Current Project Goal

The existing `post_train` project implements a Countdown mathematics
post-training pipeline for Qwen3 models:

- Qwen3-8B acts as the teacher and generation model for data construction.
- Qwen3-0.6B is the target model.
- Responses may contain useful reasoning, but must contain a final
  `<answer>...</answer>` expression.
- Training and generation disable Qwen thinking mode.
- Generation is limited to 256 new tokens.
- Mathematical correctness is verified by a deterministic rule solver.

The implemented stages are data preparation, teacher filtering, SFT, LoRA
SFT, RFT, DPO, GRPO, and offline evaluation. PPO is not implemented and is
not part of V2.

## 2. Current Directory Responsibilities

### Shared Countdown modules

`post_train/src/countdown/` contains the reusable task semantics:

| Module | Responsibility |
| --- | --- |
| `prompts.py` | Builds the standard solution prompt, forced-wrong DPO prompt, and chat messages. |
| `generation.py` | Wraps vLLM chat generation and passes `enable_thinking=false`. |
| `solver.py` | Solves Countdown inputs and calculates expression complexity metadata. |
| `validation.py` | Extracts answer tags, parses expressions safely, checks number usage, and compares exact rational values. |
| `bucketing.py` | Assigns `num_count` and expression-complexity buckets. |
| `sampling.py` | Performs deterministic stratified balanced sampling. |
| `eval.py` | Scores generations and aggregates accuracy, format, length, and truncation metrics. |
| `io.py` | Reads/writes JSONL, JSON, and manifests. |
| `config.py` | Loads YAML and resolves repository-relative paths. |
| `wandb_utils.py` | Normalizes optional W&B initialization and metric logging. |

These modules are task-specific rather than trainer-specific. Most should be
ported or reused by V2.

### Data scripts

| Script | Inputs | Outputs | Behavior |
| --- | --- | --- | --- |
| `scripts/data/build_source.py` | `datasets/raw_train.parquet`, `datasets/raw_test.json` | normalized train pool, validation, fixed eval subset, solved test data, manifest | Solves each task, builds prompts and buckets, then performs fixed-seed stratified sampling. |
| `scripts/data/build_teacher_pool.py` | normalized train pool, Qwen3-8B | accepted 20k, rejected generations, manifest | Generates once per prompt in source order, verifies with the rule validator, and supports file-based resume. |
| `scripts/data/build_sft_splits.py` | accepted 20k | SFT 8k and GRPO 4k JSONL | Samples independently with stratification over existing buckets. |

### SFT and RFT scripts

| Script | Responsibility |
| --- | --- |
| `scripts/sft/train_full.py` | Full-parameter causal SFT using Transformers `Trainer`. It masks prompt tokens and trains on the complete teacher response. |
| `scripts/sft/train_lora.py` | Applies PEFT LoRA and reuses the full SFT dataset, collator, trainer, and evaluator. |
| `scripts/sft/build_rft_data.py` | Samples multiple vLLM completions and retains solver-correct responses. |
| `scripts/sft/train_rft.py` | Maps the RFT config into the full SFT trainer. |

### DPO scripts

| Script | Responsibility |
| --- | --- |
| `scripts/dpo/build_dpo_data.py` | Uses Qwen3-8B to generate forced-wrong and high-temperature rejected candidates, classifies them, and prioritizes mathematically close negatives. |
| `scripts/dpo/train_dpo.py` | Formats prompt/chosen/rejected records and trains with TRL `DPOTrainer`. |

Rejected categories retain the existing semantic names:

- `wrong_value`
- `number_mismatch`
- `invalid_expression`
- `missing_answer_tag`
- `truncated`

The most valuable category is a complete, parseable answer that uses the
correct numbers exactly once but evaluates to the wrong target.

### GRPO script

`scripts/grpo/train_grpo.py` combines all GRPO responsibilities in one file:

- Starts a vLLM rollout model.
- Samples grouped responses.
- Computes format and correctness rewards.
- Normalizes rewards within each group.
- Encodes prompt/completion sequences.
- Executes a custom policy-gradient update.
- Periodically saves model weights.
- Attempts to reload vLLM from saved weights.
- Runs fixed evaluation.
- Writes JSONL and W&B metrics.

This is useful as a behavioral prototype, but it is not a robust distributed
GRPO implementation.

### Evaluation

`scripts/eval/evaluate_model.py`:

- Loads full Hugging Face checkpoints or PEFT adapters.
- Applies the Qwen chat template with thinking disabled.
- Generates at most 256 tokens.
- Scores with the common validator.
- Writes sample-level traces and aggregate metrics.

The Trainer callback reuses this evaluator every 100 optimizer steps on a
fixed set of 50 examples.

## 3. Current Execution Order

```text
build_source.py
  -> build_teacher_pool.py
  -> build_sft_splits.py
     -> train_full.py
     -> train_lora.py
     -> build_rft_data.py -> train_rft.py
     -> build_dpo_data.py -> train_dpo.py
     -> train_grpo.py
  -> evaluate_model.py for each final checkpoint
```

The main full-SFT checkpoint is the intended base for DPO and GRPO.

## 4. Current Data Contracts

### Normalized source record

Important fields:

- `id`
- `source_index`
- `numbers`
- `target`
- `gold_expr`
- `prompt`
- `bucket`

### Accepted SFT record

Adds:

- `response`
- `teacher_expr`
- `validation`

SFT trains on the full `response`, not only the answer span.

### DPO record

Important fields:

- `prompt`
- `chosen`
- `rejected`
- rejected category and generation-source metadata

### GRPO source record

The current GRPO input remains JSONL and contains the prompt plus solver
ground truth. verl expects a Parquet dataset with chat-message prompts,
reward metadata, and extra information, so a conversion boundary is needed.

## 5. Existing Distributed Capability

### Explicitly implemented

- vLLM accepts `tensor_parallel_size`, but only the custom GRPO config
  currently exposes it.
- Evaluation loads models with `device_map="auto"`.

### Available implicitly through dependencies

Transformers `Trainer` and TRL `DPOTrainer` can participate in distributed
training when launched through `torchrun` or Accelerate. The current scripts
do not provide launchers or distributed configuration, but their training
loops are compatible with DDP in principle.

### Not implemented

- No repository `torchrun` launcher.
- No Accelerate configuration.
- No DeepSpeed configuration.
- No FSDP/FSDP2 configuration.
- No explicit process-rank handling.
- No distributed sampler owned by custom code.
- No rank-zero guard around custom evaluation and file writes.
- No distributed checkpoint manifest.
- No resume path restoring optimizer, scheduler, scaler, sampler, and RNG.
- No multi-process sharding for teacher, RFT, or DPO generation.
- No distributed GRPO actor update.

## 6. Distributed Risks in the Current Code

### Trainer callbacks

The custom fixed-evaluation callback performs generation and writes files
without checking process rank. Under two-GPU DDP, both ranks may evaluate the
same 50 examples and write the same output paths.

V2 must run custom evaluation and side-effecting writes only on the main
process, with barriers where necessary.

### Full-model saving

Trainer-managed checkpoint saving is generally distributed-aware, but custom
final-save and callback behavior must be checked under the selected
Transformers version.

### Generation jobs

Teacher, RFT, and DPO generation currently instantiate one vLLM engine and
process a single ordered stream. They have no dataset sharding, shard
manifests, deterministic merge, or cross-process resume protocol.

### Current GRPO algorithm

The custom script does not store behavior-policy log probabilities and does
not calculate a policy probability ratio. Its `clip_eps` clips normalized
advantages rather than a PPO/GRPO policy ratio. It also has no reference
model path and deliberately rejects nonzero KL.

Consequently, V2 should preserve the reward semantics and metrics but should
not treat the current optimizer loop as the target GRPO implementation.

### Current GRPO weight synchronization

Actor weights are saved as Hugging Face checkpoints and vLLM is re-created
from the path. Reload failures return the old generator silently. The old
engine is not explicitly released, and successful synchronization is not
verified.

verl should own actor/rollout synchronization in V2.

### Current GRPO checkpointing

The custom checkpoint saves model and tokenizer only. It does not save:

- optimizer state
- scheduler state
- global data position
- Python or PyTorch RNG state
- vLLM/rollout state

It is therefore an export checkpoint, not a complete fault-tolerant training
checkpoint.

## 7. Logging and Metrics

Current Trainer-based stages can log to W&B through `report_to`. GRPO logs
manually. Current GRPO metrics include:

- loss
- mean reward
- reward standard deviation
- group reward standard deviation
- fraction of zero-standard-deviation groups
- accuracy
- format rate
- approximate KL, currently always zero
- optional entropy
- average and maximum generation tokens
- truncation count
- rollout count
- learning rate

V2 should map these names into a common metric namespace while retaining the
same observable task metrics.

## 8. Components Suitable for Reuse

Recommended for reuse or careful porting:

- Prompt construction.
- Chat-message construction.
- Exact Fraction-based expression validation.
- AST operator allow-list.
- Number multiset validation.
- Solver and expression complexity.
- Bucketing and stratified sampling.
- DPO rejected-category semantics.
- Fixed validation/evaluation split.
- Evaluation aggregation.
- Manifest concepts.

## 9. Components Requiring Replacement or Refactoring

Must be replaced for the target architecture:

- Custom GRPO training loop.
- Checkpoint-based best-effort vLLM synchronization.

Must be refactored:

- Training launch and process-rank handling.
- Trainer callbacks and side effects.
- Generation data sharding and merge.
- Configuration schema.
- Checkpoint and resume contract.
- Common experiment metadata.
- JSONL-to-verl data conversion.
- verl reward entrypoint.

## 10. Current Artifact State

The existing repository contains a small processed dataset produced by an
earlier limited source-data run. Teacher accepted data, SFT/DPO/GRPO stage
datasets, model directories, and trained output checkpoints are not present
in the local workspace. Migration tests must therefore use synthetic fixtures
until full remote artifacts are available.

