# Distributed and verl Migration Plan

## 1. Target Architecture

V2 uses a hybrid architecture selected by training-stage requirements.

The implementation baseline is pinned in `environment.md`,
`pyproject.toml`, `requirements-runtime.txt`, and
`constraints-verl071-vllm017-cu128.txt`. Core code must target those APIs
rather than dynamically adapting to arbitrary library versions.

```text
                         shared task library
               prompts / solver / validation / metrics
                              |
       +----------------------+----------------------+
       |                                             |
Transformers/TRL distributed stack                 verl stack
Full SFT / LoRA / RFT / DPO                        GRPO
torchrun or Accelerate + DDP                       FSDP/FSDP2
Trainer-managed optimization                       vLLM rollout
                                                   rule reward
```

PPO and critic training are explicitly excluded.

## 2. Stage-by-Stage Strategy

### Source data preparation

Retain as CPU-oriented preprocessing. It does not benefit materially from
GPU distributed training.

Potential later improvement:

- Parallelize solver work with deterministic ordered collection if source
  preprocessing becomes a bottleneck.

Do not migrate this stage to verl.

### Qwen3-8B teacher generation

Use vLLM with both GPUs. Two possible execution modes must be benchmarked:

1. Tensor parallel size 2
   - One Qwen3-8B vLLM engine spans both GPUs.
   - Simple ordered generation and resume.
   - Appropriate if one 40 GB GPU cannot meet the desired batch/KV-cache
     target or TP improves practical throughput.

2. Two independent tensor-parallel-size-1 workers
   - Each GPU owns one full Qwen3-8B engine.
   - Input is deterministically split into two shards.
   - Usually offers better throughput when the model fits comfortably on one
     GPU.
   - Requires shard manifests, independent resume files, and stable ordered
     merge.

The phrase "two GPUs process data in parallel" does not uniquely choose
between these modes. V2 should support a configured generation topology, but
one mode must be selected as the first implementation.

Retain teacher acceptance and solver filtering outside verl.

### Full SFT

Rewrite the launch path for two-rank DDP:

- One process per GPU.
- One full Qwen3-0.6B model replica per GPU.
- DistributedSampler supplied through Trainer/Accelerate.
- BF16.
- Gradient checkpointing retained initially for behavioral continuity, then
  benchmarked because it may be unnecessary for a 0.6B model on 40 GB GPUs.
- Main-process-only fixed evaluation and artifact writes.

Recommended conservative starting configuration:

- per-device micro batch: 8
- gradient accumulation: 2
- world size: 2
- global batch: `8 * 2 * 2 = 32`
- max sequence length: 256
- learning rate: `1e-5`
- warmup ratio: `0.03`
- cosine schedule
- save every 100 optimizer steps
- evaluate every 100 optimizer steps on the fixed 50 examples

The current effective batch is 16 on one device. Moving to a global batch of
32 changes optimization. If strict comparability is required, start with
micro batch 4 and gradient accumulation 2 for global batch 16.

### LoRA SFT

Use the same two-GPU DDP launcher and rank-safe callbacks.

Recommended conservative starting configuration:

- per-device micro batch: 16
- gradient accumulation: 1
- global batch: 32
- BF16
- LoRA rank 16, alpha 32, dropout 0.05

LoRA easily fits on one GPU, but two-GPU DDP is retained because it is an
explicit project requirement.

### RFT data generation and training

Keep the RFT definition:

```text
rollout multiple responses
  -> rule-filter correct responses
  -> supervised training on accepted full responses
```

Generation should reuse the same configurable two-GPU vLLM sharding layer as
teacher and DPO generation. RFT training should reuse the distributed full
SFT implementation with an RFT-specific config.

Recommended initial RFT training batch:

- per-device micro batch: 8
- gradient accumulation: 2
- global batch: 32

### DPO data generation

Keep the current two-route design:

- 50% forced-wrong teacher prompts.
- 50% high-temperature teacher rollout.

Reuse the two-GPU generation topology and retain current rejected semantic
categories and selection priorities.

This stage should remain outside verl because it is offline preference-data
construction, not online actor rollout.

### DPO training

Use TRL `DPOTrainer` under two-rank DDP.

Recommended conservative starting configuration:

- per-device micro batch: 2
- gradient accumulation: 4
- global batch: `2 * 4 * 2 = 16`
- BF16
- gradient checkpointing enabled initially
- beta: 0.05
- learning rate: `5e-7`
- one epoch

DPO requires chosen and rejected forward paths and is more memory-intensive
than SFT. The 0.6B model should still fit comfortably on two 40 GB GPUs.

### GRPO

Replace the current custom training loop with verl.

Recommended initial characteristics:

- training backend: verl 0.7.1 with the FSDP/FSDP2 interface verified by the
  required two-GPU smoke test on PyTorch 2.10.0;
- rollout backend: vLLM.
- actor model: full-SFT Qwen3-0.6B checkpoint.
- critic: disabled.
- reward model: disabled.
- custom rule reward: enabled.
- advantage estimator: GRPO.
- group size: 4 initially.
- prompt global train batch: 4 initially.
- generated trajectories per rollout batch: 16.
- response length: 256.
- prompt length: 256.
- BF16.
- gradient checkpointing: enabled for the first compatibility run, then
  benchmarked.
- KL coefficient: 0, matching the current confirmed requirement.
- fixed 50-example evaluation remains required.

verl terminology differs from the current config:

| Current field | Intended verl concept |
| --- | --- |
| `batch_size` | global prompt train batch size |
| `group_size` | rollout `n` |
| `policy_updates_per_rollout` | GRPO actor update epochs/count; verl may retain a legacy `ppo_epochs` field name |
| `clip_eps` | GRPO actor policy-ratio clip, not advantage clipping |
| `format_reward` | custom reward component |
| `answer_reward` | custom reward component |
| `kl_coeff` | disabled KL reward/loss configuration |
| `sync_every_steps` | removed; verl owns actor-rollout synchronization |

Exact config keys must follow verl 0.7.1. Configuration copied from newer
`main` documentation must not be used without checking it against the v0.7.1
source tree.

## 3. JSONL-to-verl-Parquet Boundary

The converter should consume the existing GRPO JSONL without changing the
canonical task warehouse.

### Input

```json
{
  "id": "train-000001",
  "prompt": "Using the numbers ...",
  "numbers": [1, 2, 3, 4],
  "target": 24,
  "gold_expr": "(1+3)*(2+4)",
  "bucket": {}
}
```

### Proposed verl record

```python
{
    "data_source": "countdown",
    "prompt": [
        {
            "role": "user",
            "content": "Using the numbers ..."
        }
    ],
    "ability": "math",
    "reward_model": {
        "style": "rule",
        "ground_truth": {
            "numbers": [1, 2, 3, 4],
            "target": 24
        }
    },
    "extra_info": {
        "id": "train-000001",
        "gold_expr": "(1+3)*(2+4)",
        "bucket": {}
    }
}
```

The converter must:

- preserve source IDs;
- preserve prompt text exactly;
- store prompts as chat-message lists;
- retain numbers and target in reward ground truth;
- retain gold expression for diagnostics, not reward equivalence;
- write Parquet deterministically;
- validate schema and row counts;
- write a conversion manifest containing source hash, output hash, count, and
  schema version;
- create train and validation Parquet files separately;
- reject duplicate IDs.

## 4. Custom Reward Adapter

The adapter should expose the function signature required by the pinned verl
version and delegate to a framework-neutral Countdown reward implementation.

Logical behavior:

```text
generated response
  -> detect complete answer tag
  -> extract expression
  -> parse allowed AST
  -> verify exact input-number multiset
  -> evaluate with Fraction arithmetic
  -> compare exactly with target
```

Reward components:

- complete extractable `<answer>...</answer>`: `0.2`
- fully valid and correct expression: `1.0`
- total correct reward: `1.2`

The adapter must not:

- call the search solver to decide whether the generated expression is
  correct;
- accept approximate floating-point equality;
- require integer-only intermediate division;
- compare generated text with `gold_expr`;
- mutate input records.

The framework-neutral reward result should expose diagnostics:

- `reward`
- `format_ok`
- `answer_correct`
- `error`
- extracted expression
- evaluated exact value

These diagnostics are needed to retain current W&B metrics.

## 5. Distributed Launch Design

### Trainer-based stages

Preferred initial launcher:

```text
torchrun --standalone --nproc_per_node=2 ...
```

Accelerate may wrap this launch, but V2 should not maintain two competing
default launch paths. A single canonical launcher should be documented and
tested.

DDP requirements:

- rank-local CUDA device selection;
- no `device_map="auto"` in training;
- main-process-only W&B initialization where Trainer does not already manage
  it;
- main-process-only custom evaluation and output writes;
- synchronization around evaluation;
- deterministic distributed sampling seed;
- resumed global step consistent across ranks;
- failure if world size is not the configured value for production runs,
  with a separate one-GPU smoke mode.

### Optional strategies

FSDP2 and DeepSpeed ZeRO-2 may be retained as future optional configurations
for larger models, but they should not be implemented in the first
SFT/DPO rewrite unless a concrete need appears.

### GRPO

verl/Ray owns worker placement and model state movement. V2 configuration must
make placement explicit for exactly two GPUs and must avoid accidental actor,
reference, critic, or reward-model allocations that are not required.

## 6. Checkpoint Strategy

### SFT, LoRA, RFT, DPO

Periodic training checkpoints must support continuation and include:

- model or adapter state;
- optimizer state;
- scheduler state;
- global step;
- Trainer state;
- RNG state where supported;
- tokenizer and model configuration.

Final exports:

- Full SFT/RFT/DPO: directly loadable with
  `AutoModelForCausalLM.from_pretrained`.
- LoRA: adapter checkpoint plus explicit base-model metadata; optionally a
  separately produced merged export.

Retention recommendation:

- save every 100 optimizer steps;
- keep the latest two periodic checkpoints by default;
- always retain `final`;
- record best validation checkpoint only if a stable selection metric is
  confirmed.

### GRPO

Use verl fault-tolerant checkpoints for training continuation. Also produce a
Hugging Face export at defined milestones or final completion so the common
evaluator can load the actor without verl.

## 7. Evaluation Strategy

Preserve:

- fixed validation set of 200;
- fixed per-step evaluation subset of 50;
- evaluation every 100 optimizer steps;
- full sample trace for each fixed evaluation;
- maximum 256 generated tokens;
- thinking disabled;
- deterministic decoding for evaluation;
- solver-backed accuracy and format metrics.

Distributed changes:

- evaluation executes once per checkpoint/step, not once per rank;
- training processes synchronize before and after evaluation;
- evaluation files use atomic writes;
- a failed evaluation must be reported clearly and must not silently corrupt
  training output.

For GRPO, the common evaluator should run from a stable actor snapshot or a
verl-supported evaluation hook. The exact integration depends on the pinned
verl version.

## 8. Logging Strategy

W&B remains optional and config-driven.

Common metadata:

- stage;
- model source;
- dataset manifest hash;
- git revision;
- world size;
- GPU type;
- precision;
- effective global batch;
- seed;
- output directory;
- dependency versions.

Common training metrics:

- loss;
- learning rate;
- optimizer step;
- epoch or consumed samples;
- evaluation accuracy;
- format rate;
- average generation length;
- truncation count.

GRPO-specific metrics:

- mean reward;
- reward standard deviation;
- group reward standard deviation;
- fraction of zero-standard-deviation groups;
- accuracy;
- format rate;
- policy loss;
- KL metric, explicitly zero/disabled when configured;
- optional entropy;
- rollout throughput and token counts where verl exposes them.

Only the global/main process should own a logical W&B run.

## 9. Recommended V2 File Responsibilities

### Shared source

- `src/countdown/prompts.py`: task prompt and chat-message construction.
- `src/countdown/solver.py`: source solving and complexity metadata.
- `src/countdown/validation.py`: exact expression validation.
- `src/countdown/sampling.py`: deterministic stratified sampling.
- `src/rewards/countdown.py`: framework-neutral reward result.
- `src/evaluation/countdown.py`: shared scoring and aggregation.
- `src/distributed/runtime.py`: rank, world size, barriers, and main-process
  helpers.
- `src/tracking/wandb.py`: common run initialization and metric normalization.

### Scripts

- `scripts/data/build_source.py`
- `scripts/data/build_splits.py`
- `scripts/data/convert_grpo_to_verl.py`
- `scripts/generation/build_teacher_pool.py`
- `scripts/generation/build_rft_data.py`
- `scripts/generation/build_dpo_data.py`
- `scripts/sft/train.py`
- `scripts/dpo/train.py`
- `scripts/grpo/launch_verl.py` or a version-pinned shell launcher
- `scripts/eval/evaluate.py`

Separate Full SFT, LoRA, and RFT behavior should preferably be selected by
validated config rather than three mostly duplicated trainer files.

### verl integration

- `verl/data/countdown.py`: dataset conversion schema helpers.
- `verl/rewards/countdown_reward.py`: thin verl-compatible adapter.
- `verl/configs/grpo_qwen3_0_6b_2x40gb.yaml`: pinned two-GPU GRPO config.
- `verl/launch/run_grpo_2gpu.sh`: canonical launch command.

## 10. Scripts to Add, Refactor, or Retire

### Add

- Distributed environment validation.
- Canonical two-GPU launch scripts.
- Generation shard planner and deterministic merger.
- JSONL-to-verl-Parquet converter.
- Framework-neutral reward module.
- Thin verl reward adapter.
- Checkpoint compatibility/export checker.
- Distributed smoke tests.
- Manifest/hash validation tools.

### Refactor from current implementation

- Source data build.
- Teacher pool generation.
- SFT trainer.
- LoRA application.
- RFT data generation and SFT reuse.
- DPO generation and training.
- Common evaluator.
- W&B utilities.

### Retire after parity is verified

- Custom GRPO policy optimizer.
- Custom checkpoint-reload vLLM synchronization.
- Duplicate SFT/RFT training entry implementations.

No current script should be deleted before V2 output parity and checkpoint
loading have been demonstrated.

## 11. Rewrite Priority

1. Freeze schemas, manifests, dependency versions, and acceptance tests.
2. Port task semantics and add parity tests against `post_train`.
3. Implement distributed runtime and rank-safe fixed evaluation.
4. Implement two-GPU DDP Full SFT and verify resumable checkpoints.
5. Reuse the trainer for LoRA and RFT.
6. Implement two-GPU generation sharding and merge.
7. Port DPO data generation and two-GPU DDP DPO.
8. Implement verl Parquet conversion and schema validation.
9. Implement framework-neutral reward and verl adapter.
10. Pin verl and implement the two-GPU GRPO configuration.
11. Verify GRPO checkpoint continuation and Hugging Face export.
12. Run end-to-end smoke and compatibility comparisons.

## 12. Verification Gates

Core implementation should not begin until the open questions are resolved.

Later implementation must pass:

- unit parity tests for prompt, solver, validation, reward, and buckets;
- deterministic split tests;
- JSONL-to-Parquet round-trip/schema tests;
- two-process CPU/Gloo distributed side-effect tests;
- two-GPU one-step SFT, LoRA, RFT, and DPO smoke tests;
- two-GPU minimal verl GRPO smoke test;
- checkpoint interruption and resume test;
- final Hugging Face loading and fixed-set evaluation test;
- W&B single-run/no-duplicate-step test.
