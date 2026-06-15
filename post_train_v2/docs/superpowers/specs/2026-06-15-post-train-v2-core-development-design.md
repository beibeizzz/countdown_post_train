# Post-Train V2 Core Development Design

## 1. Purpose

This document freezes the architecture and behavioral contracts for fully
developing `post_train_v2` as a distributed Countdown post-training project.
It replaces the planning-only status described by the current V2 README.

The project targets:

- local Qwen3-0.6B as the trainable model;
- local Qwen3-8B as the teacher and offline generation model;
- one node with two visible NVIDIA A100 GPUs, each limited to 40 GB;
- Python 3.11.15 and the exact dependency versions pinned by
  `post_train_v2/pyproject.toml`;
- two-GPU DDP for Full SFT, LoRA SFT, RFT training, and DPO;
- verl 0.6.0 with FSDP2 and vLLM for GRPO.

PPO, a learned reward model, critic training, and migration of every training
stage into verl are explicitly out of scope.

## 2. Selected Rewrite Strategy

V2 uses a staged hybrid rewrite.

```text
shared Countdown, data, artifact, evaluation, and runtime layers
                              |
              +---------------+---------------+
              |                               |
     Transformers / TRL                    verl 0.6.0
     Full SFT / LoRA / RFT / DPO           GRPO
     torchrun two-rank DDP                 FSDP2 + vLLM
```

This strategy keeps the stable Trainer and TRL stack for supervised and
preference training while using verl only for the online rollout and
group-relative optimization stage it is designed to manage.

Development is split into five independently reviewed implementation plans:

1. Shared foundations, Manifest V2, data preparation, and common evaluation.
2. Two-GPU DDP Full SFT, LoRA, and RFT.
3. DPO data generation and two-GPU DPO.
4. verl Parquet conversion, custom reward, and two-GPU GRPO.
5. End-to-end orchestration, recovery, final evaluation, and documentation.

## 3. Compatibility Boundary

`post_train` remains independently runnable and is not modified as part of
the V2 core rewrite unless a later task explicitly requires a compatibility
fix.

V2 may read:

- raw and processed data under `post_train/data/`;
- local model directories under `post_train/model/`;
- complete Hugging Face model exports produced by either project;
- LoRA adapters with explicit or embedded base-model metadata;
- legacy custom GRPO Hugging Face exports as ordinary model weights.

V2 writes new data, manifests, checkpoints, evaluations, and logs only below
`post_train_v2/` by default. It must not implicitly overwrite legacy data or
outputs.

V2 guarantees training-state continuation only for checkpoints produced by
V2:

- Transformers or TRL Trainer checkpoints;
- verl checkpoints.

Continuation from legacy Trainer optimizer state or legacy custom GRPO state
is not supported.

Common CLI semantics are retained:

- all stages support `--config`;
- data generation and evaluation support `--limit`;
- training supports `--max-steps` and `--resume-from-checkpoint`;
- configured paths may be repository-relative.

Script names and configuration schemas do not need to match V1.

## 4. Directory and Module Design

```text
post_train_v2/
  configs/
    common/
    distributed/
    generation/
    sft/
    dpo/
    grpo/
  scripts/
    data/
    generation/
    sft/
    dpo/
    grpo/
    eval/
    pipeline/
  src/
    countdown/
    config/
    artifacts/
    data/
    generation/
    distributed/
    training/
    evaluation/
    rewards/
    tracking/
  verl/
    configs/
    data/
    rewards/
    launch/
    export/
  tests/
    unit/
    distributed/
    integration/
    gpu/
  data/
  outputs/
  docs/
    runbooks/
    superpowers/
      specs/
      plans/
```

Module rules:

- `scripts/` parses CLI arguments, loads configuration, and calls `src/`.
- `src/` must not depend on the current working directory.
- `src/rewards/` is framework-neutral and does not import verl.
- `verl/rewards/` is a thin adapter around the framework-neutral reward.
- Full SFT, LoRA, and RFT reuse one supervised training core.
- Teacher, RFT, and DPO generation reuse one dual-worker generation layer.
- Rejected DPO category names remain unchanged.
- Every top-level functional directory receives a concise README.
- V2 does not copy or patch the verl source tree.

## 5. End-to-End Data Flow

```text
raw_train.parquet / raw_test.json
  -> normalized source JSONL
  -> stratified val_200 and fixed_eval_50
  -> remaining training candidates
  -> Qwen3-8B teacher accepted pool of the earliest 20k correct candidates
  -> stratified SFT 8k and GRPO 4k
  -> RFT accepted data
  -> approximately 6k DPO pairs
  -> verl train and validation Parquet
```

All prompts use the common Countdown prompt and chat-message builders.
Thinking mode is disabled for the teacher and target model. Generation is
limited to 256 new tokens.

The validation split is frozen before Teacher generation:

- sample `val_200` from the complete normalized and solvable source set by
  stratifying on number count and solver-expression complexity;
- remove all 200 validation IDs from every training-candidate pool;
- sample the fixed evaluation set of 50 from `val_200` using the same
  stratification dimensions;
- preserve the selected IDs and their source order in the manifest.

### 5.1 Canonical Schemas

Normalized source records contain:

- `id`;
- `source_index`;
- `numbers`;
- `target`;
- `gold_expr`;
- `prompt`;
- `bucket`.

SFT and RFT records add:

- complete `response`;
- validation result;
- provenance metadata.

DPO records contain:

- `prompt`;
- `chosen`;
- `rejected`;
- `rejected_category`;
- `generation_route`;
- provenance metadata.

verl records contain:

- `data_source`;
- `prompt` as a list of chat messages;
- `ability`;
- `reward_model`;
- `extra_info`.

The verl ground truth is a structured Arrow value:

```json
{
  "numbers": [1, 2, 3, 4],
  "target": 24
}
```

`gold_expr` is diagnostic and is used for bucketing. It is not the target
string for reward comparison.

### 5.2 Manifest V2

Every data or model artifact has a unified JSON manifest with:

- `schema_version`;
- `artifact_type`;
- `stage`;
- stable `artifact_id`;
- creation time;
- artifact file hashes, counts, and field schema;
- parent artifact IDs and SHA-256 hashes;
- complete configuration snapshot and configuration hash;
- model path and model fingerprint when applicable;
- global seed and seed-derivation version;
- Git revision and runtime dependency versions;
- generation shard, resume, and merge metadata in `stage_metadata`.

Stages extend `stage_metadata` rather than inventing independent manifest
formats.

Duplicate IDs, incompatible schemas, mismatched source hashes, or mismatched
configuration fingerprints cause an immediate failure. Formal files are
published through same-directory atomic replacement.

Compatibility readers may consume legacy records, but V2 writers emit only
the V2 schemas.

## 6. Reproducibility Contract

The default global seed is `42`, configurable per run.

Data splitting, stratified sampling, fixed validation selection, and fixed
evaluation selection must be exactly reproducible.

Offline Teacher, RFT, and DPO per-sample generation seeds are stably derived
from:

```text
global seed + stage name + sample ID + rollout index
```

vLLM 0.9.1 offline generation must pass one `SamplingParams` object per
prompt so that each request receives its derived seed. This keeps a sample's
random stream independent of worker assignment and batching.

Python, NumPy, PyTorch, Trainer, and DistributedSampler receive documented
derived seeds. Checkpoints restore supported RNG state and the training data
position.

verl 0.6.0 vLLM rollout uses an engine-level seed rather than the offline
per-request seed contract. GRPO therefore guarantees deterministic dataset
order and a recorded run-level rollout seed only under the same worker
topology. It does not promise that changing Ray placement, rollout batching,
or the GPU topology preserves sample-level generations.

The project does not promise bitwise equality across CUDA kernels, Flash
Attention, or vLLM. It promises stable ordering and statistically
reproducible execution under the same configuration and topology.

The manifest records the seed rule and actual process and GPU topology.

## 7. Dual-GPU Offline Generation

Teacher, RFT, and DPO generation use two independent TP1 vLLM workers, one
Qwen3-8B instance per GPU.

The shared generation system provides:

- deterministic assignment by stable record ID;
- isolated per-worker cache and output directories;
- one chat conversation per prompt;
- per-prompt `SamplingParams` with the stable derived seed;
- `enable_thinking=false`;
- batch-level atomic progress publication;
- per-record deduplication;
- independent shard manifests;
- deterministic merge back to original input order;
- resume rejection when model, generation, input, or schema fingerprints
  differ;
- isolated shard rebuild after corruption.

Teacher generation rolls out each training candidate once. Work is dispatched
in ordered waves. A wave is complete only after both workers have returned
all assigned positions, after which responses are merged by original source
position and validated.

The stopping rule is based on the completed source-order prefix, not worker
completion time. Generation may stop only when that prefix contains at least
20,000 correct responses. The accepted pool is the earliest 20,000 correct
records in original candidate order. Later completed records are not allowed
to displace earlier records.

If the candidate source is exhausted before 20,000 correct responses are
accepted, the stage writes a partial failure manifest containing the actual
accepted count and processed source range, exits non-zero, and does not
publish a complete accepted-pool artifact.

## 8. Supervised Training Stack

Full SFT, LoRA, and RFT training use:

- Transformers Trainer;
- `torchrun --standalone --nproc_per_node=2`;
- one process per GPU;
- DDP, not `device_map="auto"`;
- BF16;
- `attn_implementation="flash_attention_2"`;
- cosine learning-rate scheduling;
- 3 percent warmup;
- gradient checkpointing enabled by default and configurable per stage;
- complete response supervision with prompt-token labels masked;
- maximum total sequence length of 256;
- `logging_strategy="steps"`;
- `logging_steps=1`;
- `logging_first_step=true`;
- synchronous main-rank fixed evaluation every 100 optimizer steps.

Both ranks enter a barrier before evaluation. Rank 0 evaluates and writes
artifacts while rank 1 waits. Both ranks enter a second barrier before
training resumes.

Rank 0 must not execute generation through the DDP wrapper while the other
rank is waiting. It evaluates with the unwrapped underlying module obtained
through the Trainer or Accelerate unwrapping API, under inference mode, with
the previous train/eval state restored afterward. The Level 1 distributed
test must fail if evaluation invokes the DDP wrapper's forward path.

### 8.1 Effective Batch Preservation

The first distributed baseline preserves the old effective global batches:

| Stage | Per-device micro batch | Accumulation | World size | Global batch |
| --- | ---: | ---: | ---: | ---: |
| Full SFT | 4 | 2 | 2 | 16 |
| LoRA SFT | 8 | 2 | 2 | 32 |
| RFT | 4 | 2 | 2 | 16 |
| DPO | 4 | 2 | 2 | 16 |

Larger batches are separate performance experiments and do not alter the
compatibility baseline.

### 8.2 Checkpoints and Model Selection

Trainer checkpoints contain model or adapter state, optimizer, scheduler,
Trainer state, global step, and supported RNG state.

Periodic checkpoints are saved every 100 optimizer steps and only the latest
two are retained. `best/` and `final/` are retained independently.

The final optimizer step always triggers checkpoint publication, fixed-50
evaluation, and final export even when it is not divisible by 100. If the
final step is also a periodic step, the actions run once.

The best checkpoint is selected by:

1. highest fixed-50 `eval/accuracy`;
2. highest `eval/format_rate`;
3. earlier optimizer step.

Full SFT, RFT, and DPO `best/` and `final/` are directly loadable with
`AutoModelForCausalLM.from_pretrained()`.

LoRA `best/` remains an adapter. A separate explicit merge/export entrypoint
produces a full model without modifying the adapter artifact.

## 9. RFT Design

RFT has separate rollout and training model paths:

- `rollout_model_path`: Qwen3-8B base teacher;
- `train_model_path`: Qwen3-0.6B base.

The teacher performs four rollouts for each prompt in the SFT 8k set using:

- temperature `0.7`;
- top-p `0.95`;
- maximum 256 new tokens;
- thinking disabled.

Only responses accepted by the exact Countdown validator are retained.
For deduplication, complete responses are normalized only by converting line
endings to LF and stripping leading and trailing whitespace from the complete
response. Exact normalized duplicates within the same question are removed.
Responses with the same final expression but different normalized complete
response text remain distinct.

At most two accepted responses are retained per prompt. When more than two
normalized, distinct, correct responses remain for one prompt, the two with
the earliest `rollout_index` are retained. The accepted pool is then
stratified by number count and solver-expression complexity. The stage does
not lower validation standards or synthesize records to reach a fixed size.
Its manifest records the actual accepted count and question coverage.

RFT trains Qwen3-0.6B base so it remains an independently comparable
rejection-sampling baseline.

## 10. DPO Design

### 10.1 Pair Construction

DPO uses the SFT 8k set and targets approximately 6,000 filtered pairs.

`chosen` is the correct complete SFT response for the same question.
Rejected candidates are generated through:

- 50 percent forced-wrong teacher instructions;
- 50 percent high-temperature Qwen3-8B rollouts.

The final stratified targets are:

| Category | Target fraction |
| --- | ---: |
| `wrong_value` | 70% |
| `number_mismatch` | 15% |
| `invalid_expression` | 10% |
| `missing_answer_tag` | 3% |
| `truncated` | 2% |

Within each category, generation routes are balanced when supply permits.
Candidate ordering within a category and route is determined by a stable
seeded hash of the question ID and candidate ID.

Selection first fills each category's integer quota. All deficits are then
filled from the remaining valid candidates in the strict global priority
order:

```text
wrong_value
  -> number_mismatch
  -> invalid_expression
  -> missing_answer_tag
  -> truncated
```

This second pass may use lower-priority categories only after all remaining
higher-priority candidates have been exhausted. If fewer than 6,000 valid
candidates exist in total, the output is smaller and the manifest records the
unfilled count. Mathematical and structural filters are never relaxed merely
to reach the target count.

By default each question contributes at most one DPO pair.

### 10.2 Training

DPO uses:

- Full SFT Qwen3-0.6B `best/` as the trainable model;
- TRL 0.19.1 `DPOTrainer`;
- TRL's implicit frozen reference-model copy;
- BF16 and Flash Attention 2;
- gradient checkpointing;
- learning rate `5e-7`;
- beta `0.05`;
- 3 percent warmup;
- cosine scheduling;
- one epoch;
- maximum sequence length 256;
- global batch 16.

Reference-log-probability precomputation may be added as an optional fallback
if remote memory measurements require it. It is not the baseline.

## 11. GRPO Design

### 11.1 Data Conversion

The fixed stratified GRPO 4k JSONL set is converted to train Parquet. The
fixed 50-record evaluation set is converted separately for periodic native
verl validation. The complete `val_200` remains a final common-evaluation
dataset and is not used for periodic GRPO validation.

The converter:

- preserves IDs and prompt text;
- writes chat-message prompts;
- writes structured `numbers` and `target` ground truth;
- retains bucket and gold expression in `extra_info`;
- validates duplicate IDs, row counts, Arrow schema, and hashes;
- writes a Manifest V2 conversion record.

### 11.2 Runtime Configuration

GRPO uses:

- Full SFT `best/` as actor initialization;
- verl 0.6.0;
- `actor_rollout_ref.actor.strategy=fsdp2`;
- vLLM 0.9.1 for rollout;
- one node and two GPUs;
- no critic;
- no learned reward model;
- no reference model;
- `algorithm.adv_estimator=grpo`;
- `actor_rollout_ref.actor.use_kl_loss=false`;
- `algorithm.use_kl_in_reward=false`;
- KL coefficient and reported project KL equal to zero;
- BF16;
- gradient checkpointing enabled initially;
- `data.train_batch_size=4`;
- `actor_rollout_ref.actor.ppo_mini_batch_size=4`;
- `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2`;
- `actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2`;
- `actor_rollout_ref.rollout.tensor_model_parallel_size=1`;
- `actor_rollout_ref.rollout.temperature=1.0`;
- `actor_rollout_ref.rollout.top_p=0.95`;
- `actor_rollout_ref.rollout.n=4`;
- 16 trajectories per trainer iteration;
- `data.max_prompt_length=256`;
- `data.max_response_length=256`;
- `data.apply_chat_template_kwargs.enable_thinking=false`;
- `trainer.total_epochs=1`;
- deterministic epoch shuffle;
- a recorded engine-level rollout seed for the fixed two-GPU topology;
- `actor_rollout_ref.actor.ppo_epochs=2` for two actor-update passes over each
  rollout batch;
- no rollout reuse across trainer iterations.

One GRPO trainer iteration consumes one four-prompt batch, produces four
responses per prompt, and performs the configured two actor-update epochs.

The converted Parquet and GRPO launch configuration must be exercised by a
tokenization fixture proving that Qwen's thinking mode is disabled and that
the rendered prompt ends in the expected generation prompt.

### 11.3 Reward

The framework-neutral reward:

1. checks for a complete `<answer>...</answer>` span;
2. extracts the expression;
3. parses only permitted operators and parentheses;
4. verifies the exact input-number multiset;
5. evaluates using exact rational arithmetic;
6. compares the result exactly with the target.

Fractional intermediate values are valid.

Rewards are:

- `0.2` for a complete extractable answer tag;
- `1.0` for a valid expression using the numbers correctly and reaching the
  target;
- `1.2` total for a fully correct response.

The adapter returns a score dictionary with diagnostics including:

- `score`;
- `format_ok`;
- `answer_correct`;
- `error`, which is `null` for a correct response or one of
  `missing_answer_tag`, `invalid_expression`, `number_mismatch`, and
  `wrong_value`;
- extracted expression;
- `value`, serialized as a reduced `"numerator/denominator"` string when
  evaluation succeeds, including denominator `1` for integers.

All reward diagnostics must be JSON-, Arrow-, and W&B-serializable primitive
values.

Zero-standard-deviation groups are retained. They produce zero effective
group-relative advantage and are not resampled.

### 11.4 Metrics and Saving

Every GRPO trainer iteration records:

- reward mean and standard deviation;
- group reward standard deviation;
- fraction of zero-standard-deviation groups;
- all-correct and all-wrong group fractions;
- accuracy and format rate;
- policy loss;
- KL as zero/disabled;
- optional entropy;
- response-length statistics;
- truncated response count;
- rollout and token throughput;
- bucket-level reward and accuracy.

Every 100 trainer iterations:

- run native verl validation on the fixed 50-record validation Parquet and
  dump complete generation records to a step-addressed JSONL file;
- save a native verl continuation checkpoint containing actor model,
  optimizer, scheduler, extra state, global step, supported RNG state, and
  data position.

The initial launch configuration sets `trainer.test_freq=100`,
`trainer.save_freq=100`, `trainer.log_val_generations=50`, and a
V2-controlled `trainer.validation_data_dir`. Native checkpoint pruning is
disabled during training so every checkpoint referenced by a validation dump
remains available for post-training selection.

The final trainer iteration performs the same native validation and
checkpoint actions even when it is not divisible by 100. Duplicate work is
suppressed when the final iteration is already a periodic boundary.

All periodic continuation checkpoints are retained until post-training model
selection finishes so an earlier best step cannot be pruned. The
post-training selector recomputes `eval/accuracy` and `eval/format_rate` from
each native validation JSONL dump using the common evaluator rules and
selects:

1. highest fixed-50 `eval/accuracy`;
2. highest `eval/format_rate`;
3. earlier optimizer step.

After selection, V2 invokes the stock `verl.model_merger` entrypoint to merge
the selected best actor checkpoint and the final actor checkpoint into
directly loadable Hugging Face `best/` and `final/` exports. It then retains
the latest two continuation checkpoints plus the selected best continuation
checkpoint when that checkpoint is distinct. All other periodic continuation
checkpoints may be removed only after both exports pass direct Transformers
loading and their manifests are published.

Native validation or continuation-checkpoint failure stops training. A
post-training selection, merge, direct-load, or common-evaluation failure is
recorded as a retryable post-processing failure and does not invalidate the
native continuation checkpoints.

## 12. Common Evaluation

The common evaluator supports complete Hugging Face models and LoRA adapters.
It applies the Qwen chat template with thinking disabled and generates no
more than 256 new tokens. Fixed evaluation uses deterministic decoding.

The final evaluation matrix includes:

- Qwen3-0.6B base;
- Full SFT `best/`;
- LoRA SFT `best/` adapter;
- RFT `best/`;
- DPO `best/`;
- GRPO `best/`;
- optional Qwen3-8B teacher baseline.

Each model is evaluated on:

- the complete fixed `val_200`;
- the solved test set derived from `raw_test.json`.

Outputs include:

- accuracy;
- format rate;
- response-length statistics;
- truncation count and rate;
- complete sample-level traces.

Each stage's `final/` receives a direct-load test and a fixed-50 evaluation,
but not a duplicate full evaluation by default.

## 13. W&B and Logging

W&B is optional and configuration-driven.

Each training stage owns one logical run. Distributed helper processes do not
create additional runs. The default project is
`countdown-post-train-v2`. Grouping uses an experiment-chain ID, and run names
receive an automatic timestamp and short Git-revision suffix.

Trainer stages log loss at every optimizer step through
`logging_strategy="steps"`, `logging_steps=1`, and
`logging_first_step=true`. GRPO logs all metrics listed in Section 11.4 at
every trainer iteration. Native fixed-50 validation metrics and sampled
generations are logged every 100 steps and at the final step. Complete
fixed-50 generation traces are written locally, while W&B records the
configured validation samples and the local trace path.

W&B does not upload complete model checkpoints or datasets as Artifacts by
default. It records configuration, local paths, artifact IDs, and manifest
hashes.

## 14. Failure Recovery

Generation stages:

- publish each completed inference batch atomically;
- skip already completed stable IDs during resume;
- reject resume on input, configuration, model, or schema mismatch;
- validate shard completeness, uniqueness, and coverage before merge;
- permit isolated rebuild of a damaged shard.

Trainer stages resume through `--resume-from-checkpoint` with the complete
V2 Trainer state.

verl resumes from its native continuation checkpoint, restoring actor,
optimizer, scheduler, global step, supported RNG state, and data position.
GRPO post-processing can be rerun independently from the retained native
validation dumps and continuation checkpoints.

Temporary files are created beside their final destinations and promoted by
atomic replacement.

## 15. Verification Gates

### Level 0: CPU Unit Tests

Validate:

- schemas and Manifest V2;
- prompts, solver, exact validation, bucketing, and sampling;
- reward components and diagnostics;
- configuration parsing and path behavior;
- conversion and metric aggregation.

### Level 1: Two-Process Distributed Tests

Use CPU/Gloo to validate:

- rank and world-size handling;
- deterministic distributed sampling;
- barriers;
- main-rank-only side effects;
- evaluation through the unwrapped module rather than the DDP wrapper;
- resume state;
- failure propagation.

### Level 2: GPU Smoke Tests

On the target two-GPU host:

- run small Teacher, RFT, and DPO generation jobs;
- perform one or two optimizer steps for Full SFT, LoRA, RFT, and DPO;
- perform one GRPO iteration with four prompts, four rollouts each, and two
  actor update epochs;
- verify Flash Attention 2 for Transformers training;
- verify Qwen thinking mode is disabled in verl tokenization;
- verify one W&B run per stage;
- save and resume native checkpoints;
- select a best GRPO checkpoint from native validation dumps;
- merge best and final GRPO actors with `verl.model_merger` and directly load
  the resulting Hugging Face exports.

### Level 3: End-to-End Acceptance

Run a small complete pipeline covering:

- data construction;
- Teacher acceptance;
- SFT, DPO, and GRPO;
- interrupted-run recovery;
- manifest lineage;
- output isolation;
- common evaluation and sample traces.

Full 20k, 8k, 6k, and 4k production runs are operational workloads, not
automated acceptance tests.

## 16. Documentation and Delivery

Each implementation phase must update:

- its functional directory README;
- a remote execution runbook;
- exact launch commands;
- expected inputs and outputs;
- checkpoint resume commands;
- relevant verification commands.

No phase is considered complete until its applicable verification level
passes and the produced model or data artifact satisfies Manifest V2.
