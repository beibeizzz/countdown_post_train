# Countdown Post-Training Project Design

## Goal

Merge the existing medical-question SFT scripts and Countdown RL scripts into one Countdown math post-training project under `post_train/`.

The target model is Qwen3-0.6B. The teacher model is Qwen3-8B. The training pipeline is:

1. Build a reproducible Countdown data warehouse.
2. Run teacher-filtered SFT data construction.
3. Train and evaluate SFT models with full-parameter SFT, LoRA SFT, and rejection-sampling fine-tuning.
4. Use the best full-parameter SFT model as the base for DPO and GRPO.

The task format is Countdown arithmetic: given `nums` and `target`, generate an arithmetic expression that uses each number exactly once and evaluates to `target`.

## Model Roles

- `teacher_model`: Qwen3-8B. Used for teacher-filtered SFT data generation, default RFT data generation, and teacher-style response production.
- `target_model`: Qwen3-0.6B. This is the model being trained.
- `sft_base_model`: Qwen3-0.6B before SFT.
- `rl_base_model`: the best full-parameter SFT checkpoint, used as the base for DPO and GRPO unless explicitly overridden.
- `dpo_data_model`: Qwen3-8B teacher model. Used to construct DPO rejected responses through forced-wrong prompting and high-temperature rollout.
- `grpo_rollout_model`: the SFT-trained Qwen3-0.6B checkpoint. Used for GRPO rollout and synchronized with the training weights.
- `grpo_train_model`: the SFT-trained Qwen3-0.6B checkpoint. Updated by GRPO policy optimization.

Default local model paths:

- `teacher_model_path`: `post_train/model/qwen/qwen3-8b`
- `target_model_path`: `post_train/model/qwen/qwen3-0.6b`

## Core Constraints

- Training target model: Qwen3-0.6B.
- Teacher model: Qwen3-8B.
- Runtime environment target: Ubuntu 22.04, Python 3.12, PyTorch 2.8.0, vLLM, flash-attn, TRL.
- Both teacher and target model generation must disable thinking mode.
- Generation max new tokens is 256 for teacher inference, SFT evaluation, DPO data construction, and GRPO rollout unless a debug override is provided.
- Training prompts are English.
- SFT trains on the full accepted teacher response, not only the `<answer>...</answer>` span.
- Evaluation correctness is always based on extracting the final expression from `<answer>...</answer>` and validating it with the solver.
- All training uses cosine learning-rate scheduling.
- Every 100 training steps, evaluate on the same fixed 50-example subset from the fixed validation set and save the sample generations.
- Prompt construction must go through `src/countdown/prompts.py`.
- Generation wrappers must go through `src/countdown/generation.py`.

## Repository Layout

The consolidated project lives in `post_train/`:

```text
post_train/
  configs/
  data/
    raw/
    processed/
    teacher_rollouts/
    sft/
    dpo/
    grpo/
    eval/
  outputs/
    sft/
      full/
      lora/
      rft/
    dpo/
    grpo/
  scripts/
    data/
    sft/
    dpo/
    grpo/
    eval/
  src/
    countdown/
      data_schema.py
      prompts.py
      solver.py
      validation.py
      bucketing.py
      sampling.py
      generation.py
      eval.py
      logging.py
  docs/
```

The existing `sft/` and `rlhf/` folders remain as reference material. New code should not keep the old medical-question schema.

## Config Files

Each pipeline stage has its own config file under `post_train/configs/`. Defaults should be conservative and easy to override from the command line.

```text
post_train/configs/
  data_build.yaml
  teacher_rollout.yaml
  sft_full.yaml
  sft_lora.yaml
  rft.yaml
  dpo_data.yaml
  dpo_train.yaml
  grpo.yaml
  eval.yaml
```

### Conservative Defaults

`data_build.yaml`:

```yaml
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

`teacher_rollout.yaml`:

```yaml
model_path: post_train/model/qwen/qwen3-8b
backend: vllm
batch_size: 64
max_new_tokens: 256
temperature: 0.2
top_p: 0.95
enable_thinking: false
stop_after_accepted: 20000
```

`sft_full.yaml`:

```yaml
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

`sft_lora.yaml`:

```yaml
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

`rft.yaml`:

```yaml
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

`dpo_data.yaml`:

```yaml
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

`dpo_train.yaml`:

```yaml
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

`grpo.yaml`:

```yaml
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

`eval.yaml`:

```yaml
val_data: post_train/data/processed/val_200.jsonl
eval_subset: post_train/data/processed/val_eval_50.jsonl
test_data: post_train/data/processed/test_with_solver_answers.jsonl
max_new_tokens: 256
temperature: 0.0
top_p: 1.0
batch_size: 32
enable_thinking: false
```

## Data Warehouse

### Source Data

Inputs:

- `datasets/raw_train.parquet`
- `datasets/raw_test.json`

The source build step reads the train parquet, runs the Countdown solver, and writes a solver-backed JSONL. Unsolvable rows are skipped and reported.

Each source record has:

```json
{
  "id": "train-000001",
  "source_index": 1,
  "numbers": [44, 19, 35],
  "target": 98,
  "gold_expr": "(44+19+35)",
  "bucket": {
    "num_count": 3,
    "expr_depth": 2,
    "expr_len": 10,
    "has_division": false,
    "has_subtraction": false,
    "complexity": "easy"
  }
}
```

The raw test file is converted into a solver-backed internal test set with the same `gold_expr` and bucket metadata. Test gold answers are only for local evaluation and are not used for training.

### Validation Split

The validation set is a fixed-seed stratified random sample of 200 examples from the solver-backed train source.

Rules:

- Use `seed=42` unless explicitly overridden.
- Stratify by `num_count + complexity`.
- Exclude the 200 validation examples from all teacher rollout, SFT, DPO, and GRPO training pools.
- Keep the validation source stable for all experiments.
- Select a second fixed-seed stratified 50-example eval subset from these 200 validation examples.
- Every periodic eval uses this same 50-example subset to reduce metric noise.

### Bucketing

The first implementation uses two bucketing axes:

- `num_count`: 3, 4, or 5 numbers.
- `complexity`: easy, medium, hard, derived from solver expression structure.

Complexity is computed from:

- Expression tree depth.
- Expression length.
- Whether division appears.
- Whether subtraction appears.
- Number count.

Initial policy:

- `easy`: 3 numbers, shallow expression, no division.
- `medium`: moderate expression depth, or subtraction, or 4 numbers.
- `hard`: 5 numbers, division, deep expression, or long expression.

The exact thresholds should be centralized in `src/countdown/bucketing.py` and saved into every dataset manifest.

## Prompt and Response Format

### Base Prompt

The base prompt style is concise English:

```text
Using the numbers [1, 1, 1, 1], create an equation that equals 4.
Use each number exactly once. Only use +, -, *, / and parentheses.
Finally return <answer> equation </answer>.
```

The implementation may add stricter clauses:

```text
Do not use any other numbers. Keep the response concise.
```

The prompt must not demand no reasoning. Some useful reasoning is allowed, but the final answer must be inside `<answer>...</answer>`.

The prompt should not say "division must be exact." The backend validator allows intermediate fractional values as long as the final expression uses the given numbers exactly once and evaluates to the integer target.

### Accepted Response

SFT stores and trains the complete accepted teacher response. A valid response must contain a parseable final answer:

```text
We can combine the numbers by first making 4 and then multiplying.
<answer> (7-3)*(8-2) </answer>
```

During evaluation, only the final extracted expression is scored.

## Teacher-Filtered SFT Data

Teacher generation uses Qwen3-8B through vLLM in original data order after excluding validation records.

Rules:

- One rollout per input due to time constraints.
- Disable thinking mode.
- Max new tokens: 256.
- Batch generation in original order.
- Stop after accumulating 20,000 accepted examples.
- Save both accepted and rejected teacher generations for analysis.

A teacher response is accepted if:

1. `<answer>...</answer>` can be extracted.
2. The expression parses under the project AST validator.
3. The expression uses exactly the provided numbers.
4. The expression uses only `+`, `-`, `*`, `/`, and parentheses.
5. Intermediate fractional values are allowed.
6. The final result equals `target`.

The accepted 20k pool is the shared base pool for SFT sampling, DPO chosen data, and GRPO sampling.

Expression validation uses `fractions.Fraction` so intermediate fractional arithmetic is represented exactly.

## SFT Data

From the 20k accepted teacher pool, build an approximately 8k SFT training set by stratified balanced sampling over `num_count + complexity`.

SFT record schema:

```json
{
  "id": "train-000123",
  "prompt": "...",
  "response": "teacher full response with <answer>...</answer>",
  "numbers": [7, 3, 8, 2],
  "target": 24,
  "gold_expr": "(7-3)*(8-2)",
  "teacher_expr": "(7-3)*(8-2)",
  "bucket": {...},
  "source": "teacher_accepted"
}
```

### SFT Training Modes

1. Full-parameter SFT.
   - Main SFT path.
   - Output becomes the base model for DPO and GRPO.
   - Default hyperparameters: `lr=1e-5`, cosine scheduler, warmup ratio `0.03`, bf16, gradient checkpointing, `max_seq_len=256`.
   - Save periodic checkpoints and the final checkpoint under `post_train/outputs/sft/full/`.

2. LoRA SFT.
   - Comparison and resource-efficient baseline.
   - Save adapters and metrics under `post_train/outputs/sft/lora/`.

3. Rejection-sampling fine-tuning.
   - Use the configured generation base model to sample multiple responses for the same prompt. By default this is the teacher model for data generation experiments, while trained model checkpoints can be used for target-model RFT comparisons.
   - Keep only solver-valid correct responses.
   - Fine-tune on the accepted sampled responses.
   - Save outputs under `post_train/outputs/sft/rft/`.

## DPO

DPO uses the SFT 8k data as chosen candidates.

Chosen:

- The accepted SFT full response for the same prompt.

Rejected:

- Half from strong wrong-answer instruction.
- Half from high-temperature rollout.
- Both rejected-generation paths use Qwen3-8B teacher model for DPO data construction.

Rejected categories:

- A: `format_ok + uses_numbers_ok + wrong_value`
- B: `format_ok + number_mismatch`
- C: `format_ok + invalid_expression`
- D: `missing_answer_tag`
- E: `truncated`

The most valuable rejected category is A: complete answer tags, parseable expression, exact same numbers, but wrong target value. These samples are closest to the chosen response and isolate mathematical correctness.

DPO pair priority:

- First priority: chosen is correct, rejected has complete `<answer>` tags, rejected expression is parseable, rejected uses the same numbers exactly once, and rejected evaluates to the wrong target.
- Second priority: rejected is parseable but has a number mismatch.
- Low-volume fallback: format error or malformed output.

Filtering rules:

- Drop unexpectedly correct rejected samples.
- Drop unusable empty samples.
- Do not let malformed rejected samples dominate the dataset, because otherwise DPO will mostly learn formatting instead of mathematical correctness.
- Cap malformed rejected samples at 10% by default.
- Prefer A-category `wrong_value` samples and target at least 70% A-category when available.
- Save rejected category counts and final category ratios in the DPO manifest.

Expected output after filtering: about 6k DPO pairs.

DPO training outputs are saved under `post_train/outputs/dpo/`.

DPO schema:

```json
{
  "id": "dpo-000001",
  "prompt": "...",
  "chosen": "...",
  "rejected": "...",
  "numbers": [...],
  "target": 24,
  "bucket": {...},
  "rejected_source": "forced_wrong|high_temp_rollout",
  "rejected_error_type": "wrong_value|number_mismatch|invalid_expression|missing_answer_tag|truncated"
}
```

## GRPO

GRPO uses a further stratified selection of about 4k examples from the 20k accepted pool or its source prompts.

Training design:

- Use the SFT-trained Qwen3-0.6B checkpoint with vLLM for parallel rollout.
- Load a separate copy of the same SFT-trained Qwen3-0.6B checkpoint for gradient updates.
- Semi-on-policy synchronization:
  - Roll out with vLLM.
  - Update policy twice per batch.
  - Every 20 steps, save weights and sync them into vLLM.
- Example initial rollout shape: batch of 4 prompts, group size 4.
- GRPO KL coefficient defaults to 0.0.

Reward:

- `format_reward = 0.2` for complete extractable `<answer>...</answer>`.
- `answer_reward = 1.0` for solver-valid correct expression.
- First implementation does not require a length penalty. Token length and truncation are logged.

GRPO metrics:

```json
{
  "step": 100,
  "loss": 0.0,
  "mean_reward": 0.0,
  "reward_std": 0.0,
  "group_reward_std": 0.0,
  "frac_reward_zero_std": 0.0,
  "accuracy": 0.0,
  "format_rate": 0.0,
  "approx_kl": 0.0,
  "entropy": null,
  "avg_gen_tokens": 0.0,
  "max_gen_tokens": 0,
  "truncated_count": 0,
  "rollout_count": 0,
  "learning_rate": 0.0
}
```

Entropy is optional because it can add measurable overhead.

GRPO training outputs are saved under `post_train/outputs/grpo/`.

## Evaluation

Every model stage uses the same evaluator:

- Base Qwen3-0.6B.
- Full-parameter SFT.
- LoRA SFT.
- RFT.
- DPO.
- GRPO.

Every 100 training steps:

- Evaluate on the fixed 50-example subset selected from the fixed 200 validation set.
- Generate with max new tokens 256.
- Save sample-level outputs.
- Save aggregate metrics.

Sample evaluation record:

```json
{
  "step": 100,
  "id": "train-000123",
  "prompt": "...",
  "raw_generation": "...",
  "extracted_expr": "(7-3)*(8-2)",
  "format_ok": true,
  "valid": true,
  "correct": true,
  "error": null,
  "generated_tokens": 42,
  "truncated": false
}
```

Aggregate metrics:

- `accuracy`
- `format_rate`
- `valid_expression_rate`
- `avg_generated_tokens`
- `max_generated_tokens`
- `truncated_count`

## Error Handling and Reproducibility

Every generated dataset writes a manifest:

```json
{
  "name": "sft_train_8k",
  "created_at": "...",
  "seed": 42,
  "input_files": [],
  "output_file": "...",
  "num_records": 8000,
  "bucket_counts": {},
  "model": "Qwen3-8B",
  "generation": {
    "max_new_tokens": 256,
    "thinking": false
  }
}
```

Generation scripts must support resume behavior:

- Do not regenerate existing accepted examples.
- Track processed source IDs.
- Append new accepted and rejected rows safely.
- Write progress summaries periodically.

## Implementation Phases

1. Create shared Countdown utilities in `post_train/src/countdown/`.
2. Build source, validation, test, and bucketed data scripts.
3. Implement teacher vLLM rollout and accepted-pool construction.
4. Implement SFT dataset sampler and training scripts.
5. Implement common evaluator and 100-step eval hooks.
6. Implement DPO data construction and DPO training.
7. Implement GRPO vLLM rollout and semi-on-policy sync training.
8. Add workflow docs and run commands.

## Implemented Clarifications

- RFT has an explicit training entrypoint: `post_train/scripts/sft/train_rft.py --config post_train/configs/rft.yaml`. It reuses the full SFT trainer and maps the `train` section of `rft.yaml` to full-SFT training fields.
- GRPO writes rollout metrics and also runs the fixed 50-example common evaluator every `eval_every_steps` when `eval.yaml` is available.
- Manifest files use the shared `countdown.post_train.manifest.v1` envelope with `manifest_version`, `schema`, `name`, `stage`, and `created_at`, plus stage-specific counts and settings.
- GRPO KL is intentionally fixed to `kl_coeff: 0.0`; nonzero KL fails fast because reference-KL training is not implemented in the first minimal GRPO script.
- GRPO vLLM synchronization is implemented as checkpoint save plus reload attempt. Live in-memory vLLM weight synchronization remains environment-dependent and must be validated in the target Ubuntu/vLLM runtime.
- wandb monitoring is optional and config-driven. Trainer-based SFT, LoRA, RFT, and DPO use `report_to`; GRPO logs metric rows manually. Run names support timestamp suffixes through `run_name_auto_suffix`. The standalone evaluator remains local-only and does not upload to wandb.

## Open Decisions

The current design keeps these parameters configurable in the per-stage config files:

- SFT batch size.
- LoRA rank and target modules.
- RFT number of samples per prompt.
- DPO beta and learning rate.
- GRPO batch size, group size, learning rate, and sync interval.

The first implementation should provide conservative defaults and keep all of these in config files.
