# Next Full Workflow

This document defines the execution order after the current environment and
Flash Attention loader work. It deliberately separates validation of the
existing `post_train` implementation from production use of the future
distributed `post_train_v2` implementation.

## Current Boundary

- `post_train` contains the existing data, SFT, RFT, DPO, legacy GRPO, and
  evaluation workflows.
- `post_train_v2` currently contains the environment baseline, migration
  design, acceptance scripts, and tests.
- Two-GPU DDP training and verl GRPO are not implemented in `post_train_v2`
  yet. Do not start a production run from V2 until those entrypoints pass the
  acceptance gates below.

## Phase 0: Repository Transfer

1. Review the working tree and commit the source, tests, configuration, and
   documentation changes.
2. Push the repository and pull it on the remote GPU host.
3. Keep model weights, raw datasets, generated outputs, checkpoints, and
   manually downloaded wheels outside Git unless explicitly tracked.

## Phase 1: Rebuild the Remote Environment

Follow `post_train_v2/docs/environment_setup.md`.

1. Create the isolated Python 3.11.15 uv environment.
2. Upload the exact PyTorch cu128 and Flash Attention wheels described in the
   runbook.
3. Verify wheel filenames and SHA256 hashes before installation.
4. Run `uv lock`, `uv sync --frozen`, and `uv pip check`.
5. Commit the generated `uv.lock` only after dependency resolution succeeds
   on the remote host.

## Phase 2: Level 1 Runtime Acceptance

Run every Level 1 gate from the environment runbook:

1. Static environment tests.
2. Runtime and CUDA visibility check.
3. Direct Flash Attention forward and backward smoke test.
4. Transformers Flash Attention model smoke test.
5. Existing `post_train` model loader smoke test.
6. Two-rank NCCL smoke test.
7. vLLM tensor-parallel size 1 and 2 smoke tests.
8. Dual-engine teacher generation smoke test.
9. TRL and PEFT training smoke test.
10. Full-model and LoRA-adapter evaluation loader smoke tests.

Stop on the first failure. Do not begin data generation or training with a
partially accepted environment.

## Phase 3: Validate Source Data

Place the source files where the existing configuration expects them:

- `datasets/raw_train.parquet`
- `datasets/raw_test.json`

Build and validate the normalized source:

```bash
python post_train/scripts/data/build_source.py \
  --config post_train/configs/data_build.yaml
```

Inspect the generated manifest, row counts, validation split, fixed 50-example
evaluation subset, solver status, number usage, and expression complexity
fields before continuing.

## Phase 4: Build the Teacher-Accepted Pool

Run Qwen3-8B generation with thinking disabled and the shared prompt and
generation modules:

```bash
python post_train/scripts/data/build_teacher_pool.py \
  --config post_train/configs/teacher_rollout.yaml
```

Resume until the accepted pool reaches 20,000 correct examples. Preserve
generation order, manifests, rejected counts, and solver validation details.

## Phase 5: Build Training Splits

```bash
python post_train/scripts/data/build_sft_splits.py \
  --config post_train/configs/data_build.yaml
```

Verify the stratified outputs:

- fixed validation set: 200 examples;
- fixed periodic evaluation set: 50 examples;
- SFT training set: about 8,000 examples;
- GRPO candidate set: about 4,000 examples.

## Phase 6: Existing-Pipeline Smoke Runs

Before rewriting distributed entrypoints, run short existing-pipeline jobs:

1. Full SFT for two optimizer steps.
2. LoRA SFT for two optimizer steps.
3. RFT rollout/data construction on a small limit, then two training steps.
4. DPO pair construction on a small limit, then two training steps.
5. Legacy GRPO for the minimum viable update only if it remains needed for
   behavioral comparison.
6. Evaluate every produced full model or adapter with
   `post_train/scripts/eval/evaluate_model.py`.

These runs verify contracts and loader behavior. They are not the final
two-GPU production runs.

## Phase 7: Implement V2 Distributed SFT and DPO

Implement in this order:

1. Shared distributed bootstrap, rank-safe logging, checkpointing, and fixed
   evaluator.
2. Full SFT with two-rank DDP.
3. LoRA SFT with two-rank DDP and adapter-aware evaluation.
4. RFT training with two-rank DDP.
5. DPO training with two-rank DDP.
6. Two-GPU teacher and rejected-response generation sharding.

Each entrypoint must support a two-step smoke configuration before production
hyperparameters are enabled.

## Phase 8: Run SFT, RFT, and DPO

1. Train Full SFT as the primary model.
2. Train LoRA SFT as the parameter-efficient comparison.
3. Build RFT data with repeated base-model sampling and train the RFT variant.
4. Evaluate all variants on the same fixed validation and test inputs.
5. Use the selected Full SFT checkpoint as the base for DPO and GRPO.
6. Build DPO rejected responses with the teacher model, preserving the current
   semantic rejection categories and prioritizing parseable wrong-value
   negatives.
7. Train and evaluate DPO on the filtered pair set.

## Phase 9: Implement verl GRPO

1. Add the JSONL-to-verl-Parquet converter.
2. Add the custom Countdown reward adapter using the existing validator.
3. Add verl 0.6.0 configuration and launch entrypoints.
4. Configure FSDP/FSDP2 actor training and vLLM rollout for two GPUs.
5. Keep KL coefficient at zero.
6. Log reward, reward standard deviation, group reward standard deviation,
   fraction of zero-standard-deviation groups, accuracy, format reward, loss,
   optional entropy, response length, and truncation count.

## Phase 10: Level 2 Training Acceptance

Run a minimal end-to-end update:

- at least two questions;
- at least two rollouts per question;
- one optimizer update;
- reward and correctness verification;
- checkpoint save and reload;
- rank-safe W&B logging;
- explicit GPU placement and memory checks.

Production GRPO is blocked until this gate passes.

## Phase 11: Production GRPO

Train on the approximately 4,000-example stratified GRPO set using the
confirmed group and question batch sizes, 256-token response limit, thinking
disabled, periodic checkpoint synchronization, and the fixed 50-example
evaluator.

## Phase 12: Final Evaluation and Export

1. Compare Full SFT, LoRA SFT, RFT, DPO, and GRPO with identical evaluation
   inputs and solver rules.
2. Confirm that every final full checkpoint loads with
   `AutoModelForCausalLM.from_pretrained`.
3. Evaluate LoRA adapters with the adapter-aware loader; optionally export a
   separately named merged checkpoint.
4. Archive manifests, resolved configs, dependency lock, W&B run identifiers,
   metrics, checkpoints, and representative generation traces.
