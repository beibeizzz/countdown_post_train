# Countdown Post-Training V2

`post_train_v2` is the planning workspace for rewriting the distributed
post-training pipeline currently implemented under `post_train`.

This directory does not yet contain runnable training code. The first phase
only defines architecture, migration boundaries, execution order, data
contracts, and unresolved decisions.

## Confirmed Scope

- Target model: local Qwen3-0.6B.
- Teacher model: local Qwen3-8B.
- Hardware target: one node with two 40 GB NVIDIA GPUs.
- Full SFT: two-GPU DDP.
- LoRA SFT: two-GPU DDP.
- RFT training: two-GPU DDP.
- DPO training: two-GPU DDP.
- GRPO: verl with FSDP/FSDP2 and vLLM rollout.
- Teacher data generation: vLLM using both GPUs.
- Add a JSONL-to-verl-Parquet conversion layer.
- Add a verl custom rule-reward adapter backed by the existing Countdown
  validator.
- PPO is out of scope.

## Design Boundary

The V2 project will use two execution stacks:

1. Transformers/TRL stack
   - Full SFT, LoRA SFT, RFT training, and DPO.
   - Launched explicitly with `torchrun` or Accelerate on two GPUs.
   - DDP is the default strategy.
   - FSDP2 and DeepSpeed are not default SFT/DPO strategies.

2. verl stack
   - GRPO rollout, reward dispatch, group-relative advantage calculation,
     actor updates, distributed checkpointing, and experiment metrics.
   - vLLM is the rollout engine.
   - No critic is required for GRPO.
   - No learned reward model is required for the Countdown task.

Existing data preparation, prompt construction, solver validation, bucketing,
sampling, and evaluation semantics should be preserved unless a documented
incompatibility requires a change.

## Planned Directory Structure

```text
post_train_v2/
  README.md
  analysis.md
  migration_plan.md
  open_questions.md
  configs/
    common/
    distributed/
    sft/
    dpo/
    grpo/
    generation/
  scripts/
    data/
    generation/
    sft/
    dpo/
    grpo/
    eval/
  src/
    countdown/
    data/
    distributed/
    evaluation/
    rewards/
    tracking/
  verl/
    configs/
    data/
    rewards/
    launch/
  tests/
    unit/
    integration/
  docs/
```

The empty subdirectories are placeholders. No training entrypoint in this
tree should be treated as implemented until a later implementation plan is
approved.

## Intended Pipeline

```text
raw Countdown data
  -> solver-backed normalized source data
  -> fixed validation and evaluation sets
  -> Qwen3-8B teacher generation
  -> accepted 20k pool
  -> stratified SFT 8k and GRPO 4k sets
  -> two-GPU DDP SFT variants
  -> DPO pair generation and two-GPU DDP DPO
  -> JSONL-to-verl-Parquet conversion
  -> verl GRPO with rule reward and vLLM rollout
  -> common fixed-set evaluation
```

## Documents

- `analysis.md`: inventory and behavioral analysis of the existing project.
- `migration_plan.md`: recommended distributed and verl migration design.
- `environment.md`: pinned runtime baseline and remote installation checks.
- `open_questions.md`: decisions that must be confirmed before core
  implementation.

## Runtime Baseline

V2 uses an isolated Python 3.11.15 uv environment pinned by
`configs/environment/runtime-cu128.json`. The core runtime is PyTorch 2.7
cu128, Flash Attention 2.7.4.post1, vLLM 0.9.1, verl 0.6.0, Transformers
4.53.2, and TRL 0.19.1.

All Transformers model-loading paths used by Full SFT, LoRA, RFT, DPO,
legacy GRPO, and evaluation require Flash Attention 2 and BF16. vLLM uses its
own attention backend. See `docs/environment_setup.md` for the remote
installation and two-level acceptance process.
