# Countdown Post-Training V2

`post_train_v2` is the staged rewrite of the distributed post-training
pipeline currently implemented under `post_train`.

The runtime environment, V2 Countdown/data foundations, dual-GPU Teacher
generation, deterministic split builders, common evaluation, Manifest V2,
rank-aware tracking utilities, two-GPU supervised SFT entrypoints, LoRA SFT,
RFT data/training entrypoints, DPO data/training entrypoints, and the verl
GRPO conversion/reward/launch/export entrypoints are implemented.

The authoritative core design is:

- `docs/superpowers/specs/2026-06-15-post-train-v2-core-development-design.md`

The reviewed implementation sequence starts at:

- `docs/superpowers/plans/2026-06-15-post-train-v2-core-development-master.md`

Older planning documents are retained as historical analysis. When a
numerical default or status statement conflicts with the core design, the
core design takes precedence.

## Confirmed Scope

- Target model: local Qwen3-0.6B.
- Teacher model: local Qwen3-8B.
- Hardware target: one node with two 40 GB NVIDIA GPUs.
- Full SFT: two-GPU DDP.
- LoRA SFT: two-GPU DDP.
- RFT training: two-GPU DDP.
- DPO training: two-GPU DDP.
- GRPO: verl with FSDP2 and vLLM rollout.
- Teacher data generation: vLLM using both GPUs.
- Add a JSONL-to-verl-Parquet conversion layer.
- Add a verl custom rule-reward adapter backed by the existing Countdown
  validator.
- PPO is out of scope.

## Design Boundary

The V2 project will use two execution stacks:

1. Transformers/TRL stack
   - Full SFT, LoRA SFT, RFT training, and DPO.
   - Launched explicitly with `torchrun` on two GPUs.
   - Accelerate is used as an internal distributed utility where needed, not
     as an alternate public launcher.
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

Some later training subdirectories remain placeholders. An entrypoint is
runnable only when its phase plan and README mark it implemented and its
applicable verification gate has passed.

## Implemented Phase 1 Flow

Run these commands from the repository root:

```bash
python post_train_v2/scripts/data/build_source.py
python post_train_v2/scripts/data/build_splits.py \
  --config post_train_v2/configs/data/build_splits.yaml validation
python post_train_v2/scripts/generation/build_teacher_pool.py \
  --config post_train_v2/configs/generation/teacher_rollout_2gpu.yaml
python post_train_v2/scripts/data/build_splits.py \
  --config post_train_v2/configs/data/build_splits.yaml accepted
python post_train_v2/scripts/eval/evaluate_model.py \
  --model-path post_train_v2/outputs/sft/full/best
```

The Teacher command requires the pinned remote GPU environment and two
working GPUs. Detailed artifact checks and resume commands are in
`docs/runbooks/data_and_evaluation.md`.

## Implemented Phase 2 Flow

After Phase 1 has produced `sft_train_8k.jsonl` and `eval_50.jsonl`, run:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  post_train_v2/scripts/sft/train_full.py \
  --config post_train_v2/configs/sft/full.yaml

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  post_train_v2/scripts/sft/train_lora.py \
  --config post_train_v2/configs/sft/lora.yaml

CUDA_VISIBLE_DEVICES=0,1 python post_train_v2/scripts/sft/build_rft_data.py \
  --config post_train_v2/configs/sft/rft_rollout.yaml

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  post_train_v2/scripts/sft/train_rft.py \
  --config post_train_v2/configs/sft/rft_train.yaml
```

Smoke commands and output checks are in
`docs/runbooks/supervised_and_rft.md`.

## Implemented Phase 3 Flow

After Phase 2 Full SFT has produced `post_train_v2/outputs/sft/full/best`,
run:

```bash
CUDA_VISIBLE_DEVICES=0,1 python post_train_v2/scripts/dpo/build_dpo_data.py \
  --config post_train_v2/configs/dpo/build.yaml

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  post_train_v2/scripts/dpo/train_dpo.py \
  --config post_train_v2/configs/dpo/train.yaml
```

Smoke commands and output checks are in `docs/runbooks/dpo.md`.

## Implemented Phase 4 Flow

After Phase 2 Full SFT has produced `post_train_v2/outputs/sft/full/best` and
the GRPO 4k split exists, run:

```bash
CUDA_VISIBLE_DEVICES=0,1 python post_train_v2/scripts/grpo/convert_to_parquet.py \
  --train-jsonl post_train_v2/data/processed/grpo_train_4k.jsonl \
  --val-jsonl post_train_v2/data/processed/eval_50.jsonl \
  --output-dir post_train_v2/data/verl

CUDA_VISIBLE_DEVICES=0,1 python post_train_v2/scripts/grpo/train_grpo.py \
  --config post_train_v2/verl/configs/grpo_smoke.yaml \
  --max-steps 1
```

For a full run, use `post_train_v2/verl/configs/grpo.yaml`, then select and
export best/final actors as described in `docs/runbooks/grpo.md`.

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

- `docs/superpowers/specs/2026-06-15-post-train-v2-core-development-design.md`:
  authoritative core-development contract.
- `analysis.md`: inventory and behavioral analysis of the existing project.
- `migration_plan.md`: historical migration analysis superseded by the core
  design where the two conflict.
- `environment.md`: pinned runtime baseline and remote installation checks.
- `open_questions.md`: resolved baseline and policy for recording any newly
  discovered contract-level questions.

## Runtime Baseline

V2 uses an isolated Python 3.11.15 uv environment pinned by
`configs/environment/runtime-cu128.json`. The core runtime is PyTorch 2.7
cu128, Flash Attention 2.7.4.post1, vLLM 0.9.1, verl 0.6.0, Transformers
4.53.2, and TRL 0.19.1.

All Transformers model-loading paths used by Full SFT, LoRA, RFT, DPO, and
evaluation require Flash Attention 2 and BF16. The verl FSDP actor model
source also requires Flash Attention 2. vLLM uses its own attention backend.
See `docs/environment_setup.md` for the remote installation and two-level
acceptance process.
