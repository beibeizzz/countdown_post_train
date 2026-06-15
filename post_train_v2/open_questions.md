# Open Questions

The core-development questions previously tracked in this file are resolved.
The authoritative decisions are frozen in:

- `docs/superpowers/specs/2026-06-15-post-train-v2-core-development-design.md`

No blocking design question remains before writing the phase implementation
plans.

New questions discovered during implementation must be added here only when
they change an approved external contract, training semantic, artifact
schema, or acceptance criterion. Ordinary implementation details should be
resolved within the relevant phase plan.

## Resolved Baseline

- Two-rank `torchrun` DDP for Full SFT, LoRA, RFT training, and DPO.
- Old effective global batches are preserved for the first distributed run.
- Two independent TP1 Qwen3-8B vLLM workers for offline generation.
- GRPO uses verl 0.6.0, FSDP2, vLLM TP1, train batch 4, mini batch 4,
  per-GPU actor and rollout log-probability micro batch 2, four rollouts per
  prompt, two actor epochs per trainer iteration, and zero KL.
- V2 can read compatible legacy data and model artifacts but writes only to
  its own tree by default.
- V2 guarantees training-state resume only for checkpoints created by V2.
- Trainer fixed rank-0 evaluation runs every 100 optimizer steps and at the
  final step. GRPO uses native verl validation on the fixed 50-record Parquet
  at the same cadence.
- GRPO retains periodic native checkpoints through post-training selection,
  exports best and final actors with stock `verl.model_merger`, then keeps
  the latest two continuation checkpoints plus a distinct selected best.
- Gradient checkpointing is enabled by default and configurable per stage.
- W&B uses one logical run per stage and does not upload model or dataset
  artifacts by default.
- Manifest V2, deterministic data selection, checkpoint retention, recovery,
  final evaluation, and four verification levels are defined by the core
  design.
