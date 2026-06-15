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
- GRPO uses verl 0.6.0, FSDP2, vLLM, `ppo_epochs=2`, and zero KL.
- V2 can read compatible legacy data and model artifacts but writes only to
  its own tree by default.
- V2 guarantees training-state resume only for checkpoints created by V2.
- Fixed rank-0 evaluation runs every 100 optimizer steps and at the final
  step.
- Gradient checkpointing is enabled by default and configurable per stage.
- W&B uses one logical run per stage and does not upload model or dataset
  artifacts by default.
- Manifest V2, deterministic data selection, checkpoint retention, recovery,
  final evaluation, and four verification levels are defined by the core
  design.
