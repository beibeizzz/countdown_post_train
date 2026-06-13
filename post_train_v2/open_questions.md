# Open Questions

The environment baseline and its Level 1 acceptance contract are resolved.
Core distributed training behavior still requires the decisions below.

## Resolved Decisions

- Runtime: Python 3.11.15, PyTorch 2.7 cu128, Flash Attention 2.7.4.post1,
  vLLM 0.9.1, and base verl 0.6.0.
- Environment: independent `post_train_v2/.venv`; do not mutate AgentFlow.
- Teacher topology: two concurrent independent TP1 vLLM engines, one per GPU.
- Trainer launcher: two-rank `torchrun` DDP is the canonical first
  implementation.
- GRPO KL coefficient: zero for the current baseline.
- Acceptance: Level 1 covers runtime/existing paths; Level 2 covers a real
  verl FSDP2 plus vLLM optimizer update after its integration files exist.

## Blocking Training Decisions

1. **Effective batch preservation**

   Should two-GPU DDP preserve the old global batch exactly for comparison, or
   increase it using the available memory?

2. **GRPO policy-update semantics**

   Should two policy updates per rollout map directly to two actor epochs in
   verl 0.6.0, or should the first baseline use the release's conservative
   default?

3. **Checkpoint compatibility**

   Confirm which existing artifacts V2 must load: full Hugging Face models,
   LoRA adapters, DPO checkpoints, legacy GRPO exports, and/or Trainer
   optimizer continuation checkpoints.

4. **CLI compatibility**

   Confirm whether V2 must preserve current flags such as `--config`,
   `--max-steps`, `--limit`, and repository-relative path behavior.

5. **verl ground-truth encoding**

   Confirm through a v0.6.0 integration fixture whether
   `reward_model.ground_truth` should be a nested Arrow struct or a JSON
   string.

## Important Non-Blocking Decisions

6. Gradient checkpointing default after 0.6B memory benchmarking.
7. Explicit versus implicit DPO reference-model handling in TRL 0.19.1.
8. Synchronous rank-0 fixed evaluation versus an asynchronous evaluator.
9. Best-checkpoint metric and tie-break rule.
10. Periodic checkpoint retention and post-training optimizer-state cleanup.
11. W&B project/group naming, mode, sample tables, and artifact upload policy.
12. V2 manifest schema and compatibility reading for legacy manifests.
13. Atomic generation progress flush frequency.
14. Treatment of zero-standard-deviation GRPO groups.
15. GRPO evaluation and Hugging Face export cadence.
16. Whether V2 may read legacy data/output paths while writing only to its
    own output tree by default.
