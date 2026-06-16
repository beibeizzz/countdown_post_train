# SFT Configs

This directory contains conservative Phase 2 configs:

- `full.yaml` / `full_smoke.yaml`: full-parameter SFT.
- `lora.yaml` / `lora_smoke.yaml`: LoRA SFT with `r=16`, alpha 32,
  dropout 0.05.
- `rft_rollout.yaml` / `rft_rollout_smoke.yaml`: Qwen3-8B RFT rollouts,
  four samples per prompt, temperature 0.7, top-p 0.95, max 256 tokens,
  thinking disabled.
- `rft_train.yaml` / `rft_train_smoke.yaml`: full-parameter RFT training on
  accepted RFT data.

Launch training configs with `torchrun --standalone --nproc_per_node=2`.
