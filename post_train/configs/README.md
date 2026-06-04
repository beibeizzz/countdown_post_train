# Configs

Per-stage YAML configs live here. Paths are resolved relative to the workspace root.

Main files:

- `data_build.yaml`: raw data paths, validation/eval sizes, split targets.
- `teacher_rollout.yaml`: Qwen3-8B teacher generation defaults.
- `sft_full.yaml`, `sft_lora.yaml`, `rft.yaml`: SFT/RFT settings.
- `dpo_data.yaml`, `dpo_train.yaml`: DPO data and training settings.
- `grpo.yaml`: GRPO rollout/training settings.
- `eval.yaml`: fixed validation evaluation settings.
