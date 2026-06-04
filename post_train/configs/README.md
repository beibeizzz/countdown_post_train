# Configs

Per-stage YAML configs live here. Paths are resolved relative to the workspace root.

Main files:

- `data_build.yaml`: raw data paths, validation/eval sizes, split targets.
- `teacher_rollout.yaml`: Qwen3-8B teacher generation defaults.
- `sft_full.yaml`, `sft_lora.yaml`, `rft.yaml`: SFT/RFT settings.
- `dpo_data.yaml`, `dpo_train.yaml`: DPO data and training settings.
- `grpo.yaml`: GRPO rollout/training settings.
- `eval.yaml`: fixed validation evaluation settings.

Training configs include disabled-by-default wandb fields:

- `report_to`: set to `wandb` to enable.
- `wandb_project`: default project name.
- `wandb_entity`: optional team/user.
- `wandb_group`: optional run grouping.
- `wandb_tags`: optional tags.
- `run_name`: run display name.
- `run_name_auto_suffix`: append a timestamp to avoid duplicate run names.
- `logging_steps`: Trainer logging cadence. GRPO writes local metrics every step and logs training metrics to wandb every step when enabled.

The standalone evaluator config does not upload metrics to wandb.
