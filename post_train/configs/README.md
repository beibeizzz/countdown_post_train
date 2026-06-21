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

## Edit-Before-Run Checklist

详细环境和训练流程见
[`../docs/remote_training_guide.md`](../docs/remote_training_guide.md)。不要把
`post_train_v2` 的固定版本或双 GPU配置复制到这些文件。

| Config | Owner | Important inputs | Output / review points |
| --- | --- | --- | --- |
| `data_build.yaml` | source + splits | root `datasets/raw_*` | processed data, 200/50/8k/4k targets |
| `teacher_rollout.yaml` | Teacher | Qwen3-8B | thinking off, batch 64, accepted target 20k |
| `sft_full.yaml` | Full SFT | Qwen3-0.6B, SFT 8k | output dir, LR, batch, accumulation |
| `sft_lora.yaml` | LoRA | Qwen3-0.6B, SFT 8k | LoRA targets/rank and adapter output |
| `rft.yaml` | RFT | prompts and rollout model | review `base_model_path`; accepted and model outputs |
| `dpo_data.yaml` | DPO generation | Qwen3-8B, SFT chosen | pair target and category fractions |
| `dpo_train.yaml` | DPO training | Full SFT final, DPO pairs | beta, LR and output dir |
| `grpo.yaml` | legacy GRPO | Full SFT final, GRPO 4k | shared-GPU memory, sync/save/eval cadence |
| `eval.yaml` | evaluator | fixed validation/test files | generation limit; output is CLI-controlled |

For smoke runs, copy the YAML below `/tmp/post_train_smoke/configs/` and
rewrite every output path. Never point a two-step smoke at a production output
directory.
