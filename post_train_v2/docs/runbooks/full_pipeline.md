# Full Pipeline Runbook

Run from the repository root after activating the pinned `post_train_v2` uv
environment.

## Prerequisites

- Local models exist at `post_train/model/qwen/qwen3-0.6b` and
  `post_train/model/qwen/qwen3-8b`.
- Raw datasets exist under `post_train/datasets/`.
- Level 1 environment gates have passed and are recorded in
  `post_train_v2/outputs/env/runtime_acceptance.json` for production runs.
- `CUDA_VISIBLE_DEVICES=0,1` exposes two 40 GB GPUs on the remote host.
- W&B login and project settings are configured if `report_to` includes wandb.
- Disk space is available under `post_train_v2/data/` and
  `post_train_v2/outputs/`.

## Dry Run

```bash
python post_train_v2/scripts/pipeline/run_pipeline.py \
  --config post_train_v2/configs/pipeline/smoke.yaml \
  --dry-run
```

## Smoke Run

```bash
CUDA_VISIBLE_DEVICES=0,1 python post_train_v2/scripts/env/smoke_v2_training.py \
  --through-stage grpo_export \
  --work-dir /tmp/post_train_v2_smoke
```

## Production Run

```bash
CUDA_VISIBLE_DEVICES=0,1 python post_train_v2/scripts/pipeline/run_pipeline.py \
  --config post_train_v2/configs/pipeline/production.yaml \
  --from-stage build_source \
  --through-stage final_eval
```

Stage events are written to the configured `pipeline_events.jsonl`. Stage
outputs remain isolated below `post_train_v2/data/` and `post_train_v2/outputs/`.
