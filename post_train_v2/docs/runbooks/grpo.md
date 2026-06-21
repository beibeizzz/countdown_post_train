# GRPO Runbook

Run from the repository root after Phase 2 Full SFT has exported
`post_train_v2/outputs/sft/full/best` and Phase 1/2 data builders have produced
`post_train_v2/data/processed/grpo_train_4k.jsonl` and
`post_train_v2/data/processed/eval_50.jsonl`.

## 1. Convert JSONL to verl Parquet

```bash
CUDA_VISIBLE_DEVICES=0,1 python post_train_v2/scripts/grpo/convert_to_parquet.py \
  --train-jsonl post_train_v2/data/processed/grpo_train_4k.jsonl \
  --val-jsonl post_train_v2/data/processed/eval_50.jsonl \
  --output-dir post_train_v2/data/verl
```

Expected outputs:

- `post_train_v2/data/verl/train.parquet`
- `post_train_v2/data/verl/validation.parquet`
- matching `.schema.json` files
- matching `*_manifest.json` files

## 2. Smoke GRPO

```bash
CUDA_VISIBLE_DEVICES=0,1 python post_train_v2/scripts/grpo/train_grpo.py \
  --config post_train_v2/verl/configs/grpo_smoke.yaml \
  --max-steps 1
```

This is a Level 2 remote gate. It requires the pinned uv environment, working
Flash Attention 2, verl, vLLM, and two healthy visible GPUs.

## 3. Full GRPO

```bash
CUDA_VISIBLE_DEVICES=0,1 python post_train_v2/scripts/grpo/train_grpo.py \
  --config post_train_v2/verl/configs/grpo.yaml
```

The default config uses:

- Qwen3-0.6B Full SFT best export as actor base.
- FSDP2 actor training.
- vLLM rollout with `n=4`.
- No critic and no KL reward penalty.
- Rule reward through `post_train_v2/verl/rewards/countdown_reward.py`.
- Checkpoint and validation cadence every 100 steps.
- Fixed validation generation count of 50.

## 4. Select Best Checkpoint

After a run has validation dumps under `RUN_DIR/validation/step_<N>.jsonl` and
actor checkpoints under `RUN_DIR/checkpoints/global_step_<N>/actor`, rescore
native validation dumps with the common Countdown evaluator:

```bash
python - <<'PY'
from post_train_v2.verl.export import select_grpo_checkpoints

select_grpo_checkpoints(
    "post_train_v2/outputs/grpo/run",
    config={"config": "post_train_v2/verl/configs/grpo.yaml"},
)
PY
```

The selector writes `post_train_v2/outputs/grpo/run/export/selection.json`.
Best selection sorts by accuracy, then format rate, then earlier step. The
final checkpoint is always the largest available step.

## 5. Export Best and Final Actors

```bash
python post_train_v2/scripts/grpo/export_grpo.py \
  --run-dir post_train_v2/outputs/grpo/run \
  --prune
```

The exporter first checks `python -m verl.model_merger --help`, then calls:

```bash
python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir post_train_v2/outputs/grpo/run/checkpoints/global_step_100/actor \
  --target_dir post_train_v2/outputs/grpo/run/export/best
```

It exports both `best` and `final`, direct-load checks each merged model, writes
`export_manifest.json`, and only then prunes old continuation checkpoints. The
retention policy keeps the latest two checkpoints plus a distinct selected
best checkpoint.

## 6. Evaluate Exports

```bash
python post_train_v2/scripts/eval/evaluate_model.py \
  --model-path post_train_v2/outputs/grpo/run/export/best

python post_train_v2/scripts/eval/evaluate_model.py \
  --model-path post_train_v2/outputs/grpo/run/export/final
```

Use the existing evaluation output metrics and sample dumps to compare GRPO
best/final against the SFT and DPO baselines.
