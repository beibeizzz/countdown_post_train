# GRPO Scripts

Convert V2 JSONL data to verl Parquet:

```bash
CUDA_VISIBLE_DEVICES=0,1 python post_train_v2/scripts/grpo/convert_to_parquet.py \
  --train-jsonl post_train_v2/data/processed/grpo_train_4k.jsonl \
  --val-jsonl post_train_v2/data/processed/eval_50.jsonl \
  --output-dir post_train_v2/data/verl
```

Launch a short remote smoke run:

```bash
CUDA_VISIBLE_DEVICES=0,1 python post_train_v2/scripts/grpo/train_grpo.py \
  --config post_train_v2/verl/configs/grpo_smoke.yaml \
  --max-steps 1
```

Launch the default full GRPO config:

```bash
CUDA_VISIBLE_DEVICES=0,1 python post_train_v2/scripts/grpo/train_grpo.py \
  --config post_train_v2/verl/configs/grpo.yaml
```

Export selected best/final actor checkpoints after native validation selection:

```bash
python post_train_v2/scripts/grpo/export_grpo.py \
  --run-dir post_train_v2/outputs/grpo/run \
  --prune
```

See `post_train_v2/docs/runbooks/grpo.md` for checkpoint selection, export,
and evaluation details.
