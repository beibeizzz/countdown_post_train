# GRPO Scripts

Convert V2 JSONL data to verl Parquet:

```bash
CUDA_VISIBLE_DEVICES=0,1 python post_train_v2/scripts/grpo/convert_to_parquet.py \
  --train-jsonl post_train_v2/data/processed/grpo_train_4k.jsonl \
  --val-jsonl post_train_v2/data/processed/eval_50.jsonl \
  --output-dir post_train_v2/data/verl
```
