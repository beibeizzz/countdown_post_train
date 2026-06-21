# Data Scripts

本目录使用 CPU solver 和单 GPU Qwen3-8B vLLM。完整环境与 smoke 说明见
[`../../docs/remote_training_guide.md`](../../docs/remote_training_guide.md)。

Run in order:

```bash
python post_train/scripts/data/build_source.py --config post_train/configs/data_build.yaml
python post_train/scripts/data/build_teacher_pool.py --config post_train/configs/teacher_rollout.yaml
python post_train/scripts/data/build_sft_splits.py --config post_train/configs/data_build.yaml
```

These scripts produce solver-backed data, teacher accepted pool data, and SFT/GRPO splits.

Inputs and outputs:

- `build_source.py`: root `datasets/raw_train.parquet` and `raw_test.json` ->
  `post_train/data/processed/` plus `manifest.json`.
- `build_teacher_pool.py`: `train_pool.jsonl` + Qwen3-8B -> accepted 20k,
  rejected rows and Teacher manifest.
- `build_sft_splits.py`: accepted pool -> SFT 8k and GRPO 4k.

Use `CUDA_VISIBLE_DEVICES=0` for Teacher. The Teacher stage uses an output
lock; use `--recover-stale-lock` only after confirming the original process is
gone. Source smoke supports `--limit`, but Teacher and split outputs are fixed,
so do not run their smoke against production directories.

```bash
wc -l post_train/data/processed/val_200.jsonl \
  post_train/data/processed/val_eval_50.jsonl \
  post_train/data/sft/sft_train_8k.jsonl \
  post_train/data/grpo/grpo_train_4k.jsonl
```
