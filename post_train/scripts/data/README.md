# Data Scripts

Run in order:

```bash
python post_train/scripts/data/build_source.py --config post_train/configs/data_build.yaml
python post_train/scripts/data/build_teacher_pool.py --config post_train/configs/teacher_rollout.yaml
python post_train/scripts/data/build_sft_splits.py --config post_train/configs/data_build.yaml
```

These scripts produce solver-backed data, teacher accepted pool data, and SFT/GRPO splits.
