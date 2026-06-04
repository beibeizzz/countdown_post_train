# Countdown Post-Training

## 1. Build Solver-Backed Data

```bash
python post_train/scripts/data/build_source.py --config post_train/configs/data_build.yaml
```

## 2. Build Teacher Accepted Pool

```bash
python post_train/scripts/data/build_teacher_pool.py --config post_train/configs/teacher_rollout.yaml
```

## 3. Build SFT And GRPO Splits

```bash
python post_train/scripts/data/build_sft_splits.py --config post_train/configs/data_build.yaml
```

## 4. Train SFT

```bash
python post_train/scripts/sft/train_full.py --config post_train/configs/sft_full.yaml
python post_train/scripts/sft/train_lora.py --config post_train/configs/sft_lora.yaml
```

## 5. Build RFT Data

```bash
python post_train/scripts/sft/build_rft_data.py --config post_train/configs/rft.yaml
python post_train/scripts/sft/train_rft.py --config post_train/configs/rft.yaml
```

## 6. Build And Train DPO

```bash
python post_train/scripts/dpo/build_dpo_data.py --config post_train/configs/dpo_data.yaml
python post_train/scripts/dpo/train_dpo.py --config post_train/configs/dpo_train.yaml
```

## 7. Train GRPO

```bash
python post_train/scripts/grpo/train_grpo.py --config post_train/configs/grpo.yaml
```

## 8. Evaluate

```bash
python post_train/scripts/eval/evaluate_model.py --config post_train/configs/eval.yaml --model-path post_train/outputs/sft/full/final --output-dir post_train/data/eval/sft_full
```

## Manifest Convention

Generated manifests use the shared `countdown.post_train.manifest.v1` envelope:

- `manifest_version`
- `schema`
- `name`
- `stage`
- `created_at`

Stage-specific fields such as counts, model paths, generation settings, category ratios, and aliases are preserved alongside the shared envelope.
