# Supervised and RFT Runbook

Run commands from the repository root after Phase 1 artifacts exist:

1. `post_train_v2/data/processed/sft_train_8k.jsonl`
2. `post_train_v2/data/processed/eval_50.jsonl`
3. `post_train_v2/data/processed/validation_manifest.json`

## Full SFT Smoke

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  post_train_v2/scripts/sft/train_full.py \
  --config post_train_v2/configs/sft/full_smoke.yaml --max-steps 2
```

Production:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  post_train_v2/scripts/sft/train_full.py \
  --config post_train_v2/configs/sft/full.yaml
```

## LoRA SFT Smoke

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  post_train_v2/scripts/sft/train_lora.py \
  --config post_train_v2/configs/sft/lora_smoke.yaml --max-steps 2
```

Merge an adapter into a full model export:

```bash
python post_train_v2/scripts/sft/merge_lora.py \
  --base-model-path post_train/model/qwen/qwen3-0.6b \
  --adapter-path post_train_v2/outputs/sft/lora/best \
  --output-dir post_train_v2/outputs/sft/lora_merged/best
```

## RFT Data Smoke

```bash
CUDA_VISIBLE_DEVICES=0,1 python post_train_v2/scripts/sft/build_rft_data.py \
  --config post_train_v2/configs/sft/rft_rollout_smoke.yaml --limit 4
```

Production:

```bash
CUDA_VISIBLE_DEVICES=0,1 python post_train_v2/scripts/sft/build_rft_data.py \
  --config post_train_v2/configs/sft/rft_rollout.yaml
```

## RFT Training Smoke

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  post_train_v2/scripts/sft/train_rft.py \
  --config post_train_v2/configs/sft/rft_train_smoke.yaml --max-steps 2
```

Production:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  post_train_v2/scripts/sft/train_rft.py \
  --config post_train_v2/configs/sft/rft_train.yaml
```

## Outputs

Each supervised stage writes under its configured `output_dir`. Fixed-set
evaluation writes:

```text
eval/step_<N>/samples.jsonl
eval/step_<N>/metrics.json
eval/ledger.jsonl
```

RFT rollout writes:

```text
rft_accepted.jsonl
rft_rejected.jsonl
manifest.json
```
