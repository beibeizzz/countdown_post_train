# DPO Runbook

Run these commands from the repository root after Phase 2 has produced:

- `post_train_v2/outputs/sft/full/best`
- `post_train_v2/data/processed/sft_train_8k.jsonl`
- `post_train_v2/data/processed/eval_50.jsonl`

## Build DPO Data

Smoke:

```bash
CUDA_VISIBLE_DEVICES=0,1 python post_train_v2/scripts/dpo/build_dpo_data.py \
  --config post_train_v2/configs/dpo/build_smoke.yaml --limit 8
```

Production:

```bash
CUDA_VISIBLE_DEVICES=0,1 python post_train_v2/scripts/dpo/build_dpo_data.py \
  --config post_train_v2/configs/dpo/build.yaml
```

The builder publishes:

```text
dpo_candidates.jsonl
dpo_pairs.jsonl
manifest.json
```

Rejected categories remain:

```text
wrong_value
number_mismatch
invalid_expression
missing_answer_tag
truncated
```

## Train DPO

Smoke:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  post_train_v2/scripts/dpo/train_dpo.py \
  --config post_train_v2/configs/dpo/train_smoke.yaml --max-steps 2
```

Production:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  post_train_v2/scripts/dpo/train_dpo.py \
  --config post_train_v2/configs/dpo/train.yaml
```

DPO data generation uses Qwen3-8B, forced-wrong temperature `0.3`,
high-temperature rollout temperature `1.0`, top-p `0.95`, max 256 tokens,
and thinking disabled.

DPO training uses the Full SFT `best/` model, implicit TRL reference model,
beta `0.05`, max length `256`, BF16, Flash Attention 2, and fixed-50
evaluation.
