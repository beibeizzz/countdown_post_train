# Supervised Scripts

Phase 2 supervised entrypoints are launched with `torchrun` for two-rank DDP.

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  post_train_v2/scripts/sft/train_full.py \
  --config post_train_v2/configs/sft/full_smoke.yaml --max-steps 2
```

LoRA SFT:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  post_train_v2/scripts/sft/train_lora.py \
  --config post_train_v2/configs/sft/lora_smoke.yaml --max-steps 2
```

RFT rollout and training:

```bash
CUDA_VISIBLE_DEVICES=0,1 python post_train_v2/scripts/sft/build_rft_data.py \
  --config post_train_v2/configs/sft/rft_rollout_smoke.yaml --limit 4

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  post_train_v2/scripts/sft/train_rft.py \
  --config post_train_v2/configs/sft/rft_train_smoke.yaml --max-steps 2
```
