# Supervised Scripts

Phase 2 supervised entrypoints are launched with `torchrun` for two-rank DDP.

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  post_train_v2/scripts/sft/train_full.py \
  --config post_train_v2/configs/sft/full_smoke.yaml --max-steps 2
```
