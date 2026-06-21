# DPO Scripts

Build DPO preference pairs with the two-worker vLLM engine:

```bash
CUDA_VISIBLE_DEVICES=0,1 python post_train_v2/scripts/dpo/build_dpo_data.py \
  --config post_train_v2/configs/dpo/build_smoke.yaml --limit 8
```

Train DPO with two-rank DDP:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  post_train_v2/scripts/dpo/train_dpo.py \
  --config post_train_v2/configs/dpo/train_smoke.yaml --max-steps 2
```
