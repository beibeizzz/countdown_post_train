# DPO Scripts

Build DPO preference pairs with the two-worker vLLM engine:

```bash
CUDA_VISIBLE_DEVICES=0,1 python post_train_v2/scripts/dpo/build_dpo_data.py \
  --config post_train_v2/configs/dpo/build_smoke.yaml --limit 8
```
