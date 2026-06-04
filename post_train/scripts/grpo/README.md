# GRPO Scripts

Run:

```bash
python post_train/scripts/grpo/train_grpo.py --config post_train/configs/grpo.yaml
```

The GRPO script uses a separate vLLM rollout model and Transformers training model. It logs rollout metrics and fixed 50-example evaluation outputs.
