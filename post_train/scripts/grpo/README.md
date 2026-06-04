# GRPO Scripts

Run:

```bash
python post_train/scripts/grpo/train_grpo.py --config post_train/configs/grpo.yaml
```

wandb logging is optional and disabled by default. Set `report_to: wandb` in `post_train/configs/grpo.yaml` to enable it. GRPO logs every training metric row to wandb while preserving `metrics.jsonl` and fixed-eval JSON outputs.

The GRPO script uses a separate vLLM rollout model and Transformers training model. It logs rollout metrics and fixed 50-example evaluation outputs.
