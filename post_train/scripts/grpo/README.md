# GRPO Scripts

这是 legacy 单 GPU GRPO，不是 verl/FSDP 实现。完整硬件、显存和 smoke 流程见
[`../../docs/remote_training_guide.md`](../../docs/remote_training_guide.md)。

Run:

```bash
python post_train/scripts/grpo/train_grpo.py --config post_train/configs/grpo.yaml
```

wandb logging is optional and disabled by default. Set `report_to: wandb` in `post_train/configs/grpo.yaml` to enable it. GRPO logs every training metric row to wandb while preserving `metrics.jsonl` and fixed-eval JSON outputs.

The GRPO script uses a separate vLLM rollout model and Transformers training model. It logs rollout metrics and fixed 50-example evaluation outputs.

Both models live in the same process and share the only visible GPU. Clear
stale vLLM processes before launch and use `CUDA_VISIBLE_DEVICES=0`. A second
GPU is not used automatically.

Default behavior includes group size 4, two policy updates per rollout,
`kl_coeff=0.0`, sync/checkpoint every 20 steps, and fixed evaluation every 100
steps. Metrics are appended to `metrics.jsonl`; entropy is optional.

For a two-step smoke, copy `grpo.yaml`, redirect `output_dir` below
`/tmp/post_train_smoke/outputs/grpo`, disable W&B, and pass `--max-steps 2`.
