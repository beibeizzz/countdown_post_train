# Eval Scripts

独立评估入口会强制使用 BF16、Flash Attention 2、关闭 thinking，并使用 solver
检查 `<answer>...</answer>` 中的表达式。完整远程流程见
[`../../docs/remote_training_guide.md`](../../docs/remote_training_guide.md)。

## Full Model

```bash
CUDA_VISIBLE_DEVICES=0 python post_train/scripts/eval/evaluate_model.py \
  --config post_train/configs/eval.yaml \
  --model-path post_train/outputs/sft/full/final \
  --output-dir post_train/data/eval/sft_full
```

## LoRA Adapter

```bash
CUDA_VISIBLE_DEVICES=0 python post_train/scripts/eval/evaluate_model.py \
  --config post_train/configs/eval.yaml \
  --model-path post_train/outputs/sft/lora/final \
  --base-model-path post_train/model/qwen/qwen3-0.6b \
  --output-dir post_train/data/eval/sft_lora
```

LoRA 会从 `adapter_config.json` 自动推导 base model；当远程路径不可用时显式传
`--base-model-path`。

## Smoke

```bash
CUDA_VISIBLE_DEVICES=0 python post_train/scripts/eval/evaluate_model.py \
  --config post_train/configs/eval.yaml \
  --model-path post_train/outputs/sft/full/final \
  --output-dir /tmp/post_train_smoke/eval/sft_full \
  --limit 10
```

输出：

- `eval_samples.jsonl`：逐样本 response、表达式、正确性、token 数和截断状态；
- `eval_metrics.json`：accuracy、format rate、valid-expression rate、输出长度和
  截断统计。

Base、RFT、DPO 和 GRPO 的完整评估命令见详细指南第 18 节。离线评估不会上传
W&B。
