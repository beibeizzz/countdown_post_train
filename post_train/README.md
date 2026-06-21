# Countdown Post-Training

这是旧版、单 GPU 的 Countdown 后训练流程，不是 `post_train_v2/` 的双 GPU
分布式流程。完整远程环境、硬件、smoke、训练、恢复和评估教程见
[`docs/remote_training_guide.md`](docs/remote_training_guide.md)。

## Preflight

```bash
export CUDA_VISIBLE_DEVICES=0
which python
python -V
uv pip check
nvidia-smi
test -f datasets/raw_train.parquet
test -f datasets/raw_test.json
test -f post_train/model/qwen/qwen3-0.6b/config.json
test -f post_train/model/qwen/qwen3-8b/config.json
```

当前实现使用单个 vLLM TP=1 引擎和普通 Transformers/TRL Trainer，不要求
NCCL、DDP、FSDP 或 `torchrun`。开始正式流程前必须通过详细指南中的 PyTorch
CUDA allocation、Flash Attention 2 Qwen3-0.6B 和 Qwen3-8B vLLM gate。

## Pipeline Dependencies

| Stage | Input | Primary output |
| --- | --- | --- |
| Source | `datasets/raw_*` | `post_train/data/processed/` |
| Teacher | `train_pool.jsonl`, Qwen3-8B | `teacher_accepted_20k.jsonl` |
| Splits | Teacher accepted pool | SFT 8k, GRPO 4k |
| SFT/LoRA | SFT 8k, Qwen3-0.6B | `outputs/sft/*/final` |
| RFT | SFT prompts, configured rollout model | RFT accepted data and final model |
| DPO | SFT chosen, Qwen3-8B rejected | DPO pairs and final model |
| GRPO | Full SFT final, GRPO 4k | metrics, checkpoints, final model |
| Eval | Any final model or LoRA adapter | isolated eval samples and metrics |

## 1. Build Solver-Backed Data

```bash
python post_train/scripts/data/build_source.py --config post_train/configs/data_build.yaml
```

## 2. Build Teacher Accepted Pool

```bash
python post_train/scripts/data/build_teacher_pool.py --config post_train/configs/teacher_rollout.yaml
```

## 3. Build SFT And GRPO Splits

```bash
python post_train/scripts/data/build_sft_splits.py --config post_train/configs/data_build.yaml
```

## 4. Train SFT

```bash
python post_train/scripts/sft/train_full.py --config post_train/configs/sft_full.yaml
python post_train/scripts/sft/train_lora.py --config post_train/configs/sft_lora.yaml
```

## 5. Build RFT Data

```bash
python post_train/scripts/sft/build_rft_data.py --config post_train/configs/rft.yaml
python post_train/scripts/sft/train_rft.py --config post_train/configs/rft.yaml
```

## 6. Build And Train DPO

```bash
python post_train/scripts/dpo/build_dpo_data.py --config post_train/configs/dpo_data.yaml
python post_train/scripts/dpo/train_dpo.py --config post_train/configs/dpo_train.yaml
```

## 7. Train GRPO

```bash
python post_train/scripts/grpo/train_grpo.py --config post_train/configs/grpo.yaml
```

## 8. Evaluate

```bash
python post_train/scripts/eval/evaluate_model.py --config post_train/configs/eval.yaml --model-path post_train/outputs/sft/full/final --output-dir post_train/data/eval/sft_full
```

## Optional wandb Monitoring

Training scripts support optional wandb logging. It is disabled by default with `report_to: null`.

Enable it in the relevant training config:

```yaml
report_to: wandb
wandb_project: countdown-post-train
run_name: sft_full
run_name_auto_suffix: true
```

On the training machine:

```bash
pip install wandb
wandb login
```

The offline evaluator does not upload to wandb:

```bash
python post_train/scripts/eval/evaluate_model.py --config post_train/configs/eval.yaml --model-path post_train/outputs/sft/full/final --output-dir post_train/data/eval/sft_full
```

## Manifest Convention

Generated manifests use the shared `countdown.post_train.manifest.v1` envelope:

- `manifest_version`
- `schema`
- `name`
- `stage`
- `created_at`

Stage-specific fields such as counts, model paths, generation settings, category ratios, and aliases are preserved alongside the shared envelope.

## Stage Documentation

- [Data](scripts/data/README.md)
- [SFT, LoRA and RFT](scripts/sft/README.md)
- [DPO](scripts/dpo/README.md)
- [Legacy GRPO](scripts/grpo/README.md)
- [Evaluation](scripts/eval/README.md)
- [Configuration reference](configs/README.md)
