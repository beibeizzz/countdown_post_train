# SFT Scripts

这些入口是单 GPU Transformers Trainer 或单 GPU vLLM，不使用 DDP。完整环境、
隔离 smoke 和评估流程见
[`../../docs/remote_training_guide.md`](../../docs/remote_training_guide.md)。

Common commands:

```bash
python post_train/scripts/sft/train_full.py --config post_train/configs/sft_full.yaml
python post_train/scripts/sft/train_lora.py --config post_train/configs/sft_lora.yaml
python post_train/scripts/sft/build_rft_data.py --config post_train/configs/rft.yaml
python post_train/scripts/sft/train_rft.py --config post_train/configs/rft.yaml
```

`train_rft.py` reuses the full SFT trainer and maps `rft.yaml`'s `train` section to a full SFT config.

wandb logging is optional and disabled by default. Set `report_to: wandb` in the corresponding config to enable it. SFT, LoRA, and RFT use the Transformers Trainer integration.

Execution order:

1. Full SFT and LoRA consume `sft_train_8k.jsonl` independently.
2. RFT first runs `build_rft_data.py`, then `train_rft.py`.
3. Evaluate every `final/` independently.

All Transformers loads require BF16 and Flash Attention 2. Use
`--max-steps 2` with a copied config whose output is below
`/tmp/post_train_smoke/`; never smoke into `post_train/outputs/`.

Full/RFT `final/` are complete models. LoRA `final/` is an adapter and may
require `--base-model-path` during evaluation. Review `rft.yaml`'s current
Qwen3-8B `base_model_path` before rollout.
