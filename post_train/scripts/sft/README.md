# SFT Scripts

Common commands:

```bash
python post_train/scripts/sft/train_full.py --config post_train/configs/sft_full.yaml
python post_train/scripts/sft/train_lora.py --config post_train/configs/sft_lora.yaml
python post_train/scripts/sft/build_rft_data.py --config post_train/configs/rft.yaml
python post_train/scripts/sft/train_rft.py --config post_train/configs/rft.yaml
```

`train_rft.py` reuses the full SFT trainer and maps `rft.yaml`'s `train` section to a full SFT config.

wandb logging is optional and disabled by default. Set `report_to: wandb` in the corresponding config to enable it. SFT, LoRA, and RFT use the Transformers Trainer integration.
