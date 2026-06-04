# DPO Scripts

Common commands:

```bash
python post_train/scripts/dpo/build_dpo_data.py --config post_train/configs/dpo_data.yaml
python post_train/scripts/dpo/train_dpo.py --config post_train/configs/dpo_train.yaml
```

wandb logging is optional and disabled by default. Set `report_to: wandb` in `post_train/configs/dpo_train.yaml` to enable it. DPO uses the TRL/Transformers Trainer integration.

DPO data uses teacher-generated rejected responses and the SFT full responses as chosen samples.
