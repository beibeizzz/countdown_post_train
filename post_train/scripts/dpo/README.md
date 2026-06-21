# DPO Scripts

完整环境、隔离 smoke 和错误类别说明见
[`../../docs/remote_training_guide.md`](../../docs/remote_training_guide.md)。

Common commands:

```bash
python post_train/scripts/dpo/build_dpo_data.py --config post_train/configs/dpo_data.yaml
python post_train/scripts/dpo/train_dpo.py --config post_train/configs/dpo_train.yaml
```

wandb logging is optional and disabled by default. Set `report_to: wandb` in `post_train/configs/dpo_train.yaml` to enable it. DPO uses the TRL/Transformers Trainer integration.

DPO data uses teacher-generated rejected responses and the SFT full responses as chosen samples.

`build_dpo_data.py` uses one TP=1 Qwen3-8B vLLM engine. It combines
forced-wrong and high-temperature candidates, filters unexpected correct
answers, controls malformed samples, and targets about 6,000 pairs.

`train_dpo.py` starts from `post_train/outputs/sft/full/final` and uses TRL on
one GPU. For smoke, copy both configs, redirect data/model outputs to
`/tmp/post_train_smoke/`, then use `--limit` and `--max-steps 2`.

Inspect `post_train/data/dpo/manifest.json` and category counts before
training; do not weaken correctness filtering just to reach the target count.
