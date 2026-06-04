# Eval Outputs

Standalone model evaluation outputs are written here.

Example:

```bash
python post_train/scripts/eval/evaluate_model.py --config post_train/configs/eval.yaml --model-path post_train/outputs/sft/full/final --output-dir post_train/data/eval/sft_full
```

Each run writes `eval_samples.jsonl` and `eval_metrics.json`.
