# Eval Scripts

Run standalone evaluation with:

```bash
python post_train/scripts/eval/evaluate_model.py --config post_train/configs/eval.yaml --model-path post_train/outputs/sft/full/final --output-dir post_train/data/eval/sft_full
```

Evaluation extracts `<answer>...</answer>`, validates expressions with exact arithmetic, and writes sample and aggregate metrics.
