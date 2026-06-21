# Model Evaluation

Run the fixed common evaluator:

```text
python post_train_v2/scripts/eval/evaluate_model.py \
  --model-path post_train_v2/outputs/sft/full/best
```

For an unmerged LoRA adapter, add `--base-model-path` unless the adapter
config already records a usable base model path. Outputs are
`samples.jsonl`, `metrics.json`, and `manifest.json`.

