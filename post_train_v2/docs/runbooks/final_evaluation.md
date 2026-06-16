# Final Evaluation Runbook

The final matrix compares base, Full SFT, LoRA, RFT, DPO, GRPO, and optional
Teacher models on configured validation and solved-test datasets.

```bash
python post_train_v2/scripts/eval/evaluate_matrix.py \
  --config post_train_v2/configs/eval/final_matrix.yaml
```

Outputs are written under `post_train_v2/outputs/eval/final_matrix/`:

- `<model>/<dataset>/samples.jsonl`
- `<model>/<dataset>/metrics.json`
- `summary.json`

Failures are recorded per model/dataset. Independent entries continue running,
and the CLI exits nonzero if any entry failed.
