# GRPO Export Helpers

This package contains framework-light utilities around verl GRPO runs:

- `select_checkpoint.py` rescoring native validation JSONL dumps with the
  shared Countdown evaluator and writes `export/selection.json`.
- Actor merge helpers live beside the selector so best/final checkpoint export
  can share the same selection artifact.
