# Acceptance Workflow

## Local Evidence

Run from the repository root:

```bash
python -m pytest -q -p no:cacheprovider post_train_v2/tests
git diff --check
python post_train_v2/scripts/pipeline/run_pipeline.py \
  --config post_train_v2/configs/pipeline/smoke.yaml \
  --dry-run
```

Expected:

- The complete local test suite passes.
- `git diff --check` reports no whitespace errors.
- The dry-run prints all 14 stages in dependency order and does not load a
  model.

## Remote Evidence

Run only on the pinned two-GPU environment:

```bash
CUDA_VISIBLE_DEVICES=0,1 python post_train_v2/scripts/env/smoke_v2_training.py \
  --through-stage grpo_export \
  --work-dir /tmp/post_train_v2_smoke

CUDA_VISIBLE_DEVICES=0,1 python post_train_v2/scripts/pipeline/run_pipeline.py \
  --config post_train_v2/configs/pipeline/smoke.yaml \
  --through-stage final_eval
```

Expected:

- Smoke stages write only under the provided work directory.
- Stage manifests are complete and hash-valid.
- Exported best/final model artifacts pass direct-load checks.
- Final evaluation writes `summary.json` without any stage writing below
  `post_train/`.
