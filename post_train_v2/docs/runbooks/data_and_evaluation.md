# Data, Teacher, and Evaluation Runbook

Run all commands from the repository root in the pinned V2 environment.

## 1. Build Normalized Source Data

```bash
python post_train_v2/scripts/data/build_source.py \
  --config post_train_v2/configs/data/build_source.yaml
```

Inputs are `datasets/raw_train.parquet` and `datasets/raw_test.json`. Outputs
and `manifest.json` are written under `post_train_v2/data/processed`.
`--limit N` is for development only; accepted production splits require the
unlimited source manifest.

## 2. Freeze Validation and Teacher Candidates

```bash
python post_train_v2/scripts/data/build_splits.py \
  --config post_train_v2/configs/data/build_splits.yaml validation
```

This publishes `val_200.jsonl`, fixed `eval_50.jsonl`, and
`train_candidates.jsonl`. Validation IDs are excluded from candidates.

## 3. Build the 20k Teacher Pool

First run the isolated smoke:

```bash
python post_train_v2/scripts/generation/build_teacher_pool.py \
  --config post_train_v2/configs/generation/teacher_rollout_2gpu_smoke.yaml
```

Then run production:

```bash
python post_train_v2/scripts/generation/build_teacher_pool.py \
  --config post_train_v2/configs/generation/teacher_rollout_2gpu.yaml
```

Rerun the same command to resume. Exit code `2` means the source was
exhausted before 20k accepted rows; inspect
`post_train_v2/data/teacher_rollouts/manifest.json`.

## 4. Build SFT and GRPO Splits

```bash
python post_train_v2/scripts/data/build_splits.py \
  --config post_train_v2/configs/data/build_splits.yaml accepted
```

This requires a complete Teacher Manifest V2 linked to the validation
artifact and publishes `sft_train_8k.jsonl`, `grpo_train_4k.jsonl`, and
`accepted_splits_manifest.json`.

## 5. Evaluate a Full Model

```bash
python post_train_v2/scripts/eval/evaluate_model.py \
  --config post_train_v2/configs/common/eval.yaml \
  --model-path post_train_v2/outputs/sft/full/best \
  --output-dir post_train_v2/data/eval/sft-full
```

## 6. Evaluate an Unmerged LoRA Adapter

```bash
python post_train_v2/scripts/eval/evaluate_model.py \
  --config post_train_v2/configs/common/eval.yaml \
  --model-path post_train_v2/outputs/sft/lora/best \
  --base-model-path post_train/model/qwen/qwen3-0.6b \
  --output-dir post_train_v2/data/eval/sft-lora
```

The base path may be omitted only when `adapter_config.json` contains a
usable `base_model_name_or_path`.

Evaluation always uses the fixed 50 examples, greedy decoding, thinking
disabled, and at most 256 new tokens. Each output directory contains
`samples.jsonl`, `metrics.json`, and Manifest V2.

## 7. Inspect Any Manifest

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("post_train_v2/data/teacher_rollouts/manifest.json")
manifest = json.loads(path.read_text(encoding="utf-8"))
print(json.dumps({
    "artifact_id": manifest["artifact_id"],
    "artifact_type": manifest["artifact_type"],
    "stage": manifest["stage"],
    "files": manifest["files"],
    "parents": manifest["parents"],
    "stage_metadata": manifest["stage_metadata"],
}, indent=2))
PY
```
