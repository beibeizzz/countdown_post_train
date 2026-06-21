# Recovery Runbook

The V2 pipeline is manifest-driven. A stage is complete only when its manifest
is valid, marked `completed=true`, and all declared output hashes match.

## Resume A Stage

```bash
python post_train_v2/scripts/pipeline/run_pipeline.py \
  --config post_train_v2/configs/pipeline/production.yaml \
  --from-stage full_sft \
  --through-stage full_sft
```

If a stage is stale because inputs or config changed, the runner stops. To
intentionally rebuild:

```bash
python post_train_v2/scripts/pipeline/run_pipeline.py \
  --config post_train_v2/configs/pipeline/production.yaml \
  --from-stage full_sft \
  --through-stage full_sft \
  --rebuild-stage full_sft
```

Trainer and GRPO stages can append `--resume-from-checkpoint` from configured
checkpoint directories. Teacher generation uses its own committed manifest and
continues from the last committed source position.

Temporary files and partial manifests are never accepted as complete artifacts.
