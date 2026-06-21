# Pipeline Scripts

Dry-run the full V2 stage DAG:

```bash
python post_train_v2/scripts/pipeline/run_pipeline.py \
  --config post_train_v2/configs/pipeline/smoke.yaml \
  --dry-run
```

Production runs require a runtime acceptance file recording successful Level 1
environment gates.
