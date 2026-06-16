# Pipeline Core

This package defines the static V2 stage DAG and the runner used by
`scripts/pipeline/run_pipeline.py`.

The runner launches existing stage CLIs. It does not duplicate training,
generation, reward, or evaluation logic.
