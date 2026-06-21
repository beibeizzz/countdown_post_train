# Common Configuration

`paths.yaml` records the default repository-relative model, data, and output
locations. `eval.yaml` defines deterministic fixed-set evaluation with
thinking disabled and at most 256 new tokens. `tracking.yaml` contains
optional rank-zero W&B defaults.

Stage-specific configs may override these values, but paths stored in
Manifest V2 remain logical repository-relative paths where possible.
