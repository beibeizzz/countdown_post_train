# Data Configuration

`build_source.yaml` maps the raw Countdown inputs to the normalized V2
warehouse. `build_splits.yaml` freezes `val_200` and `eval_50`, excludes
validation IDs from Teacher candidates, and later creates the stratified SFT
8k and GRPO 4k datasets from a complete 20k Teacher artifact.
