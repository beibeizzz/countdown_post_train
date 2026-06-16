# verl Data

This package converts canonical V2 Countdown JSONL source rows into verl's
rule-reward Parquet schema. The conversion keeps prompts as chat messages and
stores Countdown ground truth under `reward_model.ground_truth`.
