# Generation Configuration

`teacher_rollout_2gpu.yaml` is the production two-engine Teacher config. It
reads `post_train_v2/data/processed/train_candidates.jsonl` and publishes to
`post_train_v2/data/teacher_rollouts`.

`teacher_rollout_2gpu_smoke.yaml` uses the tracked V2 fixture and isolated
`/tmp` output/cache roots. Both configs use two independent TP=1 workers,
thinking disabled, 256 output tokens, and Manifest V2.
