# verl Rewards

This package exposes thin custom reward adapters for stock verl. The adapters
delegate scoring to framework-neutral code under `post_train_v2.src.rewards`
and return JSON/Arrow-safe scalar diagnostics.
