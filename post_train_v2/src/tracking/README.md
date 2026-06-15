# Tracking

W&B tracking is optional and rank-aware. Disabled runs and nonzero ranks
return before importing `wandb`, so distributed workers never create
duplicate runs.

Rank 0 names runs with the configured base name, a UTC timestamp, and the
short Git revision. Training integrations call `log_metrics(..., step=step)`
for every trainer step, allowing reward and loss curves to retain the
framework's exact step axis.

