# Distributed Runtime

This package contains small rank-aware helpers used by V2 training entrypoints.
It intentionally lazy-loads `torch.distributed` so CPU-only documentation and
schema tests can import the package without a PyTorch installation.
