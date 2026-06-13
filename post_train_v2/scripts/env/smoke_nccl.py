from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Callable, Mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a two-rank NCCL all-reduce smoke test."
    )
    return parser.parse_args()


def report_peer_access(
    peer_access: bool,
    ipc_available: bool,
    writer: Callable[[str], None] = print,
) -> int:
    if peer_access and ipc_available:
        writer("OK: CUDA peer access and IPC are available")
    else:
        writer(
            "WARNING: CUDA P2P/IPC is unavailable; "
            "the successful NCCL collective remains the hard requirement"
        )
    return 0


def probe_cuda_ipc(tensor: Any) -> bool:
    try:
        storage = tensor.untyped_storage()
        share_cuda = getattr(storage, "_share_cuda_", None)
        return share_cuda is not None and bool(share_cuda())
    except (AttributeError, RuntimeError):
        return False


def run_collective(
    torch_module: Any,
    dist_module: Any,
    environment: Mapping[str, str],
    writer: Callable[[str], None] = print,
    ipc_probe: Callable[[Any], bool] = probe_cuda_ipc,
) -> int:
    required = ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    missing = [name for name in required if name not in environment]
    if missing:
        raise RuntimeError(
            "run this script with torchrun; missing " + ", ".join(missing)
        )

    world_size = int(environment["WORLD_SIZE"])
    local_rank = int(environment["LOCAL_RANK"])
    if world_size != 2:
        raise RuntimeError(f"expected WORLD_SIZE=2, got {world_size}")

    initialized = False
    try:
        torch_module.cuda.set_device(local_rank)
        dist_module.init_process_group("nccl")
        initialized = True
        rank = dist_module.get_rank()

        value = torch_module.tensor([float(rank + 1)], device=f"cuda:{local_rank}")
        scratch = torch_module.ones(1024, device=f"cuda:{local_rank}")
        dist_module.all_reduce(value)
        torch_module.cuda.synchronize(local_rank)
        if value.item() != 3.0 or not torch_module.isfinite(scratch).all().item():
            raise RuntimeError(
                f"NCCL all-reduce failed on rank {rank}: result={value.item()}"
            )

        if rank == 0:
            peer_access = torch_module.cuda.can_device_access_peer(0, 1) and (
                torch_module.cuda.can_device_access_peer(1, 0)
            )
            report_peer_access(
                peer_access,
                ipc_probe(scratch),
                writer=writer,
            )
        dist_module.barrier()
        writer(f"OK: rank={rank} local_rank={local_rank} all_reduce=3.0")
        return 0
    finally:
        if initialized and dist_module.is_initialized():
            dist_module.destroy_process_group()


def main() -> int:
    parse_args()
    import torch
    import torch.distributed as dist

    return run_collective(torch, dist, os.environ)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
