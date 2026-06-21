"""Small distributed runtime helpers shared by V2 training stages."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from typing import TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def _load_dist():
    try:
        return import_module("torch.distributed")
    except ModuleNotFoundError:
        return None


def _dist_ready(dist) -> bool:
    return bool(
        dist is not None
        and dist.is_available()
        and dist.is_initialized()
    )


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error


def current_context() -> DistributedContext:
    dist = _load_dist()
    if _dist_ready(dist):
        return DistributedContext(
            rank=dist.get_rank(),
            local_rank=_env_int("LOCAL_RANK", dist.get_rank()),
            world_size=dist.get_world_size(),
        )
    return DistributedContext(
        rank=_env_int("RANK", 0),
        local_rank=_env_int("LOCAL_RANK", 0),
        world_size=_env_int("WORLD_SIZE", 1),
    )


def barrier() -> None:
    dist = _load_dist()
    if _dist_ready(dist):
        dist.barrier()


def main_rank_section(fn: Callable[[], T]) -> T | None:
    """Run ``fn`` on rank zero and broadcast success or failure to all ranks."""

    dist = _load_dist()
    context = current_context()
    if not _dist_ready(dist) or context.world_size == 1:
        return fn()

    envelope: list[dict[str, object] | None]
    if context.is_main_process:
        try:
            result = fn()
        except BaseException as error:
            envelope = [
                {
                    "ok": False,
                    "type": type(error).__name__,
                    "message": str(error),
                }
            ]
        else:
            envelope = [{"ok": True, "result": result}]
    else:
        envelope = [None]

    dist.broadcast_object_list(envelope, src=0)
    payload = envelope[0]
    if not isinstance(payload, dict):
        raise RuntimeError("rank-zero main section returned invalid status")
    if not payload.get("ok"):
        message = payload.get("message") or payload.get("type") or "unknown"
        raise RuntimeError(f"rank-zero main section failed: {message}")
    if context.is_main_process:
        return payload.get("result")  # type: ignore[return-value]
    return None
