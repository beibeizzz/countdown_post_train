from __future__ import annotations

import queue
from multiprocessing.context import BaseContext
from pathlib import Path

import pytest

import post_train_v2.src.distributed.runtime as runtime
from post_train_v2.src.distributed.runtime import (
    current_context,
    main_rank_section,
)

try:
    import torch.distributed as dist
    import torch.multiprocessing as mp
except ModuleNotFoundError:
    dist = None
    mp = None


def _spawn_context() -> BaseContext:
    if mp is None:
        pytest.skip("torch is not installed")
    return mp.get_context("spawn")


def _join(processes) -> None:
    for process in processes:
        process.join(timeout=20)
    alive = [process.pid for process in processes if process.is_alive()]
    for process in processes:
        if process.is_alive():
            process.terminate()
    assert alive == []
    assert [process.exitcode for process in processes] == [0, 0]


def _init_gloo(rank: int, world_size: int, init_file: str) -> None:
    if dist is None:
        pytest.skip("torch is not installed")
    dist.init_process_group(
        backend="gloo",
        init_method=f"file:///{Path(init_file).as_posix()}",
        rank=rank,
        world_size=world_size,
    )


def _success_worker(rank, world_size, init_file, side_effect_count, results):
    _init_gloo(rank, world_size, init_file)
    try:
        context = current_context()

        def only_once():
            with side_effect_count.get_lock():
                side_effect_count.value += 1
            return "rank-zero-result"

        result = main_rank_section(only_once)
        results.put(
            {
                "rank": context.rank,
                "local_rank": context.local_rank,
                "world_size": context.world_size,
                "is_main_process": context.is_main_process,
                "result": result,
            }
        )
    finally:
        dist.destroy_process_group()


def _failure_worker(rank, world_size, init_file, results):
    _init_gloo(rank, world_size, init_file)
    try:
        def explode():
            raise ValueError("rank zero failed")

        try:
            main_rank_section(explode)
        except RuntimeError as error:
            results.put((rank, str(error)))
        else:
            results.put((rank, "no error"))
    finally:
        dist.destroy_process_group()


def test_current_context_uses_environment_without_initialized_dist(monkeypatch):
    monkeypatch.setattr(runtime, "_load_dist", lambda: None)
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")

    context = current_context()

    assert context.rank == 1
    assert context.local_rank == 0
    assert context.world_size == 2
    assert context.is_main_process is False


def test_main_rank_section_runs_locally_without_initialized_dist(monkeypatch):
    calls = []
    monkeypatch.setattr(runtime, "_load_dist", lambda: None)

    assert main_rank_section(lambda: calls.append("ran") or 7) == 7
    assert calls == ["ran"]


@pytest.mark.skipif(dist is None, reason="torch is not installed")
def test_main_rank_section_runs_side_effect_once(tmp_path):
    context = _spawn_context()
    init_file = tmp_path / "gloo-success"
    side_effect_count = context.Value("i", 0)
    results = context.Queue()
    processes = [
        context.Process(
            target=_success_worker,
            args=(rank, 2, str(init_file), side_effect_count, results),
        )
        for rank in range(2)
    ]

    for process in processes:
        process.start()
    _join(processes)

    rows = [results.get_nowait() for _ in range(2)]
    assert side_effect_count.value == 1
    assert sorted(row["rank"] for row in rows) == [0, 1]
    assert {row["world_size"] for row in rows} == {2}
    assert {
        row["rank"]: row["is_main_process"] for row in rows
    } == {0: True, 1: False}
    assert {row["rank"]: row["result"] for row in rows} == {
        0: "rank-zero-result",
        1: None,
    }


@pytest.mark.skipif(dist is None, reason="torch is not installed")
def test_main_rank_section_propagates_rank_zero_failure(tmp_path):
    context = _spawn_context()
    init_file = tmp_path / "gloo-failure"
    results = context.Queue()
    processes = [
        context.Process(
            target=_failure_worker,
            args=(rank, 2, str(init_file), results),
        )
        for rank in range(2)
    ]

    for process in processes:
        process.start()
    _join(processes)

    rows = sorted(results.get_nowait() for _ in range(2))
    assert rows == [
        (0, "rank-zero main section failed: rank zero failed"),
        (1, "rank-zero main section failed: rank zero failed"),
    ]
    with pytest.raises(queue.Empty):
        results.get_nowait()
