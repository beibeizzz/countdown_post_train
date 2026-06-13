from __future__ import annotations

import multiprocessing
import os
import queue
import time
import traceback as traceback_module
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


@dataclass(frozen=True)
class PositionedPrompt:
    position: int
    prompt: str


@dataclass(frozen=True)
class WorkerSpec:
    worker_index: int
    device: int
    cache_root: str


@dataclass(frozen=True)
class WorkerRequest:
    batch_id: int
    items: tuple[PositionedPrompt, ...]


@dataclass(frozen=True)
class WorkerReady:
    worker_index: int


@dataclass(frozen=True)
class WorkerResult:
    worker_index: int
    batch_id: int
    items: tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class WorkerError:
    worker_index: int
    batch_id: int | None
    message: str
    traceback: str


@dataclass(frozen=True)
class _StopWorker:
    pass


_STOP = _StopWorker()


def split_contiguous(
    items: Sequence[PositionedPrompt],
) -> tuple[tuple[PositionedPrompt, ...], tuple[PositionedPrompt, ...]]:
    ordered = tuple(items)
    midpoint = (len(ordered) + 1) // 2
    return ordered[:midpoint], ordered[midpoint:]


def merge_worker_results(
    batch_id: int,
    expected_positions: Sequence[int],
    messages: Iterable[object],
) -> list[tuple[int, str]]:
    received = list(messages)
    if len(received) != 2:
        raise ValueError(
            f"expected exactly two worker results, received {len(received)}"
        )

    results: list[WorkerResult] = []
    workers: set[int] = set()
    for message in received:
        if isinstance(message, WorkerError):
            raise RuntimeError(_format_worker_error(message))
        if not isinstance(message, WorkerResult):
            raise ValueError(f"malformed worker result: {message!r}")
        if message.worker_index not in (0, 1):
            raise ValueError(f"unknown worker index: {message.worker_index}")
        if message.worker_index in workers:
            raise ValueError(
                f"duplicate worker result from worker {message.worker_index}"
            )
        if message.batch_id != batch_id:
            raise ValueError(
                f"worker {message.worker_index} returned batch "
                f"{message.batch_id}, expected batch {batch_id}"
            )
        workers.add(message.worker_index)
        results.append(message)

    if workers != {0, 1}:
        raise ValueError(f"expected results from workers 0 and 1, got {workers}")

    expected = list(expected_positions)
    if len(set(expected)) != len(expected):
        raise ValueError("expected positions contain duplicates")

    merged: list[tuple[int, str]] = []
    seen_positions: set[int] = set()
    for result in results:
        for item in result.items:
            if (
                not isinstance(item, tuple)
                or len(item) != 2
                or not isinstance(item[0], int)
                or not isinstance(item[1], str)
            ):
                raise ValueError(f"malformed positioned response: {item!r}")
            position, text = item
            if position in seen_positions:
                raise ValueError(f"duplicate position in worker results: {position}")
            seen_positions.add(position)
            merged.append((position, text))

    expected_set = set(expected)
    unknown = sorted(seen_positions - expected_set)
    missing = sorted(expected_set - seen_positions)
    details = []
    if len(merged) != len(expected):
        details.append(
            f"result count mismatch: received {len(merged)}, "
            f"expected {len(expected)}"
        )
    if unknown:
        details.append(f"unknown positions: {unknown}")
    if missing:
        details.append(f"missing positions: {missing}")
    if details:
        raise ValueError("; ".join(details))

    merged.sort(key=lambda item: item[0])
    sorted_expected = sorted(expected)
    if [position for position, _ in merged] != sorted_expected:
        raise ValueError("merged positions do not match the expected position list")
    return merged


def _default_generator_factory(**kwargs):
    from post_train.src.countdown.generation import VLLMGenerator

    return VLLMGenerator(**kwargs)


def worker_main(
    spec: WorkerSpec,
    request_queue,
    response_queue,
    model_path: str,
    gpu_memory_utilization: float,
    max_model_len: int,
    seed: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    enable_thinking: bool,
    generator_factory: Callable[..., Any] | None = None,
) -> None:
    batch_id: int | None = None
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(spec.device)
        os.environ["VLLM_CACHE_ROOT"] = spec.cache_root
        Path(spec.cache_root).mkdir(parents=True, exist_ok=True)

        from post_train.src.countdown.generation import GenerationConfig

        factory = generator_factory or _default_generator_factory
        generator = factory(
            model_path=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            seed=seed,
        )
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            enable_thinking=enable_thinking,
        )
        response_queue.put(WorkerReady(spec.worker_index))

        while True:
            message = request_queue.get()
            if isinstance(message, _StopWorker):
                return
            if not isinstance(message, WorkerRequest):
                raise TypeError(f"malformed worker request: {message!r}")

            batch_id = message.batch_id
            if not message.items:
                response_queue.put(
                    WorkerResult(spec.worker_index, batch_id, ())
                )
                batch_id = None
                continue

            prompt_texts = [item.prompt for item in message.items]
            responses = generator.generate(prompt_texts, generation_config)
            if len(responses) != len(message.items):
                raise ValueError(
                    "response count mismatch: "
                    f"received {len(responses)}, expected {len(message.items)}"
                )
            positioned = tuple(
                (item.position, response)
                for item, response in zip(message.items, responses, strict=True)
            )
            response_queue.put(
                WorkerResult(spec.worker_index, batch_id, positioned)
            )
            batch_id = None
    except Exception as exc:
        response_queue.put(
            WorkerError(
                worker_index=spec.worker_index,
                batch_id=batch_id,
                message=str(exc) or type(exc).__name__,
                traceback=traceback_module.format_exc(),
            )
        )
        raise SystemExit(1) from exc


class ParallelVLLMEngine:
    def __init__(
        self,
        *,
        model_path: str,
        worker_specs: Sequence[WorkerSpec],
        gpu_memory_utilization: float,
        max_model_len: int,
        seed: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        enable_thinking: bool,
        timeout_seconds: float,
        context=None,
        worker_target: Callable[..., None] = worker_main,
        monotonic: Callable[[], float] = time.monotonic,
        poll_interval: float = 0.05,
        shutdown_timeout: float = 5.0,
    ) -> None:
        specs = tuple(worker_specs)
        if len(specs) != 2:
            raise ValueError("ParallelVLLMEngine requires exactly two worker specs")
        if tuple(spec.worker_index for spec in specs) != (0, 1):
            raise ValueError("worker indices must be ordered exactly as (0, 1)")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        if shutdown_timeout < 0:
            raise ValueError("shutdown_timeout must be non-negative")

        self.model_path = model_path
        self.worker_specs = specs
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.seed = seed
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.enable_thinking = enable_thinking
        self.timeout_seconds = timeout_seconds
        self._context = context or multiprocessing.get_context("spawn")
        self._worker_target = worker_target
        self._monotonic = monotonic
        self._poll_interval = poll_interval
        self._shutdown_timeout = shutdown_timeout
        self._request_queues = [
            self._context.Queue(),
            self._context.Queue(),
        ]
        self._response_queue = self._context.Queue()
        self._processes: list[Any] = []
        self._started = False
        self._closed = False
        self._last_batch_id: int | None = None

    def start(self) -> ParallelVLLMEngine:
        if self._started:
            raise RuntimeError("ParallelVLLMEngine is already started")
        if self._closed:
            raise RuntimeError("ParallelVLLMEngine is already closed")

        try:
            for spec, request_queue in zip(
                self.worker_specs, self._request_queues, strict=True
            ):
                process = self._context.Process(
                    target=self._worker_target,
                    args=(
                        spec,
                        request_queue,
                        self._response_queue,
                        self.model_path,
                        self.gpu_memory_utilization,
                        self.max_model_len,
                        self.seed,
                        self.max_new_tokens,
                        self.temperature,
                        self.top_p,
                        self.enable_thinking,
                    ),
                )
                self._processes.append(process)
                process.start()

            deadline = self._monotonic() + self.timeout_seconds
            ready_workers: set[int] = set()
            while len(ready_workers) < 2:
                message = self._next_message(deadline, "worker startup")
                if isinstance(message, WorkerError):
                    raise RuntimeError(_format_worker_error(message))
                if not isinstance(message, WorkerReady):
                    raise ValueError(f"malformed worker startup message: {message!r}")
                if message.worker_index not in (0, 1):
                    raise ValueError(
                        f"unknown worker index in ready message: "
                        f"{message.worker_index}"
                    )
                if message.worker_index in ready_workers:
                    raise ValueError(
                        f"duplicate ready message from worker "
                        f"{message.worker_index}"
                    )
                ready_workers.add(message.worker_index)
        except Exception:
            self.close()
            raise

        self._started = True
        return self

    def generate(
        self,
        batch_id: int,
        items: Sequence[PositionedPrompt],
    ) -> list[tuple[int, str]]:
        if not self._started or self._closed:
            raise RuntimeError("ParallelVLLMEngine is not started")
        if self._last_batch_id is not None and batch_id <= self._last_batch_id:
            raise ValueError(
                "batch IDs must be strictly increasing: "
                f"received {batch_id} after {self._last_batch_id}"
            )

        self._last_batch_id = batch_id
        shards = split_contiguous(items)
        for request_queue, shard in zip(
            self._request_queues, shards, strict=True
        ):
            request_queue.put(WorkerRequest(batch_id, shard))

        deadline = self._monotonic() + self.timeout_seconds
        messages: list[WorkerResult] = []
        workers: set[int] = set()
        try:
            while len(messages) < 2:
                message = self._next_message(deadline, f"batch {batch_id}")
                if isinstance(message, WorkerError):
                    if message.batch_id not in (None, batch_id):
                        raise ValueError(
                            f"worker {message.worker_index} returned error for "
                            f"batch {message.batch_id}, expected batch {batch_id}"
                        )
                    raise RuntimeError(_format_worker_error(message))
                if not isinstance(message, WorkerResult):
                    raise ValueError(
                        f"malformed worker batch message: {message!r}"
                    )
                if message.worker_index not in (0, 1):
                    raise ValueError(
                        f"unknown worker index in result: "
                        f"{message.worker_index}"
                    )
                if message.batch_id != batch_id:
                    raise ValueError(
                        f"worker {message.worker_index} returned batch "
                        f"{message.batch_id}, expected batch {batch_id}"
                    )
                if message.worker_index in workers:
                    raise ValueError(
                        f"duplicate worker result from worker "
                        f"{message.worker_index}"
                    )
                workers.add(message.worker_index)
                messages.append(message)

            return merge_worker_results(
                batch_id=batch_id,
                expected_positions=[item.position for item in items],
                messages=messages,
            )
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        for request_queue in self._request_queues:
            try:
                request_queue.put(_STOP)
            except Exception:
                pass

        for process in self._processes:
            try:
                process.join(timeout=self._shutdown_timeout)
            except Exception:
                pass

        stubborn = [
            process
            for process in self._processes
            if getattr(process, "exitcode", None) is None
        ]
        for process in stubborn:
            try:
                process.terminate()
            except Exception:
                pass
        for process in stubborn:
            try:
                process.join(timeout=self._shutdown_timeout)
            except Exception:
                pass

    def __enter__(self) -> ParallelVLLMEngine:
        return self.start()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _next_message(self, deadline: float, phase: str) -> object:
        while True:
            self._raise_if_worker_dead()
            remaining = deadline - self._monotonic()
            if remaining <= 0:
                raise TimeoutError(f"{phase} timed out")
            try:
                return self._response_queue.get(
                    timeout=min(self._poll_interval, remaining)
                )
            except queue.Empty:
                continue

    def _raise_if_worker_dead(self) -> None:
        for worker_index, process in enumerate(self._processes):
            exitcode = getattr(process, "exitcode", None)
            if exitcode is not None:
                raise RuntimeError(
                    f"worker {worker_index} exited unexpectedly with "
                    f"exit {exitcode}"
                )


def _format_worker_error(error: WorkerError) -> str:
    batch = "initialization" if error.batch_id is None else f"batch {error.batch_id}"
    return (
        f"worker {error.worker_index} failed during {batch}: "
        f"{error.message}; traceback: {error.traceback}"
    )
