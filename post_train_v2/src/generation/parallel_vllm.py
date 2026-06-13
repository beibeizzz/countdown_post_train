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

    def __post_init__(self) -> None:
        _validate_worker_index(self.worker_index)


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
_NO_MESSAGE = object()


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

    results: dict[int, WorkerResult] = {}
    workers: set[int] = set()
    for message in received:
        result = _validate_worker_message(batch_id, message, workers)
        workers.add(result.worker_index)
        results[result.worker_index] = result

    if workers != {0, 1}:
        raise ValueError(f"expected results from workers 0 and 1, got {workers}")

    expected = list(expected_positions)
    if len(set(expected)) != len(expected):
        raise ValueError("expected positions contain duplicates")

    merged: list[tuple[int, str]] = []
    seen_positions: set[int] = set()
    actual_positions: dict[int, tuple[int, ...]] = {}
    for worker_index in (0, 1):
        result = results[worker_index]
        worker_positions: list[int] = []
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
            worker_positions.append(position)
            merged.append((position, text))
        actual_positions[worker_index] = tuple(worker_positions)

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

    midpoint = (len(expected) + 1) // 2
    expected_by_worker = {
        0: tuple(expected[:midpoint]),
        1: tuple(expected[midpoint:]),
    }
    for worker_index in (0, 1):
        if actual_positions[worker_index] != expected_by_worker[worker_index]:
            raise ValueError(
                f"worker {worker_index} positions "
                f"{actual_positions[worker_index]} do not match expected shard "
                f"{expected_by_worker[worker_index]}"
            )

    return [*results[0].items, *results[1].items]


def _validate_worker_message(
    batch_id: int,
    message: object,
    seen_workers: set[int],
) -> WorkerResult:
    if isinstance(message, WorkerError):
        _validate_worker_index(message.worker_index)
        if message.batch_id not in (None, batch_id):
            raise ValueError(
                f"worker {message.worker_index} returned error for batch "
                f"{message.batch_id}, expected batch {batch_id}"
            )
        raise RuntimeError(_format_worker_error(message))
    if not isinstance(message, WorkerResult):
        raise ValueError(f"malformed worker result: {message!r}")
    _validate_worker_index(message.worker_index)
    if message.worker_index in seen_workers:
        raise ValueError(
            f"duplicate worker result from worker {message.worker_index}"
        )
    if message.batch_id != batch_id:
        raise ValueError(
            f"worker {message.worker_index} returned batch "
            f"{message.batch_id}, expected batch {batch_id}"
        )
    return message


def _validate_worker_index(worker_index: int) -> None:
    if type(worker_index) is not int or worker_index not in (0, 1):
        raise ValueError(f"unknown worker index: {worker_index}")


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
        if len({spec.device for spec in specs}) != 2:
            raise ValueError("worker devices must be distinct")
        normalized_cache_roots = {
            os.path.normcase(os.path.abspath(spec.cache_root))
            for spec in specs
        }
        if len(normalized_cache_roots) != 2:
            raise ValueError("worker cache roots must be distinct")
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

        deadline = self._monotonic() + self.timeout_seconds
        try:
            for spec, request_queue in zip(
                self.worker_specs, self._request_queues, strict=True
            ):
                self._raise_if_deadline_expired(deadline, "worker startup")
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
                self._raise_if_deadline_expired(deadline, "worker startup")

            ready_workers: set[int] = set()
            while len(ready_workers) < 2:
                message = self._next_message(deadline, "worker startup")
                if isinstance(message, WorkerError):
                    _validate_worker_index(message.worker_index)
                    raise RuntimeError(_format_worker_error(message))
                if not isinstance(message, WorkerReady):
                    raise ValueError(f"malformed worker startup message: {message!r}")
                _validate_worker_index(message.worker_index)
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

        deadline = self._monotonic() + self.timeout_seconds
        try:
            self._last_batch_id = batch_id
            shards = split_contiguous(items)
            for request_queue, shard in zip(
                self._request_queues, shards, strict=True
            ):
                self._raise_if_deadline_expired(deadline, f"batch {batch_id}")
                request_queue.put(WorkerRequest(batch_id, shard))
                self._raise_if_deadline_expired(deadline, f"batch {batch_id}")

            messages: list[WorkerResult] = []
            workers: set[int] = set()
            while len(messages) < 2:
                message = self._next_message(deadline, f"batch {batch_id}")
                result = _validate_worker_message(batch_id, message, workers)
                workers.add(result.worker_index)
                messages.append(result)

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
        deadline = self._monotonic() + self._shutdown_timeout
        cleanup_errors: list[str] = []

        for request_queue in self._request_queues:
            try:
                request_queue.put(_STOP)
            except Exception as exc:
                cleanup_errors.append(f"failed to send worker stop: {exc}")

        self._join_processes(self._processes, deadline, cleanup_errors)
        remaining = self._alive_processes()
        for process in remaining:
            try:
                process.terminate()
            except Exception as exc:
                cleanup_errors.append(f"failed to terminate worker: {exc}")

        self._join_processes(remaining, deadline, cleanup_errors)
        remaining = self._alive_processes()
        for process in remaining:
            kill = getattr(process, "kill", None)
            if kill is None:
                continue
            try:
                kill()
            except Exception as exc:
                cleanup_errors.append(f"failed to kill worker: {exc}")

        self._join_processes(remaining, deadline, cleanup_errors)
        orphans = self._alive_processes()

        for resource_queue in [*self._request_queues, self._response_queue]:
            self._close_queue(resource_queue, cleanup_errors)

        for process in self._processes:
            if process in orphans:
                continue
            close = getattr(process, "close", None)
            if close is None:
                continue
            try:
                close()
            except Exception as exc:
                cleanup_errors.append(f"failed to close worker handle: {exc}")

        if orphans:
            cleanup_errors.append(
                f"{len(orphans)} worker process(es) still alive after shutdown"
            )
        if cleanup_errors:
            raise RuntimeError("; ".join(cleanup_errors))

    def __enter__(self) -> ParallelVLLMEngine:
        return self.start()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _next_message(self, deadline: float, phase: str) -> object:
        while True:
            queued = self._get_message_nowait()
            if queued is not _NO_MESSAGE:
                return queued
            remaining = deadline - self._monotonic()
            if remaining <= 0:
                raise TimeoutError(f"{phase} timed out")
            try:
                return self._response_queue.get(
                    timeout=min(self._poll_interval, remaining)
                )
            except queue.Empty:
                queued = self._get_message_nowait()
                if queued is not _NO_MESSAGE:
                    return queued
                self._raise_if_worker_dead()

    def _raise_if_worker_dead(self) -> None:
        for worker_index, process in enumerate(self._processes):
            exitcode = getattr(process, "exitcode", None)
            if exitcode is not None:
                raise RuntimeError(
                    f"worker {worker_index} exited unexpectedly with "
                    f"exit {exitcode}"
                )

    def _get_message_nowait(self) -> object:
        try:
            return self._response_queue.get_nowait()
        except queue.Empty:
            return _NO_MESSAGE

    def _raise_if_deadline_expired(self, deadline: float, phase: str) -> None:
        if self._monotonic() >= deadline:
            raise TimeoutError(f"{phase} timed out")

    def _join_processes(
        self,
        processes: Sequence[Any],
        deadline: float,
        cleanup_errors: list[str],
    ) -> None:
        for process in processes:
            if not self._process_was_started(process):
                continue
            try:
                process.join(timeout=max(0.0, deadline - self._monotonic()))
            except Exception as exc:
                cleanup_errors.append(f"failed to join worker: {exc}")

    def _alive_processes(self) -> list[Any]:
        return [
            process for process in self._processes if self._process_is_alive(process)
        ]

    @staticmethod
    def _process_was_started(process: Any) -> bool:
        if hasattr(process, "started"):
            return bool(process.started)
        return getattr(process, "pid", None) is not None

    @classmethod
    def _process_is_alive(cls, process: Any) -> bool:
        if not cls._process_was_started(process):
            return False
        is_alive = getattr(process, "is_alive", None)
        if is_alive is not None:
            try:
                return bool(is_alive())
            except (AssertionError, ValueError):
                pass
        return getattr(process, "exitcode", None) is None

    @staticmethod
    def _close_queue(resource_queue: Any, cleanup_errors: list[str]) -> None:
        close = getattr(resource_queue, "close", None)
        if close is not None:
            try:
                close()
            except Exception as exc:
                cleanup_errors.append(f"failed to close queue: {exc}")
        cancel_join_thread = getattr(resource_queue, "cancel_join_thread", None)
        if cancel_join_thread is not None:
            try:
                cancel_join_thread()
            except Exception as exc:
                cleanup_errors.append(f"failed to cancel queue join thread: {exc}")
            return
        join_thread = getattr(resource_queue, "join_thread", None)
        if join_thread is not None:
            try:
                join_thread()
            except Exception as exc:
                cleanup_errors.append(f"failed to join queue thread: {exc}")


def _format_worker_error(error: WorkerError) -> str:
    batch = "initialization" if error.batch_id is None else f"batch {error.batch_id}"
    return (
        f"worker {error.worker_index} failed during {batch}: "
        f"{error.message}; traceback: {error.traceback}"
    )
