from __future__ import annotations

import ast
import multiprocessing
import os
import queue
from pathlib import Path

import pytest

from post_train_v2.src.generation.parallel_vllm import (
    ParallelVLLMEngine,
    PositionedPrompt,
    WorkerError,
    WorkerReady,
    WorkerRequest,
    WorkerResult,
    WorkerSpec,
    _STOP,
    merge_worker_results,
    split_contiguous,
    worker_main,
)


def prompts(count: int, start: int = 0) -> list[PositionedPrompt]:
    return [
        PositionedPrompt(position=index, prompt=f"prompt-{index}")
        for index in range(start, start + count)
    ]


def spawn_fake_worker(
    spec,
    request_queue,
    response_queue,
    model_path,
    gpu_memory_utilization,
    max_model_len,
    seed,
    max_new_tokens,
    temperature,
    top_p,
    enable_thinking,
) -> None:
    del (
        model_path,
        gpu_memory_utilization,
        max_model_len,
        seed,
        max_new_tokens,
        temperature,
        top_p,
        enable_thinking,
    )
    response_queue.put(WorkerReady(spec.worker_index))
    while True:
        message = request_queue.get()
        if message == _STOP:
            return
        response_queue.put(
            WorkerResult(
                spec.worker_index,
                message.batch_id,
                tuple(
                    (item.position, f"worker-{spec.worker_index}:{item.prompt}")
                    for item in message.items
                ),
            )
        )


@pytest.mark.parametrize(
    ("count", "left_positions", "right_positions"),
    (
        (0, [], []),
        (1, [0], []),
        (4, [0, 1], [2, 3]),
        (5, [0, 1, 2], [3, 4]),
    ),
)
def test_split_contiguous_preserves_order_and_gives_odd_tail_to_worker_zero(
    count: int,
    left_positions: list[int],
    right_positions: list[int],
) -> None:
    left, right = split_contiguous(prompts(count))

    assert [item.position for item in left] == left_positions
    assert [item.position for item in right] == right_positions
    assert isinstance(left, tuple)
    assert isinstance(right, tuple)


def test_merge_worker_results_is_independent_of_worker_completion_order() -> None:
    merged = merge_worker_results(
        batch_id=9,
        expected_positions=[0, 1, 2, 3],
        messages=[
            WorkerResult(1, 9, ((2, "r2"), (3, "r3"))),
            WorkerResult(0, 9, ((0, "r0"), (1, "r1"))),
        ],
    )

    assert merged == [(0, "r0"), (1, "r1"), (2, "r2"), (3, "r3")]


@pytest.mark.parametrize(
    "messages",
    (
        [
            WorkerResult(0, 9, ((2, "r2"), (3, "r3"))),
            WorkerResult(1, 9, ((0, "r0"), (1, "r1"))),
        ],
        [
            WorkerResult(0, 9, ((0, "r0"),)),
            WorkerResult(1, 9, ((1, "r1"), (2, "r2"), (3, "r3"))),
        ],
        [
            WorkerResult(0, 9, ((1, "r1"), (0, "r0"))),
            WorkerResult(1, 9, ((2, "r2"), (3, "r3"))),
        ],
    ),
)
def test_merge_worker_results_rejects_wrong_shard_ownership_and_order(
    messages: list[WorkerResult],
) -> None:
    with pytest.raises(ValueError, match="worker .*positions"):
        merge_worker_results(
            batch_id=9,
            expected_positions=[0, 1, 2, 3],
            messages=messages,
        )


@pytest.mark.parametrize(
    ("messages", "match"),
    (
        (
            [
                WorkerResult(0, 8, ((0, "r0"),)),
                WorkerResult(1, 9, ((1, "r1"),)),
            ],
            "batch",
        ),
        (
            [
                WorkerResult(0, 9, ((0, "r0"),)),
                WorkerResult(0, 9, ((1, "r1"),)),
            ],
            "duplicate worker",
        ),
        (
            [
                WorkerResult(0, 9, ((0, "r0"),)),
                WorkerResult(1, 9, ((0, "again"),)),
            ],
            "duplicate position",
        ),
        (
            [
                WorkerResult(0, 9, ((0, "r0"),)),
                WorkerResult(1, 9, ()),
            ],
            "count mismatch",
        ),
        (
            [
                WorkerResult(0, 9, ((0, "r0"),)),
                WorkerResult(1, 9, ((2, "unknown"),)),
            ],
            "unknown position",
        ),
        (
            [
                WorkerResult(0, 9, ((0, "r0"), (2, "unknown"))),
                WorkerResult(1, 9, ()),
            ],
            "missing position",
        ),
        ([WorkerResult(0, 9, ((0, "r0"),))], "exactly two"),
        (
            [
                WorkerResult(0, 9, ((0, "r0"),)),
                "malformed",
            ],
            "malformed",
        ),
    ),
)
def test_merge_worker_results_rejects_protocol_failures(messages, match: str) -> None:
    with pytest.raises((RuntimeError, ValueError), match=match):
        merge_worker_results(
            batch_id=9,
            expected_positions=[0, 1],
            messages=messages,
        )


def test_merge_worker_results_raises_worker_error() -> None:
    error = WorkerError(1, 9, "generation failed", "trace details")

    with pytest.raises(RuntimeError, match="generation failed.*trace details"):
        merge_worker_results(
            batch_id=9,
            expected_positions=[0, 1],
            messages=[WorkerResult(0, 9, ((0, "r0"),)), error],
        )


def test_merge_worker_results_rejects_worker_error_with_unknown_worker() -> None:
    with pytest.raises(ValueError, match="worker index"):
        merge_worker_results(
            batch_id=9,
            expected_positions=[0, 1],
            messages=[
                WorkerResult(0, 9, ((0, "r0"),)),
                WorkerError(7, 9, "bad", "trace"),
            ],
        )


class RecordingGenerator:
    def __init__(self, responses: list[str] | None = None, error: Exception | None = None):
        self.responses = responses
        self.error = error
        self.calls: list[tuple[list[str], object]] = []

    def generate(self, prompt_texts: list[str], config: object) -> list[str]:
        self.calls.append((prompt_texts, config))
        if self.error is not None:
            raise self.error
        if self.responses is not None:
            return self.responses
        return [f"response:{prompt}" for prompt in prompt_texts]


def run_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    worker_index: int = 1,
    device: int = 7,
    requests: list[object],
    generator: RecordingGenerator,
):
    cache_root = tmp_path / f"gpu{worker_index}"
    request_queue: queue.Queue[object] = queue.Queue()
    response_queue: queue.Queue[object] = queue.Queue()
    for request in requests:
        request_queue.put(request)

    observed: dict[str, object] = {}

    def factory(**kwargs):
        observed["cuda"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        observed["cache"] = os.environ.get("VLLM_CACHE_ROOT")
        observed["cache_exists"] = cache_root.is_dir()
        observed["kwargs"] = kwargs
        return generator

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("VLLM_CACHE_ROOT", raising=False)
    worker_main(
        spec=WorkerSpec(worker_index, device, str(cache_root)),
        request_queue=request_queue,
        response_queue=response_queue,
        model_path="/models/qwen",
        gpu_memory_utilization=0.8,
        max_model_len=512,
        seed=17,
        max_new_tokens=64,
        temperature=0.2,
        top_p=0.95,
        enable_thinking=False,
        generator_factory=factory,
    )
    messages = []
    while not response_queue.empty():
        messages.append(response_queue.get_nowait())
    return observed, messages


@pytest.mark.parametrize(
    ("worker_index", "device"),
    ((0, 3), (1, 7)),
)
def test_worker_sets_environment_creates_distinct_cache_and_uses_runtime_kwargs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    worker_index: int,
    device: int,
) -> None:
    generator = RecordingGenerator()

    observed, messages = run_worker(
        tmp_path,
        monkeypatch,
        worker_index=worker_index,
        device=device,
        requests=[
            WorkerRequest(4, tuple(prompts(2))),
            _STOP,
        ],
        generator=generator,
    )

    assert observed == {
        "cuda": str(device),
        "cache": str(tmp_path / f"gpu{worker_index}"),
        "cache_exists": True,
        "kwargs": {
            "model_path": "/models/qwen",
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.8,
            "max_model_len": 512,
            "seed": 17,
        },
    }
    assert isinstance(messages[0], WorkerReady)
    assert messages[0].worker_index == worker_index
    assert messages[1] == WorkerResult(
        worker_index,
        4,
        ((0, "response:prompt-0"), (1, "response:prompt-1")),
    )
    prompt_texts, config = generator.calls[0]
    assert prompt_texts == ["prompt-0", "prompt-1"]
    assert vars(config) == {
        "max_new_tokens": 64,
        "temperature": 0.2,
        "top_p": 0.95,
        "enable_thinking": False,
    }


def test_worker_returns_empty_result_without_calling_generate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = RecordingGenerator()

    _, messages = run_worker(
        tmp_path,
        monkeypatch,
        requests=[WorkerRequest(5, ()), _STOP],
        generator=generator,
    )

    assert messages == [WorkerReady(1), WorkerResult(1, 5, ())]
    assert generator.calls == []


@pytest.mark.parametrize(
    ("generator", "match"),
    (
        (RecordingGenerator(responses=[]), "response count mismatch"),
        (RecordingGenerator(error=ValueError("model exploded")), "model exploded"),
    ),
)
def test_worker_emits_error_with_traceback_and_exits_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    generator: RecordingGenerator,
    match: str,
) -> None:
    cache_root = tmp_path / "gpu0"
    request_queue: queue.Queue[object] = queue.Queue()
    response_queue: queue.Queue[object] = queue.Queue()
    request_queue.put(WorkerRequest(12, (PositionedPrompt(3, "prompt"),)))

    with pytest.raises(SystemExit) as exc_info:
        worker_main(
            spec=WorkerSpec(0, 0, str(cache_root)),
            request_queue=request_queue,
            response_queue=response_queue,
            model_path="/model",
            gpu_memory_utilization=0.7,
            max_model_len=256,
            seed=0,
            max_new_tokens=32,
            temperature=0.0,
            top_p=1.0,
            enable_thinking=False,
            generator_factory=lambda **kwargs: generator,
        )

    assert exc_info.value.code == 1
    assert response_queue.get_nowait() == WorkerReady(0)
    error = response_queue.get_nowait()
    assert isinstance(error, WorkerError)
    assert error.worker_index == 0
    assert error.batch_id == 12
    assert match in error.message
    assert match in error.traceback


def test_worker_initialization_error_has_no_batch_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response_queue: queue.Queue[object] = queue.Queue()

    def broken_factory(**kwargs):
        raise RuntimeError("initialization failed")

    with pytest.raises(SystemExit):
        worker_main(
            spec=WorkerSpec(0, 0, str(tmp_path / "gpu0")),
            request_queue=queue.Queue(),
            response_queue=response_queue,
            model_path="/model",
            gpu_memory_utilization=0.8,
            max_model_len=512,
            seed=0,
            max_new_tokens=64,
            temperature=0.2,
            top_p=0.95,
            enable_thinking=False,
            generator_factory=broken_factory,
        )

    error = response_queue.get_nowait()
    assert isinstance(error, WorkerError)
    assert error.batch_id is None
    assert "initialization failed" in error.traceback


class FakeQueue:
    def __init__(
        self,
        *,
        put_error: Exception | None = None,
        put_hook=None,
        join_thread_hook=None,
    ) -> None:
        self.items: list[object] = []
        self.put_error = put_error
        self.put_hook = put_hook
        self.join_thread_hook = join_thread_hook
        self.close_calls = 0
        self.join_thread_calls = 0
        self.cancel_join_thread_calls = 0

    def put(self, item: object) -> None:
        if self.put_hook is not None:
            self.put_hook()
        if self.put_error is not None:
            raise self.put_error
        self.items.append(item)

    def get(self, timeout: float | None = None) -> object:
        if not self.items:
            raise queue.Empty
        return self.items.pop(0)

    def get_nowait(self) -> object:
        return self.get(timeout=0)

    def close(self) -> None:
        self.close_calls += 1

    def join_thread(self) -> None:
        self.join_thread_calls += 1
        if self.join_thread_hook is not None:
            self.join_thread_hook()

    def cancel_join_thread(self) -> None:
        self.cancel_join_thread_calls += 1


class FakeProcess:
    def __init__(
        self,
        *,
        target,
        args,
        stubborn: bool = False,
        terminate_exits: bool = True,
        kill_exits: bool = True,
        start_error: Exception | None = None,
        start_hook=None,
        join_hook=None,
    ) -> None:
        self.target = target
        self.args = args
        self.stubborn = stubborn
        self.terminate_exits = terminate_exits
        self.kill_exits = kill_exits
        self.start_error = start_error
        self.start_hook = start_hook
        self.join_hook = join_hook
        self.exitcode: int | None = None
        self.started = False
        self.join_timeouts: list[float | None] = []
        self.terminated = False
        self.killed = False
        self.close_calls = 0

    def start(self) -> None:
        if self.start_error is not None:
            raise self.start_error
        self.started = True
        if self.start_hook is not None:
            self.start_hook()

    def join(self, timeout: float | None = None) -> None:
        if not self.started:
            raise AssertionError("cannot join process before it is started")
        self.join_timeouts.append(timeout)
        if self.join_hook is not None:
            self.join_hook(timeout)
        if not self.stubborn:
            self.exitcode = 0

    def terminate(self) -> None:
        if not self.started:
            raise AttributeError("process has no pid")
        self.terminated = True
        if self.terminate_exits:
            self.exitcode = -15

    def kill(self) -> None:
        if not self.started:
            raise AttributeError("process has no pid")
        self.killed = True
        if self.kill_exits:
            self.exitcode = -9

    def close(self) -> None:
        self.close_calls += 1

    def is_alive(self) -> bool:
        return self.started and self.exitcode is None


class FakeContext:
    def __init__(
        self,
        *,
        stubborn: bool = False,
        terminate_exits: bool = True,
        kill_exits: bool = True,
        start_error: Exception | None = None,
        queues: list[FakeQueue] | None = None,
        start_hook=None,
        join_hook=None,
    ) -> None:
        self.queues = queues or [FakeQueue(), FakeQueue(), FakeQueue()]
        self._next_queue = 0
        self.processes: list[FakeProcess] = []
        self.stubborn = stubborn
        self.terminate_exits = terminate_exits
        self.kill_exits = kill_exits
        self.start_error = start_error
        self.start_hook = start_hook
        self.join_hook = join_hook

    @property
    def request_queues(self) -> list[FakeQueue]:
        return self.queues[:2]

    @property
    def response_queue(self) -> FakeQueue:
        return self.queues[2]

    def Queue(self) -> FakeQueue:
        result = self.queues[self._next_queue]
        self._next_queue += 1
        return result

    def Process(self, *, target, args) -> FakeProcess:
        process = FakeProcess(
            target=target,
            args=args,
            stubborn=self.stubborn,
            terminate_exits=self.terminate_exits,
            kill_exits=self.kill_exits,
            start_error=self.start_error,
            start_hook=self.start_hook,
            join_hook=self.join_hook,
        )
        self.processes.append(process)
        return process


class StepClock:
    def __init__(self, step: float = 0.1) -> None:
        self.value = 0.0
        self.step = step

    def __call__(self) -> float:
        current = self.value
        self.value += self.step
        return current

    def advance(self, amount: float) -> None:
        self.value += amount


class ManualClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, amount: float) -> None:
        self.value += amount


def make_engine(
    context: FakeContext,
    *,
    timeout_seconds: float = 1.0,
    clock=None,
) -> ParallelVLLMEngine:
    return ParallelVLLMEngine(
        model_path="/model",
        worker_specs=(
            WorkerSpec(0, 0, "/cache/gpu0"),
            WorkerSpec(1, 1, "/cache/gpu1"),
        ),
        gpu_memory_utilization=0.8,
        max_model_len=512,
        seed=0,
        max_new_tokens=64,
        temperature=0.2,
        top_p=0.95,
        enable_thinking=False,
        timeout_seconds=timeout_seconds,
        context=context,
        worker_target=lambda *args: None,
        monotonic=clock or StepClock(0.01),
        poll_interval=0.01,
        shutdown_timeout=0.01,
    )


def test_start_waits_for_two_unique_ready_messages_in_any_order() -> None:
    context = FakeContext()
    context.response_queue.put(WorkerReady(1))
    context.response_queue.put(WorkerReady(0))
    engine = make_engine(context)

    assert engine.start() is engine

    assert len(context.processes) == 2
    assert all(process.started for process in context.processes)


@pytest.mark.parametrize(
    ("messages", "match"),
    (
        ([WorkerReady(0), WorkerReady(0)], "duplicate"),
        ([WorkerReady(0), WorkerReady(4)], "worker index"),
        ([WorkerReady(0), "bad"], "malformed"),
        ([None], "malformed"),
        ([WorkerError(1, None, "init failed", "trace")], "init failed"),
        ([WorkerError(8, None, "init failed", "trace")], "worker index"),
        ([WorkerError(True, None, "init failed", "trace")], "worker index"),
    ),
)
def test_start_rejects_duplicate_unknown_error_and_malformed_messages(
    messages: list[object],
    match: str,
) -> None:
    context = FakeContext()
    for message in messages:
        context.response_queue.put(message)
    engine = make_engine(context)

    with pytest.raises((RuntimeError, ValueError), match=match):
        engine.start()

    assert all(process.exitcode is not None for process in context.processes)


def test_start_timeout_closes_both_workers_with_one_deadline() -> None:
    context = FakeContext()
    engine = make_engine(context, timeout_seconds=0.2, clock=StepClock(0.11))

    with pytest.raises(TimeoutError, match="startup"):
        engine.start()

    assert all(process.exitcode is not None for process in context.processes)
    assert all(request.items for request in context.request_queues)


def test_start_deadline_begins_before_first_process_start() -> None:
    clock = ManualClock()
    context = FakeContext(start_hook=lambda: clock.advance(0.6))
    context.response_queue.put(WorkerReady(0))
    context.response_queue.put(WorkerReady(1))
    engine = make_engine(context, timeout_seconds=1.0, clock=clock)

    with pytest.raises(TimeoutError, match="startup"):
        engine.start()

    assert clock.value >= 1.2


def test_start_failure_cleans_unstarted_process_without_masking_error() -> None:
    context = FakeContext(start_error=RuntimeError("start failed"))
    engine = make_engine(context)

    with pytest.raises(RuntimeError, match="start failed"):
        engine.start()

    assert len(context.processes) == 1
    assert context.processes[0].started is False
    assert context.processes[0].close_calls == 1
    assert all(item.close_calls == 1 for item in context.queues)


def test_start_detects_dead_worker_without_waiting_for_timeout() -> None:
    context = FakeContext()
    context.response_queue.put(WorkerReady(0))
    engine = make_engine(context, timeout_seconds=100.0, clock=StepClock(0.01))

    original_process = context.Process

    def process_factory(*, target, args):
        process = original_process(target=target, args=args)
        if len(context.processes) == 2:
            process.exitcode = 7
        return process

    context.Process = process_factory

    with pytest.raises(RuntimeError, match="worker 1.*exit 7"):
        engine.start()


def start_engine(context: FakeContext) -> ParallelVLLMEngine:
    context.response_queue.put(WorkerReady(0))
    context.response_queue.put(WorkerReady(1))
    engine = make_engine(context)
    engine.start()
    return engine


def test_generate_sends_both_contiguous_requests_including_empty_tail_and_merges() -> None:
    context = FakeContext()
    engine = start_engine(context)
    context.response_queue.put(WorkerResult(1, 1, ()))
    context.response_queue.put(WorkerResult(0, 1, ((8, "answer"),)))

    result = engine.generate(1, [PositionedPrompt(8, "prompt")])

    assert result == [(8, "answer")]
    assert context.request_queues[0].items == [
        WorkerRequest(1, (PositionedPrompt(8, "prompt"),))
    ]
    assert context.request_queues[1].items == [WorkerRequest(1, ())]


def test_generate_rejects_results_with_swapped_worker_shards() -> None:
    context = FakeContext()
    engine = start_engine(context)
    context.response_queue.put(
        WorkerResult(0, 1, ((2, "r2"), (3, "r3")))
    )
    context.response_queue.put(
        WorkerResult(1, 1, ((0, "r0"), (1, "r1")))
    )

    with pytest.raises(ValueError, match="worker .*positions"):
        engine.generate(1, prompts(4))


def test_generate_deadline_begins_before_first_queue_put() -> None:
    clock = ManualClock()
    request_queues = [
        FakeQueue(put_hook=lambda: clock.advance(0.6)),
        FakeQueue(put_hook=lambda: clock.advance(0.6)),
    ]
    context = FakeContext(queues=[*request_queues, FakeQueue()])
    engine = start_engine(context)
    context.response_queue.put(WorkerResult(0, 1, ((0, "r0"),)))
    context.response_queue.put(WorkerResult(1, 1, ((1, "r1"),)))
    engine._monotonic = clock
    engine.timeout_seconds = 1.0

    with pytest.raises(TimeoutError, match="batch 1"):
        engine.generate(1, prompts(2))

    assert clock.value >= 1.2


@pytest.mark.parametrize("failing_queue_index", (0, 1))
def test_generate_put_failure_closes_engine_and_consumes_batch_id(
    failing_queue_index: int,
) -> None:
    request_queues = [FakeQueue(), FakeQueue()]
    request_queues[failing_queue_index].put_error = RuntimeError("put failed")
    context = FakeContext(
        queues=[*request_queues, FakeQueue()]
    )
    engine = start_engine(context)

    with pytest.raises(RuntimeError, match="put failed"):
        engine.generate(11, prompts(2))

    assert engine._last_batch_id == 11
    assert engine._closed is True
    assert all(process.exitcode is not None for process in context.processes)


def test_generate_raises_worker_error_and_closes_workers() -> None:
    context = FakeContext()
    engine = start_engine(context)
    context.response_queue.put(WorkerError(1, 2, "generation failed", "trace"))

    with pytest.raises(RuntimeError, match="generation failed"):
        engine.generate(2, prompts(2))

    assert all(process.exitcode is not None for process in context.processes)


@pytest.mark.parametrize(
    ("messages", "match"),
    (
        ([WorkerReady(0)], "malformed"),
        ([WorkerResult(0, 8, ())], "batch"),
        ([WorkerResult(3, 9, ())], "worker index"),
        ([WorkerError(3, 9, "bad", "trace")], "worker index"),
    ),
)
def test_generate_rejects_malformed_wrong_batch_and_unknown_worker(
    messages: list[object],
    match: str,
) -> None:
    context = FakeContext()
    engine = start_engine(context)
    for message in messages:
        context.response_queue.put(message)

    with pytest.raises((RuntimeError, ValueError), match=match):
        engine.generate(9, [])


def test_generate_detects_dead_worker_during_batch() -> None:
    context = FakeContext()
    engine = start_engine(context)
    context.processes[0].exitcode = 3

    with pytest.raises(RuntimeError, match="worker 0.*exit 3"):
        engine.generate(1, [])


def test_generate_prefers_queued_worker_error_over_dead_worker_status() -> None:
    context = FakeContext()
    engine = start_engine(context)
    context.processes[0].exitcode = 3
    context.response_queue.put(
        WorkerError(0, 1, "specific generation failure", "specific trace")
    )

    with pytest.raises(RuntimeError, match="specific generation failure"):
        engine.generate(1, [])


class FinalDrainQueue(FakeQueue):
    def __init__(self, final_message: object) -> None:
        super().__init__()
        self.final_message = final_message
        self.blocking_get_calls = 0
        self.nowait_calls = 0

    def get(self, timeout: float | None = None) -> object:
        self.blocking_get_calls += 1
        raise queue.Empty

    def get_nowait(self) -> object:
        self.nowait_calls += 1
        if self.nowait_calls == 1:
            raise queue.Empty
        message = self.final_message
        self.final_message = None
        if message is None:
            raise queue.Empty
        return message


def test_generate_final_queue_drain_prefers_error_before_dead_worker() -> None:
    context = FakeContext()
    engine = start_engine(context)
    final_queue = FinalDrainQueue(
        WorkerError(1, 2, "late detailed failure", "late trace")
    )
    engine._response_queue = final_queue
    context.processes[1].exitcode = 9

    with pytest.raises(RuntimeError, match="late detailed failure"):
        engine.generate(2, [])

    assert final_queue.blocking_get_calls == 1
    assert final_queue.nowait_calls == 2


def test_generate_times_out_and_closes_workers() -> None:
    context = FakeContext()
    engine = start_engine(context)
    engine._monotonic = StepClock(0.6)
    engine.timeout_seconds = 1.0

    with pytest.raises(TimeoutError, match="batch 1"):
        engine.generate(1, prompts(2))

    assert all(process.exitcode is not None for process in context.processes)


def test_state_errors_and_batch_ids_are_strictly_increasing() -> None:
    context = FakeContext()
    engine = make_engine(context)

    with pytest.raises(RuntimeError, match="not started"):
        engine.generate(1, [])

    context.response_queue.put(WorkerReady(0))
    context.response_queue.put(WorkerReady(1))
    engine.start()
    with pytest.raises(RuntimeError, match="already started"):
        engine.start()

    context.response_queue.put(WorkerResult(0, 5, ()))
    context.response_queue.put(WorkerResult(1, 5, ()))
    assert engine.generate(5, []) == []
    with pytest.raises(ValueError, match="strictly increasing"):
        engine.generate(5, [])
    with pytest.raises(ValueError, match="strictly increasing"):
        engine.generate(4, [])


def test_close_is_safe_before_start_and_when_called_twice() -> None:
    context = FakeContext()
    engine = make_engine(context)

    engine.close()
    engine.close()

    assert context.processes == []
    assert all(item.close_calls == 1 for item in context.queues)
    assert all(
        item.join_thread_calls + item.cancel_join_thread_calls == 1
        for item in context.queues
    )


def test_close_sends_stop_joins_terminates_stubborn_workers_and_joins_again() -> None:
    context = FakeContext(stubborn=True)
    engine = start_engine(context)

    engine.close()
    engine.close()

    assert all(len(request.items) == 1 for request in context.request_queues)
    assert all(type(request.items[0]).__name__ == "_StopWorker" for request in context.request_queues)
    assert all(process.terminated for process in context.processes)
    assert all(len(process.join_timeouts) == 2 for process in context.processes)
    assert all(process.close_calls == 1 for process in context.processes)
    assert all(item.close_calls == 1 for item in context.queues)
    assert all(
        item.join_thread_calls + item.cancel_join_thread_calls == 1
        for item in context.queues
    )


def test_close_uses_one_deadline_then_kills_and_releases_resources() -> None:
    clock = ManualClock()

    def consume_join_time(timeout: float | None) -> None:
        clock.advance(min(timeout or 0.0, 0.3))

    context = FakeContext(
        stubborn=True,
        terminate_exits=False,
        kill_exits=True,
        join_hook=consume_join_time,
    )
    engine = start_engine(context)
    engine._monotonic = clock
    engine._shutdown_timeout = 1.0

    engine.close()
    engine.close()

    assert clock.value <= 1.0
    assert all(process.terminated for process in context.processes)
    assert all(process.killed for process in context.processes)
    assert all(process.exitcode == -9 for process in context.processes)
    assert all(process.close_calls == 1 for process in context.processes)
    assert all(item.close_calls == 1 for item in context.queues)
    assert all(
        item.join_thread_calls + item.cancel_join_thread_calls == 1
        for item in context.queues
    )


def test_close_raises_for_orphan_after_kill_but_second_close_is_noop() -> None:
    context = FakeContext(
        stubborn=True,
        terminate_exits=False,
        kill_exits=False,
    )
    engine = start_engine(context)

    with pytest.raises(RuntimeError, match="still alive"):
        engine.close()

    engine.close()
    assert all(process.terminated for process in context.processes)
    assert all(process.killed for process in context.processes)
    assert all(item.close_calls == 1 for item in context.queues)
    assert all(
        item.join_thread_calls + item.cancel_join_thread_calls == 1
        for item in context.queues
    )


def test_close_cancels_queue_join_threads_without_exceeding_deadline() -> None:
    clock = ManualClock()

    def exhaust_deadline(timeout: float | None) -> None:
        clock.advance(timeout or 0.0)

    queues = [
        FakeQueue(join_thread_hook=lambda: clock.advance(10.0))
        for _ in range(3)
    ]
    context = FakeContext(
        stubborn=True,
        terminate_exits=False,
        kill_exits=True,
        queues=queues,
        join_hook=exhaust_deadline,
    )
    engine = start_engine(context)
    engine._monotonic = clock
    engine._shutdown_timeout = 0.5

    engine.close()

    assert clock.value <= 0.5
    assert all(item.close_calls == 1 for item in queues)
    assert all(item.join_thread_calls == 0 for item in queues)
    assert all(item.cancel_join_thread_calls == 1 for item in queues)


def test_context_manager_starts_and_closes_engine() -> None:
    context = FakeContext()
    context.response_queue.put(WorkerReady(0))
    context.response_queue.put(WorkerReady(1))
    engine = make_engine(context)

    with engine as entered:
        assert entered is engine

    assert all(process.exitcode is not None for process in context.processes)


def test_constructor_requires_exactly_two_ordered_worker_specs() -> None:
    context = FakeContext()

    with pytest.raises(ValueError, match="exactly two"):
        ParallelVLLMEngine(
            model_path="/model",
            worker_specs=(WorkerSpec(0, 0, "/cache/gpu0"),),
            gpu_memory_utilization=0.8,
            max_model_len=512,
            seed=0,
            max_new_tokens=64,
            temperature=0.2,
            top_p=0.95,
            enable_thinking=False,
            timeout_seconds=1.0,
            context=context,
        )

    with pytest.raises(ValueError, match="worker indices"):
        ParallelVLLMEngine(
            model_path="/model",
            worker_specs=(
                WorkerSpec(1, 1, "/cache/gpu1"),
                WorkerSpec(0, 0, "/cache/gpu0"),
            ),
            gpu_memory_utilization=0.8,
            max_model_len=512,
            seed=0,
            max_new_tokens=64,
            temperature=0.2,
            top_p=0.95,
            enable_thinking=False,
            timeout_seconds=1.0,
            context=context,
        )


@pytest.mark.parametrize(
    ("specs", "match"),
    (
        (
            (
                WorkerSpec(0, 0, "/cache/gpu0"),
                WorkerSpec(1, 0, "/cache/gpu1"),
            ),
            "devices",
        ),
        (
            (
                WorkerSpec(0, 0, "/cache/shared"),
                WorkerSpec(1, 1, "/cache/shared"),
            ),
            "cache roots",
        ),
    ),
)
def test_constructor_rejects_duplicate_devices_and_cache_roots(
    specs: tuple[WorkerSpec, WorkerSpec],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        ParallelVLLMEngine(
            model_path="/model",
            worker_specs=specs,
            gpu_memory_utilization=0.8,
            max_model_len=512,
            seed=0,
            max_new_tokens=64,
            temperature=0.2,
            top_p=0.95,
            enable_thinking=False,
            timeout_seconds=1.0,
            context=FakeContext(),
        )


def test_real_spawn_context_round_trip_and_shutdown(tmp_path: Path) -> None:
    try:
        context = multiprocessing.get_context("spawn")
    except ValueError:
        pytest.skip("multiprocessing spawn context is unavailable")

    engine = ParallelVLLMEngine(
        model_path="/unused",
        worker_specs=(
            WorkerSpec(0, 0, str(tmp_path / "gpu0")),
            WorkerSpec(1, 1, str(tmp_path / "gpu1")),
        ),
        gpu_memory_utilization=0.8,
        max_model_len=128,
        seed=0,
        max_new_tokens=16,
        temperature=0.0,
        top_p=1.0,
        enable_thinking=False,
        timeout_seconds=5.0,
        context=context,
        worker_target=spawn_fake_worker,
        poll_interval=0.01,
        shutdown_timeout=2.0,
    )

    engine.start()
    processes = tuple(engine._processes)
    process_ids = {process.pid for process in processes}
    result = engine.generate(1, prompts(3))
    engine.close()

    assert result == [
        (0, "worker-0:prompt-0"),
        (1, "worker-0:prompt-1"),
        (2, "worker-1:prompt-2"),
    ]
    active_ids = {process.pid for process in multiprocessing.active_children()}
    assert process_ids.isdisjoint(active_ids)


def test_parallel_module_has_no_module_scope_torch_vllm_or_legacy_generation_imports() -> None:
    module_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "generation"
        / "parallel_vllm.py"
    )
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    forbidden = {
        "torch",
        "vllm",
        "post_train.src.countdown.generation",
    }
    imported: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    assert imported.isdisjoint(forbidden)
