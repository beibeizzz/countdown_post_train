from __future__ import annotations

from post_train_v2.src.generation.metadata import (
    GenerationRecord,
    classify_truncation,
)
from post_train_v2.src.generation.parallel_vllm import (
    ParallelVLLMEngine,
    PositionedPrompt,
    WorkerGeneration,
    WorkerResult,
)
from post_train_v2.tests.generation.test_parallel_vllm import (
    FakeContext,
    make_engine,
)


def test_classify_truncation_uses_finish_reason_first():
    record = GenerationRecord(
        text="<answer>1+1</answer>",
        finish_reason="stop",
        token_count=256,
        stop_reason=None,
    )

    assert classify_truncation(record, max_new_tokens=256).truncated is False
    assert classify_truncation(
        GenerationRecord(
            text="<answer>1+1</answer>",
            finish_reason="length",
            token_count=8,
            stop_reason=None,
        ),
        max_new_tokens=256,
    ).truncated is True


def test_classify_truncation_uses_token_count_only_without_finish_metadata():
    assert classify_truncation(
        GenerationRecord(
            text="partial",
            finish_reason=None,
            token_count=256,
            stop_reason=None,
        ),
        max_new_tokens=256,
    ).truncated is True
    assert classify_truncation(
        GenerationRecord(
            text="short",
            finish_reason=None,
            token_count=12,
            stop_reason=None,
        ),
        max_new_tokens=256,
    ).truncated is False


def test_parallel_generate_default_still_returns_strings():
    context = FakeContext()
    engine = make_engine(context=context)
    engine._started = True
    context.response_queue.put(WorkerResult(0, 1, ((0, "a"),)))
    context.response_queue.put(WorkerResult(1, 1, ((1, "b"),)))

    assert engine.generate(1, [PositionedPrompt(0, "pa"), PositionedPrompt(1, "pb")]) == [
        (0, "a"),
        (1, "b"),
    ]


def test_parallel_generate_with_metadata_returns_generation_records():
    context = FakeContext()
    engine = make_engine(context=context)
    engine._started = True
    context.response_queue.put(
        WorkerResult(
            0,
            1,
            (
                WorkerGeneration(
                    0,
                    GenerationRecord(
                        text="a",
                        finish_reason="stop",
                        token_count=1,
                        stop_reason=None,
                    ),
                ),
            ),
        )
    )
    context.response_queue.put(
        WorkerResult(
            1,
            1,
            (
                WorkerGeneration(
                    1,
                    GenerationRecord(
                        text="b",
                        finish_reason="length",
                        token_count=256,
                        stop_reason=None,
                    ),
                ),
            ),
        )
    )

    result = engine.generate(
        1,
        [PositionedPrompt(0, "pa"), PositionedPrompt(1, "pb")],
        include_metadata=True,
    )

    assert [item[0] for item in result] == [0, 1]
    assert [item[1].text for item in result] == ["a", "b"]
    assert [item[1].truncated for item in result] == [False, True]
