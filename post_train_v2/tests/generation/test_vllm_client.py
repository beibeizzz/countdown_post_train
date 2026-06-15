from __future__ import annotations

import sys
import types
import ast
from pathlib import Path

from post_train_v2.src.generation.seeding import derive_request_seed
from post_train_v2.src.generation.vllm_client import (
    GenerationConfig,
    GenerationRequest,
    VLLMGenerator,
)


def test_request_seed_is_stable_and_rollout_specific() -> None:
    seed = derive_request_seed(
        global_seed=42,
        stage="teacher",
        sample_id="train-000123",
        rollout_index=0,
    )

    assert seed == derive_request_seed(42, "teacher", "train-000123", 0)
    assert seed != derive_request_seed(42, "teacher", "train-000123", 1)


def test_generate_with_metadata_uses_ordered_chat_and_per_request_seeds(
    monkeypatch,
) -> None:
    created_sampling_params = []
    chat_calls = []

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            created_sampling_params.append(self)

    class FakeCompletion:
        def __init__(self, text: str):
            self.text = text
            self.token_ids = [1, 2]
            self.finish_reason = "stop"
            self.stop_reason = None

    class FakeOutput:
        def __init__(self, text: str):
            self.outputs = [FakeCompletion(text)]

    class FakeLLM:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def chat(self, **kwargs):
            chat_calls.append(kwargs)
            return [FakeOutput("first"), FakeOutput("second")]

    monkeypatch.setitem(
        sys.modules,
        "vllm",
        types.SimpleNamespace(LLM=FakeLLM, SamplingParams=FakeSamplingParams),
    )
    generator = VLLMGenerator(
        model_path="/model",
        tensor_parallel_size=1,
        seed=7,
    )
    requests = [
        GenerationRequest(prompt="prompt-a", seed=101),
        GenerationRequest(prompt="prompt-b", seed=202),
    ]

    records = generator.generate_with_metadata(
        requests,
        GenerationConfig(
            max_new_tokens=256,
            temperature=0.2,
            top_p=0.95,
            enable_thinking=False,
        ),
    )

    assert [record["text"] for record in records] == ["first", "second"]
    assert chat_calls == [
        {
            "messages": [
                [{"role": "user", "content": "prompt-a"}],
                [{"role": "user", "content": "prompt-b"}],
            ],
            "sampling_params": created_sampling_params,
            "use_tqdm": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ]
    assert [params.kwargs["seed"] for params in created_sampling_params] == [
        101,
        202,
    ]
    assert all(
        params.kwargs["max_tokens"] == 256
        and params.kwargs["temperature"] == 0.2
        and params.kwargs["top_p"] == 0.95
        for params in created_sampling_params
    )


def test_v2_generation_runtime_has_no_v1_imports() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    runtime_files = [
        *sorted((repo_root / "post_train_v2/src/generation").glob("*.py")),
        repo_root
        / "post_train_v2/scripts/generation/build_teacher_pool.py",
    ]

    offenders = []
    for path in runtime_files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (
                node.module == "post_train"
                or (node.module or "").startswith("post_train.")
            ):
                offenders.append(path.relative_to(repo_root).as_posix())
            if isinstance(node, ast.Import) and any(
                alias.name == "post_train"
                or alias.name.startswith("post_train.")
                for alias in node.names
            ):
                offenders.append(path.relative_to(repo_root).as_posix())

    assert offenders == []
