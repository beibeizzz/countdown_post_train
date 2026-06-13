import sys
import types

import pytest

from post_train.src.countdown.generation import GenerationConfig, VLLMGenerator, apply_chat_template


class ThinkingTokenizer:
    def apply_chat_template(self, messages, enable_thinking, **kwargs):
        return {
            "messages": messages,
            "enable_thinking": enable_thinking,
            "kwargs": kwargs,
        }


class LegacyTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return {
            "messages": messages,
            "kwargs": {
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
            },
        }


class KwargsTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return {
            "messages": messages,
            "kwargs": kwargs,
        }


class BrokenThinkingTokenizer:
    def apply_chat_template(self, messages, enable_thinking, **kwargs):
        raise TypeError("internal template failure")


def test_generation_config_defaults_disable_thinking():
    config = GenerationConfig(max_new_tokens=256, temperature=0.2, top_p=0.95)

    assert config.enable_thinking is False


def test_apply_chat_template_passes_enable_thinking_when_supported():
    rendered = apply_chat_template(ThinkingTokenizer(), "solve this", enable_thinking=True)

    assert rendered["messages"] == [{"role": "user", "content": "solve this"}]
    assert rendered["enable_thinking"] is True
    assert rendered["kwargs"] == {"tokenize": False, "add_generation_prompt": True}


def test_apply_chat_template_falls_back_for_legacy_tokenizers():
    rendered = apply_chat_template(LegacyTokenizer(), "solve this", enable_thinking=True)

    assert rendered["messages"] == [{"role": "user", "content": "solve this"}]
    assert rendered["kwargs"] == {"tokenize": False, "add_generation_prompt": True}


def test_apply_chat_template_passes_enable_thinking_to_kwargs_tokenizers():
    rendered = apply_chat_template(KwargsTokenizer(), "solve this", enable_thinking=True)

    assert rendered["messages"] == [{"role": "user", "content": "solve this"}]
    assert rendered["kwargs"] == {
        "enable_thinking": True,
        "tokenize": False,
        "add_generation_prompt": True,
    }


def test_apply_chat_template_propagates_internal_type_error_when_supported():
    with pytest.raises(TypeError, match="internal template failure"):
        apply_chat_template(BrokenThinkingTokenizer(), "solve this", enable_thinking=True)


class FakeCompletion:
    def __init__(self):
        self.text = "generated"
        self.finish_reason = "length"
        self.token_ids = [1, 2, 3]
        self.stop_reason = None


class FakeRequestOutput:
    outputs = [FakeCompletion()]


class FakeLLM:
    def chat(self, messages, sampling_params, use_tqdm, chat_template_kwargs):
        assert messages == [[{"role": "user", "content": "prompt"}]]
        assert use_tqdm is False
        assert chat_template_kwargs == {"enable_thinking": False}
        sampling = sampling_params
        assert sampling.max_tokens == 3
        return [FakeRequestOutput()]


class FakeSamplingParams:
    def __init__(self, temperature, top_p, max_tokens):
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens


def test_vllm_generator_uses_exact_legacy_llm_kwargs(monkeypatch):
    calls = []

    class RecordingLLM:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=RecordingLLM))

    VLLMGenerator("/model")

    assert calls == [
        {
            "model": "/model",
            "tensor_parallel_size": 1,
            "trust_remote_code": True,
        }
    ]


def test_vllm_generator_forwards_all_configured_optional_llm_kwargs(monkeypatch):
    calls = []

    class RecordingLLM:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=RecordingLLM))

    VLLMGenerator(
        "/model",
        tensor_parallel_size=2,
        gpu_memory_utilization=0.75,
        max_model_len=4096,
        seed=0,
    )

    assert calls == [
        {
            "model": "/model",
            "tensor_parallel_size": 2,
            "trust_remote_code": True,
            "gpu_memory_utilization": 0.75,
            "max_model_len": 4096,
            "seed": 0,
        }
    ]


def test_vllm_generate_with_metadata_extracts_finish_reason_and_token_count(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(SamplingParams=FakeSamplingParams))
    generator = VLLMGenerator.__new__(VLLMGenerator)
    generator.llm = FakeLLM()

    records = generator.generate_with_metadata(
        ["prompt"],
        GenerationConfig(max_new_tokens=3, temperature=0.7, top_p=0.9),
    )

    assert records == [
        {
            "text": "generated",
            "finish_reason": "length",
            "token_count": 3,
            "stop_reason": None,
        }
    ]
