from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Protocol

from post_train.src.countdown.prompts import build_chat_messages


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int
    temperature: float
    top_p: float
    enable_thinking: bool = False


class TextGenerator(Protocol):
    def generate(self, prompts: list[str], config: GenerationConfig) -> list[str]:
        ...


def apply_chat_template(tokenizer, prompt: str, enable_thinking: bool) -> str:
    messages = build_chat_messages(prompt)
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if _supports_enable_thinking(tokenizer.apply_chat_template):
        return tokenizer.apply_chat_template(messages, enable_thinking=enable_thinking, **kwargs)
    return tokenizer.apply_chat_template(messages, **kwargs)


def _supports_enable_thinking(apply_chat_template) -> bool:
    try:
        signature = inspect.signature(apply_chat_template)
    except (TypeError, ValueError):
        return True

    return any(
        parameter.name == "enable_thinking" or parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


class VLLMGenerator:
    def __init__(self, model_path: str, tensor_parallel_size: int = 1):
        from vllm import LLM

        self.llm = LLM(model=model_path, tensor_parallel_size=tensor_parallel_size, trust_remote_code=True)

    def generate(self, prompts: list[str], config: GenerationConfig) -> list[str]:
        return [str(record["text"]) for record in self.generate_with_metadata(prompts, config)]

    def generate_with_metadata(self, prompts: list[str], config: GenerationConfig) -> list[dict[str, object]]:
        from vllm import SamplingParams

        sampling = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_new_tokens,
        )
        conversations = [build_chat_messages(prompt) for prompt in prompts]
        outputs = self.llm.chat(
            messages=conversations,
            sampling_params=sampling,
            use_tqdm=False,
            chat_template_kwargs={"enable_thinking": config.enable_thinking},
        )
        return [_completion_metadata(item.outputs[0]) for item in outputs]


def _completion_metadata(completion) -> dict[str, object]:
    token_ids = getattr(completion, "token_ids", None)
    token_count = len(token_ids) if token_ids is not None else None
    return {
        "text": getattr(completion, "text", ""),
        "finish_reason": getattr(completion, "finish_reason", None),
        "token_count": token_count,
        "stop_reason": getattr(completion, "stop_reason", None),
    }
