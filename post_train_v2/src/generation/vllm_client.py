"""Ordered chat-based vLLM generation with per-request sampling seeds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from post_train_v2.src.countdown.prompts import build_chat_messages


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int
    temperature: float
    top_p: float
    enable_thinking: bool = False


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str) or not self.prompt:
            raise ValueError("prompt must be a nonempty string")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative exact integer")


class VLLMGenerator:
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float | None = None,
        max_model_len: int | None = None,
        seed: int | None = None,
    ) -> None:
        from vllm import LLM

        kwargs: dict[str, Any] = {
            "model": model_path,
            "tensor_parallel_size": tensor_parallel_size,
            "trust_remote_code": True,
        }
        if gpu_memory_utilization is not None:
            kwargs["gpu_memory_utilization"] = gpu_memory_utilization
        if max_model_len is not None:
            kwargs["max_model_len"] = max_model_len
        if seed is not None:
            kwargs["seed"] = seed
        self.llm = LLM(**kwargs)

    def generate(
        self,
        requests: list[GenerationRequest],
        config: GenerationConfig,
    ) -> list[str]:
        return [
            str(record["text"])
            for record in self.generate_with_metadata(requests, config)
        ]

    def generate_with_metadata(
        self,
        requests: list[GenerationRequest],
        config: GenerationConfig,
    ) -> list[dict[str, object]]:
        from vllm import SamplingParams

        if not requests:
            return []
        sampling_params = [
            SamplingParams(
                temperature=config.temperature,
                top_p=config.top_p,
                max_tokens=config.max_new_tokens,
                seed=request.seed,
            )
            for request in requests
        ]
        outputs = self.llm.chat(
            messages=[
                build_chat_messages(request.prompt) for request in requests
            ],
            sampling_params=sampling_params,
            use_tqdm=False,
            chat_template_kwargs={
                "enable_thinking": config.enable_thinking,
            },
        )
        return [_completion_metadata(item.outputs[0]) for item in outputs]


def _completion_metadata(completion: Any) -> dict[str, object]:
    token_ids = getattr(completion, "token_ids", None)
    return {
        "text": getattr(completion, "text", ""),
        "finish_reason": getattr(completion, "finish_reason", None),
        "token_count": len(token_ids) if token_ids is not None else None,
        "stop_reason": getattr(completion, "stop_reason", None),
    }
