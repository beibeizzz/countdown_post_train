from __future__ import annotations

import argparse
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a deterministic vLLM chat smoke test.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, choices=(1, 2), default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=256,
    )
    outputs = llm.chat(
        [[
            {
                "role": "user",
                "content": (
                    "Using the numbers [1, 1, 1, 1], create an equation that "
                    "equals 4. Use each number exactly once. Only use +, -, *, "
                    "/ and parentheses. Finally return "
                    "<answer> equation </answer>."
                ),
            }
        ]],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=32),
        chat_template_kwargs={"enable_thinking": False},
    )
    text = outputs[0].outputs[0].text.strip()
    if not text:
        raise RuntimeError("vLLM returned an empty generation")
    print(text)
    print(f"OK: vLLM chat completed with tensor_parallel_size={args.tensor_parallel_size}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
