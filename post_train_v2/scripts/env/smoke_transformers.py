from __future__ import annotations

import argparse
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Qwen3 Transformers Flash Attention training-step smoke test."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-seq-length", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.to(args.device)
    model.train()
    if model.config._attn_implementation != "flash_attention_2":
        raise RuntimeError(
            "Transformers did not activate flash_attention_2: "
            f"{model.config._attn_implementation!r}"
        )
    if next(model.parameters()).dtype != torch.bfloat16:
        raise RuntimeError("model parameters are not BF16")

    prompt = (
        "Using the numbers [1, 1, 1, 1], create an equation that equals 4. "
        "Use each number exactly once. Only use +, -, *, / and parentheses. "
        "Finally return <answer> equation </answer>."
    )
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_seq_length,
    )
    encoded = {name: tensor.to(args.device) for name, tensor in encoded.items()}
    labels = encoded["input_ids"].clone().to(args.device)
    outputs = model(**encoded, labels=labels)
    loss = outputs.loss
    if loss is None or not torch.isfinite(loss).item():
        raise RuntimeError("Transformers loss is missing or non-finite")
    loss.backward()

    finite_gradient = any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all().item()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if not finite_gradient:
        raise RuntimeError("no finite trainable gradient was produced")

    print(
        f"OK: Qwen3 flash_attention_2 BF16 forward/backward on {args.device}; "
        f"loss={loss.item():.6f}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
