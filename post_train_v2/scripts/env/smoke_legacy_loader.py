from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from post_train.scripts.sft.train_full import load_model_and_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a BF16 Flash Attention 2 forward/backward through the legacy model loader."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-seq-length", type=int, default=64)
    return parser.parse_args()


def run_smoke(model_path: Path, device: str, max_seq_length: int) -> None:
    if max_seq_length < 2:
        raise ValueError("max_seq_length must be at least 2")

    import torch

    model = None
    tokenizer = None
    batch = None
    labels = None
    outputs = None
    loss = None
    try:
        model, tokenizer = load_model_and_tokenizer(
            model_path,
            gradient_checkpointing=False,
        )
        model = model.to(device)
        model.train()

        floating_parameter = next(
            (parameter for parameter in model.parameters() if parameter.is_floating_point()),
            None,
        )
        if floating_parameter is None:
            raise RuntimeError("Legacy loader model has no floating-point parameters")
        if floating_parameter.dtype != torch.bfloat16:
            raise RuntimeError(
                "Legacy loader did not load BF16 parameters: "
                f"{floating_parameter.dtype}"
            )

        attention_implementation = model.config._attn_implementation
        if attention_implementation != "flash_attention_2":
            raise RuntimeError(
                "Legacy loader did not activate flash_attention_2: "
                f"{attention_implementation!r}"
            )

        batch = tokenizer(
            "Use the numbers 1, 1, 1, 1 to make 4. <answer>1+1+1+1</answer>",
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_length,
        )
        batch = {name: tensor.to(device) for name, tensor in batch.items()}
        labels = batch["input_ids"].clone()

        model.zero_grad(set_to_none=True)
        outputs = model(**batch, labels=labels)
        loss = outputs.loss
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"Legacy loader produced non-finite loss: {loss.item()}")
        loss.backward()

        finite_gradients = [
            torch.isfinite(parameter.grad).all().item()
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        if not finite_gradients:
            raise RuntimeError("Legacy loader smoke produced no gradients")
        if not all(finite_gradients):
            raise RuntimeError("Legacy loader smoke produced non-finite gradients")

        print(
            "OK legacy_loader "
            f"attention={attention_implementation} dtype={torch.bfloat16} "
            f"device={device} loss={loss.detach().float().item():.6f}"
        )
    finally:
        if model is not None:
            model.zero_grad(set_to_none=True)
        loss = None
        outputs = None
        labels = None
        batch = None
        tokenizer = None
        model = None
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    run_smoke(
        model_path=Path(args.model_path),
        device=args.device,
        max_seq_length=args.max_seq_length,
    )


if __name__ == "__main__":
    main()
