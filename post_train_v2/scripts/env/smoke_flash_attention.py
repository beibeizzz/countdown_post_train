from __future__ import annotations

import argparse
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a direct BF16 Flash Attention forward/backward smoke test."
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import torch
    from flash_attn import flash_attn_func

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    q = torch.randn(
        2,
        32,
        4,
        64,
        device=args.device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    output = flash_attn_func(q, k, v, causal=True)
    if not torch.isfinite(output).all().item():
        raise RuntimeError("Flash Attention output contains non-finite values")

    output.float().square().mean().backward()
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if tensor.grad is None or not torch.isfinite(tensor.grad).all().item():
            raise RuntimeError(f"Flash Attention produced invalid {name} gradients")

    print(
        f"OK: flash_attn BF16 forward/backward on {args.device}; "
        f"shape={tuple(output.shape)}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
