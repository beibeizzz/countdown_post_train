"""Shared model and tokenizer loading for supervised V2 stages."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any


def load_causal_lm_and_tokenizer(
    model_path: str | Path,
    *,
    gradient_checkpointing: bool,
) -> tuple[Any, Any]:
    transformers = import_module("transformers")
    torch = import_module("torch")
    model_path = str(model_path)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    if getattr(tokenizer, "pad_token_id", None) is None:
        if getattr(tokenizer, "eos_token", None) is None:
            raise ValueError("tokenizer must define eos_token when pad_token is absent")
        tokenizer.pad_token = tokenizer.eos_token

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )
    if gradient_checkpointing and hasattr(model, "config"):
        model.config.use_cache = False
    return model, tokenizer
