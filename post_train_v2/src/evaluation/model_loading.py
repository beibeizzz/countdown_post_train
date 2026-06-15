"""Full-model and PEFT adapter loading for deterministic evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def adapter_base_model_path(adapter_config_path: str | Path) -> str | None:
    path = Path(adapter_config_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid adapter config JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"adapter config must be an object: {path}")
    value = payload.get("base_model_name_or_path")
    return value.strip() if isinstance(value, str) and value.strip() else None


def load_model_and_tokenizer(
    model_path: str | Path,
    *,
    base_model_path: str | Path | None = None,
    device_map: str | dict[str, Any] = "auto",
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = Path(model_path)
    adapter_config = model_path / "adapter_config.json"
    is_adapter = adapter_config.is_file()

    resolved_base: str | Path | None = base_model_path
    tokenizer_path: str | Path = model_path
    if is_adapter:
        resolved_base = resolved_base or adapter_base_model_path(adapter_config)
        if resolved_base is None:
            raise ValueError(
                "LoRA adapter checkpoint detected; pass --base-model-path or "
                "set base_model_name_or_path in adapter_config.json"
            )
        if not (model_path / "tokenizer_config.json").is_file():
            tokenizer_path = resolved_base

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
    )
    load_kwargs = {
        "device_map": device_map,
        "trust_remote_code": True,
        "attn_implementation": "flash_attention_2",
        "torch_dtype": torch.bfloat16,
    }
    if is_adapter:
        from peft import PeftModel

        base_model = AutoModelForCausalLM.from_pretrained(
            resolved_base,
            **load_kwargs,
        )
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **load_kwargs,
        )

    model.eval()
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model

