"""LoRA helpers for V2 supervised training."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

from post_train_v2.src.training.model_loading import load_causal_lm_and_tokenizer

QWEN_LORA_TARGET_SUFFIXES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}


def resolve_lora_target_modules(model) -> list[str]:
    found = {
        name.rsplit(".", 1)[-1]
        for name, _module in model.named_modules()
        if name.rsplit(".", 1)[-1] in QWEN_LORA_TARGET_SUFFIXES
    }
    if not found:
        raise ValueError("no Qwen LoRA target modules found")
    return sorted(found)


def apply_lora(model, config: dict[str, Any]):
    peft = import_module("peft")
    if config.get("gradient_checkpointing", True) and hasattr(
        model,
        "enable_input_require_grads",
    ):
        model.enable_input_require_grads()
    lora_config = peft.LoraConfig(
        task_type=peft.TaskType.CAUSAL_LM,
        r=int(config.get("lora_r", 16)),
        lora_alpha=int(config.get("lora_alpha", 32)),
        lora_dropout=float(config.get("lora_dropout", 0.05)),
        target_modules=resolve_lora_target_modules(model),
    )
    wrapped = peft.get_peft_model(model, lora_config)
    _mark_only_lora_trainable(wrapped)
    return wrapped


def _mark_only_lora_trainable(model) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = "lora_" in name


def merge_lora_adapter(
    *,
    base_model_path: str | Path,
    adapter_path: str | Path,
    output_dir: str | Path,
) -> Path:
    peft = import_module("peft")
    base_model, tokenizer = load_causal_lm_and_tokenizer(
        base_model_path,
        gradient_checkpointing=False,
    )
    adapter_model = peft.PeftModel.from_pretrained(base_model, str(adapter_path))
    merged_model = adapter_model.merge_and_unload()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(destination)
    tokenizer.save_pretrained(destination)
    return destination
