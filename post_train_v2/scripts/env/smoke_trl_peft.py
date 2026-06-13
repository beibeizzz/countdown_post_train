from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate TRL 0.19.1 trainer construction and PEFT LoRA round-trip."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def load_model(model_path: str, torch_module, device: str):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_module.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": device},
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if model.config._attn_implementation != "flash_attention_2":
        raise RuntimeError("model did not activate flash_attention_2")
    if next(model.parameters()).dtype != torch_module.bfloat16:
        raise RuntimeError("model parameters are not BF16")
    return model


def main() -> int:
    args = parse_args()
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoTokenizer
    from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    args.work_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = args.work_dir / "adapter"
    if adapter_dir.exists():
        shutil.rmtree(adapter_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )

    base_model = load_model(args.model_path, torch, args.device)
    lora_model = get_peft_model(base_model, lora_config)
    lora_model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    if not (adapter_dir / "adapter_config.json").is_file():
        raise RuntimeError("PEFT did not save adapter_config.json")
    if not (adapter_dir / "tokenizer_config.json").is_file():
        raise RuntimeError("Tokenizer did not save tokenizer_config.json")
    del lora_model, base_model
    torch.cuda.empty_cache()

    reload_base = load_model(args.model_path, torch, args.device)
    reloaded = PeftModel.from_pretrained(reload_base, adapter_dir)
    merged_model = reloaded.merge_and_unload()
    if merged_model.config._attn_implementation != "flash_attention_2":
        raise RuntimeError("merged PEFT model lost flash_attention_2")
    if next(merged_model.parameters()).dtype != torch.bfloat16:
        raise RuntimeError("merged PEFT model is not BF16")

    sft_dataset = Dataset.from_dict(
        {
            "text": [
                (
                    "Using [1,1,1,1], make 4. "
                    "<answer>1+1+1+1</answer>"
                ),
                (
                    "Using [2,2,2,2], make 4. "
                    "<answer>2+2+2-2</answer>"
                ),
            ]
        }
    )
    sft_config = SFTConfig(
        output_dir=str(args.work_dir / "sft"),
        max_length=64,
        dataset_text_field="text",
        per_device_train_batch_size=1,
        max_steps=1,
        bf16=True,
        save_strategy="no",
        logging_strategy="no",
        report_to="none",
    )
    sft_trainer = SFTTrainer(
        model=merged_model,
        args=sft_config,
        train_dataset=sft_dataset,
        processing_class=tokenizer,
    )
    if sft_trainer.train_dataset is None:
        raise RuntimeError("SFTTrainer did not retain the training dataset")
    sft_trainer.train()
    del sft_trainer, merged_model, reloaded, reload_base
    torch.cuda.empty_cache()

    preference_dataset = Dataset.from_dict(
        {
            "prompt": [
                "Using [1,1,1,1], make 4.",
                "Using [2,2,2,2], make 4.",
            ],
            "chosen": [
                "<answer>1+1+1+1</answer>",
                "<answer>2+2+2-2</answer>",
            ],
            "rejected": [
                "<answer>1+1+1-1</answer>",
                "<answer>2+2-2-2</answer>",
            ],
        }
    )
    dpo_config = DPOConfig(
        output_dir=str(args.work_dir / "dpo"),
        max_length=64,
        max_prompt_length=32,
        max_completion_length=32,
        per_device_train_batch_size=1,
        max_steps=1,
        bf16=True,
        save_strategy="no",
        logging_strategy="no",
        report_to="none",
    )
    dpo_base_model = load_model(args.model_path, torch, args.device)
    dpo_trainer = DPOTrainer(
        model=dpo_base_model,
        ref_model=None,
        args=dpo_config,
        train_dataset=preference_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    if dpo_trainer.train_dataset is None:
        raise RuntimeError("DPOTrainer did not retain the training dataset")
    dpo_trainer.train()

    print(
        "OK: PEFT adapter save/reload/merge and TRL 0.19.1 "
        "single-step SFT/DPO training completed"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
