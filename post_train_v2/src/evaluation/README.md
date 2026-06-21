# Evaluation

This package provides the shared Countdown evaluator used by all V2 training
stages. It validates full generated responses with exact rational arithmetic,
records format, correctness, length, and truncation diagnostics, and
aggregates stable metrics.

Transformers evaluation always applies the Qwen chat template with
`enable_thinking=False`, uses greedy decoding, and permits at most 256 new
tokens. Full checkpoints load directly. A directory containing
`adapter_config.json` is treated as a PEFT adapter and requires either an
explicit base model path or `base_model_name_or_path` in that config.

Evaluation fingerprints the full checkpoint contents before loading and
verifies the fingerprint again after generation and immediately before
publishing the manifest. LoRA fingerprints include both adapter and base
model identity. An exclusive output lock prevents concurrent evaluations
from interleaving `samples.jsonl`, `metrics.json`, and `manifest.json`.
