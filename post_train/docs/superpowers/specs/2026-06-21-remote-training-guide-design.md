# Legacy Post-Train Remote Training Guide Design

## Goal

Add an operator-focused remote training guide for the existing `post_train/`
project without changing training behavior. The guide must validate an already
existing uv environment, establish the actual single-GPU hardware contract,
and document the complete data, training, evaluation, monitoring, recovery,
and artifact workflow.

## Scope

Create:

- `post_train/docs/remote_training_guide.md`

Update only documentation navigation and stage-specific instructions in:

- `post_train/README.md`
- `post_train/configs/README.md`
- `post_train/scripts/data/README.md`
- `post_train/scripts/sft/README.md`
- `post_train/scripts/dpo/README.md`
- `post_train/scripts/grpo/README.md`
- `post_train/scripts/eval/README.md`

Do not modify Python code, YAML defaults, model weights, datasets, or output
artifacts as part of this documentation change.

## Hardware Contract

The existing `post_train/` implementation is single-GPU oriented:

- `VLLMGenerator` defaults to `tensor_parallel_size=1`.
- Teacher, RFT, and DPO generation create one vLLM engine.
- Full SFT, LoRA SFT, RFT, and DPO use ordinary Transformers/TRL Trainer
  entrypoints and are documented with plain `python` launch commands.
- Legacy GRPO loads the trainable model and one vLLM rollout engine without
  DDP, FSDP, DeepSpeed, or a two-GPU worker topology.

One CUDA GPU is therefore the required execution contract. The recommended
remote default is:

```bash
export CUDA_VISIBLE_DEVICES=0
```

A second GPU is tested independently when available, but NCCL, P2P, and
two-rank training are not required acceptance gates for this project. The
guide must explicitly distinguish this contract from `post_train_v2/`.

## Environment Validation Model

The guide assumes that a uv environment already exists. It must not begin by
creating an environment or installing a fixed dependency set.

Validation proceeds in levels:

1. Identity and package inventory
   - Resolve the active Python executable and version.
   - Record `uv --version`, `uv pip list`, and selected package versions.
   - Run `uv pip check` and interpret conflicts before training.
2. CUDA and PyTorch
   - Record driver output from `nvidia-smi`.
   - Verify `torch.version.cuda`, device visibility, BF16 support, and a real
     CUDA allocation on each intended GPU.
3. Framework imports
   - Import Transformers, TRL, PEFT, Accelerate, Datasets, vLLM, Flash
     Attention, pandas, pyarrow, PyYAML, tokenizers, and safetensors.
4. Model runtime gates
   - Load local Qwen3-0.6B with BF16 and
     `attn_implementation="flash_attention_2"`.
   - Run a one-prompt Qwen3-8B vLLM chat smoke with thinking disabled.
   - Construct or run a minimal Trainer/TRL/PEFT smoke without writing to
     production output directories.

Only after a failed check may the guide discuss targeted installation or
version repair. Remediation must be diagnostic rather than a blanket uv
environment rebuild.

## Dependency Contract

The guide will check functionality rather than enforce the V2 pinned version
set. Required packages are derived from current imports:

- `torch`
- `transformers`
- `trl`
- `peft`
- `accelerate`
- `datasets`
- `vllm`
- `flash-attn`
- `pandas`
- `pyarrow`
- `PyYAML`
- `tokenizers`
- `safetensors`

`wandb` and `pytest` are optional for training, but required for monitoring
and local test execution respectively.

The guide must warn that a package import alone is insufficient. Flash
Attention and vLLM require separate model runtime smoke tests.

## Training Workflow

The detailed guide follows the actual script dependency order:

```text
raw_train.parquet + raw_test.json
  -> build_source.py
  -> build_teacher_pool.py
  -> build_sft_splits.py
  -> Full SFT / LoRA SFT
  -> RFT rollout and RFT training
  -> DPO pair generation and DPO training
  -> legacy GRPO
  -> evaluate_model.py
```

Each stage documents:

- prerequisite files and model paths;
- the production command;
- a safe smoke command using `--limit` or `--max-steps` when supported;
- expected output files and directories;
- manifest or metrics inspection commands;
- common failure modes;
- whether vLLM, Trainer, TRL, PEFT, or Flash Attention is involved.

## Evaluation Coverage

The guide documents evaluation for:

- base Qwen3-0.6B;
- Full SFT final model;
- LoRA adapter with explicit base model fallback;
- RFT final model;
- DPO final model;
- GRPO final model.

Evaluation commands must use isolated output directories. LoRA evaluation
must explain `--base-model-path` when the adapter metadata is insufficient.

## W&B and Recovery

W&B remains disabled by default. The guide documents existing configuration
keys and verifies login only when `report_to: wandb` is enabled.

Recovery instructions cover:

- identifying existing Trainer checkpoints;
- preserving production outputs during smoke tests;
- Teacher output locking and stale-lock recovery;
- terminating stale vLLM processes before switching stages;
- checking manifests, `metrics.jsonl`, fixed evaluation samples, and final
  model directories before declaring a stage complete.

## README Structure

`post_train/README.md` remains the concise primary workflow and gains:

- the single-GPU hardware statement;
- environment and hardware gate summary;
- stage dependency overview;
- a link to the detailed remote guide;
- links to stage-specific README files.

Child README files own local details such as configuration fields, smoke
commands, stage outputs, and failure checks. They link back to the detailed
guide instead of duplicating the full environment audit.

## Verification

Documentation acceptance requires:

1. Every command references an existing script and config.
2. Model and dataset paths match current YAML defaults.
3. The documented launch mode remains single GPU unless explicitly marked as
   optional experimentation.
4. No command writes smoke artifacts into production output directories.
5. The existing `post_train/tests` suite still passes.
6. `git diff --check` reports no whitespace errors.

Remote hardware commands are documented but cannot be claimed as passed until
their output is collected on the remote machine.
