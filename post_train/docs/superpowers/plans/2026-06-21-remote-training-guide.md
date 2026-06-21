# Legacy Post-Train Remote Training Guide Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a detailed remote operator guide for the existing single-GPU `post_train/` workflow and improve README navigation without changing runtime behavior.

**Architecture:** `post_train/docs/remote_training_guide.md` owns the full environment audit, hardware gates, ordered workflow, recovery, and acceptance procedure. The root README remains a concise executable overview, while stage README files own local commands, inputs, outputs, and failure checks. All commands are derived from current scripts and YAML defaults; no V2 runtime lock or two-GPU assumption is introduced.

**Tech Stack:** Markdown, uv environment inspection, PyTorch/CUDA, Transformers, TRL, PEFT, Accelerate, Datasets, vLLM, Flash Attention 2, pandas, pyarrow, PyYAML, W&B, pytest.

---

## File Map

Create:

- `post_train/docs/remote_training_guide.md`: complete remote validation and training tutorial.

Modify:

- `post_train/README.md`: concise primary workflow and navigation.
- `post_train/configs/README.md`: configuration ownership, edit-before-run checklist, and W&B fields.
- `post_train/scripts/data/README.md`: source, Teacher, and split commands and artifacts.
- `post_train/scripts/sft/README.md`: Full SFT, LoRA, RFT rollout/training, smoke, and evaluation handoff.
- `post_train/scripts/dpo/README.md`: DPO generation/training workflow and category checks.
- `post_train/scripts/grpo/README.md`: legacy single-GPU GRPO constraints, metrics, checkpoints, and memory warnings.
- `post_train/scripts/eval/README.md`: full-model and LoRA evaluation matrix.

No Python, YAML, model, dataset, or generated-output file is modified.

## Task 1: Write the Remote Environment and Hardware Audit

**Files:**

- Create: `post_train/docs/remote_training_guide.md`

- [ ] **Step 1: Add scope and single-GPU contract**

Start the document with these explicit rules:

```markdown
# Legacy Post-Train Remote Training Guide

This guide applies to `post_train/`, not `post_train_v2/`.

- Required topology: one CUDA GPU.
- Recommended launch isolation: `export CUDA_VISIBLE_DEVICES=0`.
- A second GPU may be tested independently, but NCCL/DDP/FSDP are not
  acceptance requirements for this implementation.
- The guide validates an existing uv environment. It does not rebuild the
  environment unless a concrete check fails.
```

- [ ] **Step 2: Add active environment identity checks**

Include exact commands:

```bash
pwd
which uv
uv --version
which python
python -V
python -c "import sys; print(sys.executable); print(sys.path)"
uv pip check
```

Explain that `which python` must point inside the intended uv environment and
that `uv pip check` conflicts must be reviewed before any model smoke.

- [ ] **Step 3: Add package inventory without fixed version assumptions**

Include a Python inventory script for:

```python
packages = [
    "torch", "transformers", "trl", "peft", "accelerate", "datasets",
    "vllm", "flash-attn", "pandas", "pyarrow", "PyYAML", "tokenizers",
    "safetensors", "wandb", "pytest",
]
```

The output must distinguish `NOT INSTALLED` from an installed version. State
that `wandb` and `pytest` are optional unless monitoring or tests are used.

- [ ] **Step 4: Add driver, CUDA, BF16, and real-allocation gates**

Document:

```bash
nvidia-smi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("visible GPUs:", torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    print(index, torch.cuda.get_device_name(index), torch.cuda.get_device_capability(index))
print("BF16 supported:", torch.cuda.is_bf16_supported())
x = torch.empty(256 * 1024 * 1024, dtype=torch.float32, device="cuda")
torch.cuda.synchronize()
print("allocated bytes:", torch.cuda.memory_allocated())
del x
torch.cuda.empty_cache()
PY
```

Repeat the allocation with `CUDA_VISIBLE_DEVICES=1` only when a second GPU is
available. Explain that failure on GPU 1 does not invalidate single-GPU
execution on GPU 0, but that GPU 1 must not be used until its HAMI/accounting
problem is resolved.

- [ ] **Step 5: Add framework import and Flash Attention gates**

Include imports for all required libraries and a local Qwen3-0.6B load:

```bash
CUDA_VISIBLE_DEVICES=0 python - <<'PY'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

path = "post_train/model/qwen/qwen3-0.6b"
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
).to("cuda")
print("attention:", model.config._attn_implementation)
print("dtype:", next(model.parameters()).dtype)
print("device:", next(model.parameters()).device)
PY
```

Acceptance requires `flash_attention_2`, BF16, and CUDA.

- [ ] **Step 6: Add single-GPU Qwen3-8B vLLM smoke**

Use the project wrapper so chat templating and `enable_thinking=false` match
production:

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_CACHE_ROOT=/tmp/post-train-vllm-cache \
python - <<'PY'
from post_train.src.countdown.generation import GenerationConfig, VLLMGenerator

generator = VLLMGenerator(
    "post_train/model/qwen/qwen3-8b",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.8,
    max_model_len=512,
)
result = generator.generate(
    ["Using the numbers [1, 1, 1, 1], create an equation equal to 4."],
    GenerationConfig(max_new_tokens=64, temperature=0.0, top_p=1.0, enable_thinking=False),
)
print(result[0])
PY
```

Document cache isolation, stale process inspection with `nvidia-smi`, and the
rule that import success is not a vLLM acceptance result.

- [ ] **Step 7: Add conditional remediation policy**

State that installation commands are selected only after identifying one of:

- missing distribution;
- dependency conflict from `uv pip check`;
- CUDA/PyTorch mismatch;
- Flash Attention import/load failure;
- vLLM engine initialization failure.

Do not include a blanket `uv sync`, environment deletion, or fixed V2 lock.

- [ ] **Step 8: Verify the new environment section**

Run:

```bash
rg -n "CUDA_VISIBLE_DEVICES=0|tensor_parallel_size=1|flash_attention_2|uv pip check|NCCL" post_train/docs/remote_training_guide.md
```

Expected: all required concepts are present and NCCL is described as not
required.

- [ ] **Step 9: Commit**

```bash
git add post_train/docs/remote_training_guide.md
git commit -m "docs: add legacy remote environment gates"
```

## Task 2: Document the Complete Data and Training Workflow

**Files:**

- Modify: `post_train/docs/remote_training_guide.md`

- [ ] **Step 1: Add preflight path checks**

Document checks for:

```bash
test -f post_train/datasets/raw_train.parquet
test -f post_train/datasets/raw_test.json
test -f post_train/model/qwen/qwen3-0.6b/config.json
test -f post_train/model/qwen/qwen3-8b/config.json
```

Add disk and process inspection:

```bash
df -h .
nvidia-smi
ps -ef | grep -E 'python|vllm' | grep -v grep
```

- [ ] **Step 2: Add source-data construction**

Document production and isolated smoke commands:

```bash
python post_train/scripts/data/build_source.py \
  --config post_train/configs/data_build.yaml

mkdir -p /tmp/post_train_smoke/configs /tmp/post_train_smoke/data/processed
python - <<'PY'
from pathlib import Path
import yaml

source = Path("post_train/configs/data_build.yaml")
config = yaml.safe_load(source.read_text())
config["output_dir"] = "/tmp/post_train_smoke/data/processed"
Path("/tmp/post_train_smoke/configs/data_build.yaml").write_text(
    yaml.safe_dump(config, sort_keys=False)
)
PY
python post_train/scripts/data/build_source.py \
  --config /tmp/post_train_smoke/configs/data_build.yaml \
  --limit 100
```

List the actual outputs: `source_all.jsonl`, `train_pool.jsonl`, `val_200.jsonl`,
`val_eval_50.jsonl`, `test_with_solver_answers.jsonl`, aliases, and
`manifest.json`.

- [ ] **Step 3: Add Teacher accepted-pool workflow**

Document the production command, `CUDA_VISIBLE_DEVICES=0`, vLLM requirement,
20k stop target, accepted/rejected outputs, output lock, and
`--recover-stale-lock`. Explicitly warn that `post_train/` refuses V2-owned
Teacher state in the same output directory.

- [ ] **Step 4: Add SFT/GRPO split construction**

Document:

```bash
python post_train/scripts/data/build_sft_splits.py \
  --config post_train/configs/data_build.yaml
```

List `post_train/data/sft/sft_train_8k.jsonl`,
`post_train/data/grpo/grpo_train_4k.jsonl`, and manifests.

- [ ] **Step 5: Add Full SFT and LoRA workflows**

Document production and bounded smoke commands:

```bash
CUDA_VISIBLE_DEVICES=0 python post_train/scripts/sft/train_full.py \
  --config post_train/configs/sft_full.yaml
CUDA_VISIBLE_DEVICES=0 python post_train/scripts/sft/train_full.py \
  --config /tmp/post_train_smoke/configs/sft_full.yaml \
  --max-steps 2

CUDA_VISIBLE_DEVICES=0 python post_train/scripts/sft/train_lora.py \
  --config post_train/configs/sft_lora.yaml
```

The smoke config must be copied and rewritten so `output_dir` points below
`/tmp/post_train_smoke/outputs/`. Describe `final/`, `checkpoint-*`, and fixed
evaluation directories.

- [ ] **Step 6: Add RFT workflow**

Document:

```bash
CUDA_VISIBLE_DEVICES=0 python post_train/scripts/sft/build_rft_data.py \
  --config post_train/configs/rft.yaml
CUDA_VISIBLE_DEVICES=0 python post_train/scripts/sft/train_rft.py \
  --config post_train/configs/rft.yaml
```

Explain that the current default `base_model_path` is Qwen3-8B and must be
reviewed before running because the project goal may instead require the Full
SFT 0.6B model for RFT rollout. Do not silently change the YAML.

- [ ] **Step 7: Add DPO workflow**

Document pair generation with Qwen3-8B, target 6k, rejected categories, and
training from Full SFT final. Include `--limit` and `--max-steps 2` smoke
examples using copied configs and isolated output directories.

- [ ] **Step 8: Add legacy GRPO workflow**

Document the current architecture accurately:

- one process;
- one trainable Transformers model;
- one TP=1 vLLM rollout engine;
- no DDP/FSDP/DeepSpeed;
- `kl_coeff=0.0`;
- checkpoint every 20 steps and evaluation every 100 steps by default.

Include:

```bash
CUDA_VISIBLE_DEVICES=0 python post_train/scripts/grpo/train_grpo.py \
  --config post_train/configs/grpo.yaml \
  --max-steps 2
```

Warn that the smoke command must use a copied config with an isolated
`output_dir`, and that the trainable model and vLLM engine share the visible
GPU.

- [ ] **Step 9: Add W&B and recovery sections**

Document disabled-by-default behavior, `wandb login`, config keys, checkpoint
inspection, Teacher lock recovery, stale GPU process cleanup, and artifact
acceptance checks.

- [ ] **Step 10: Verify all documented workflow paths**

Run:

```bash
python - <<'PY'
from pathlib import Path
paths = [
    "post_train/scripts/data/build_source.py",
    "post_train/scripts/data/build_teacher_pool.py",
    "post_train/scripts/data/build_sft_splits.py",
    "post_train/scripts/sft/train_full.py",
    "post_train/scripts/sft/train_lora.py",
    "post_train/scripts/sft/build_rft_data.py",
    "post_train/scripts/sft/train_rft.py",
    "post_train/scripts/dpo/build_dpo_data.py",
    "post_train/scripts/dpo/train_dpo.py",
    "post_train/scripts/grpo/train_grpo.py",
    "post_train/scripts/eval/evaluate_model.py",
]
missing = [path for path in paths if not Path(path).is_file()]
assert not missing, missing
print("OK", len(paths), "entrypoints")
PY
```

Expected: `OK 11 entrypoints`.

- [ ] **Step 11: Commit**

```bash
git add post_train/docs/remote_training_guide.md
git commit -m "docs: document legacy training workflow"
```

## Task 3: Document the Evaluation Matrix

**Files:**

- Modify: `post_train/docs/remote_training_guide.md`
- Modify: `post_train/scripts/eval/README.md`

- [ ] **Step 1: Add full-model evaluation commands**

Document isolated commands for base, Full SFT, RFT, DPO, and GRPO, for example:

```bash
CUDA_VISIBLE_DEVICES=0 python post_train/scripts/eval/evaluate_model.py \
  --config post_train/configs/eval.yaml \
  --model-path post_train/outputs/sft/full/final \
  --output-dir post_train/data/eval/sft_full
```

Each model must use a distinct output directory.

- [ ] **Step 2: Add LoRA adapter evaluation**

Document:

```bash
CUDA_VISIBLE_DEVICES=0 python post_train/scripts/eval/evaluate_model.py \
  --config post_train/configs/eval.yaml \
  --model-path post_train/outputs/sft/lora/final \
  --base-model-path post_train/model/qwen/qwen3-0.6b \
  --output-dir post_train/data/eval/sft_lora
```

Explain adapter auto-detection and explicit base fallback.

- [ ] **Step 3: Add output interpretation**

Document `eval_samples.jsonl`, `eval_metrics.json`, accuracy, format rate,
valid-expression rate, response length, and truncation fields.

- [ ] **Step 4: Verify evaluation CLI options**

Run:

```bash
python post_train/scripts/eval/evaluate_model.py --help
```

Expected: help includes `--config`, `--model-path`, `--base-model-path`,
`--output-dir`, and `--limit`.

- [ ] **Step 5: Commit**

```bash
git add post_train/docs/remote_training_guide.md post_train/scripts/eval/README.md
git commit -m "docs: add legacy evaluation matrix"
```

## Task 4: Expand Root and Stage README Navigation

**Files:**

- Modify: `post_train/README.md`
- Modify: `post_train/configs/README.md`
- Modify: `post_train/scripts/data/README.md`
- Modify: `post_train/scripts/sft/README.md`
- Modify: `post_train/scripts/dpo/README.md`
- Modify: `post_train/scripts/grpo/README.md`

- [ ] **Step 1: Expand the root README without replacing its role**

Preserve the existing eight-stage command flow and add:

- a one-GPU hardware statement;
- a warning that this is not the V2 distributed pipeline;
- links to `docs/remote_training_guide.md` and stage README files;
- a preflight checklist for models, data, active uv environment, and GPU 0;
- an input/output dependency table.

- [ ] **Step 2: Expand the config README**

For every YAML file, document:

- owning stage;
- model/data inputs;
- output directory;
- smoke-sensitive fields;
- fields that must be reviewed before remote execution.

Call out `rft.yaml:base_model_path` explicitly without changing it.

- [ ] **Step 3: Expand the data README**

Document source build, Teacher lock/recovery, split prerequisites, outputs,
and manifest inspection commands.

- [ ] **Step 4: Expand the SFT README**

Document Full/LoRA/RFT order, Flash Attention 2, response supervision,
`--max-steps`, output isolation, checkpoint/final artifacts, and evaluation
handoff.

- [ ] **Step 5: Expand the DPO README**

Document Teacher-generated rejected candidates, pair filtering, 6k target,
Full SFT base model, `--limit`, `--max-steps`, and output files.

- [ ] **Step 6: Expand the GRPO README**

Document single-GPU architecture, shared GPU memory risk, sync/checkpoint/eval
cadence, metrics, W&B, and isolated smoke output.

- [ ] **Step 7: Verify README links and command references**

Run:

```bash
rg -n "remote_training_guide.md|single GPU|CUDA_VISIBLE_DEVICES=0" \
  post_train/README.md \
  post_train/scripts/data/README.md \
  post_train/scripts/sft/README.md \
  post_train/scripts/dpo/README.md \
  post_train/scripts/grpo/README.md
```

Expected: root and stage READMEs link to the detailed guide and accurately
state the single-GPU contract where relevant.

- [ ] **Step 8: Commit**

```bash
git add \
  post_train/README.md \
  post_train/configs/README.md \
  post_train/scripts/data/README.md \
  post_train/scripts/sft/README.md \
  post_train/scripts/dpo/README.md \
  post_train/scripts/grpo/README.md
git commit -m "docs: expand legacy training readmes"
```

## Task 5: Final Documentation Acceptance

**Files:**

- Verify all files changed in Tasks 1-4.

- [ ] **Step 1: Check documentation scope**

Run:

```bash
git diff --name-only HEAD~4..HEAD
```

Expected: only the new guide and the approved README files are listed.

- [ ] **Step 2: Run the legacy test suite**

Run:

```bash
python -m pytest -q -p no:cacheprovider post_train/tests
```

Expected: all tests pass; GPU-dependent runtime gates remain remote manual
checks.

- [ ] **Step 3: Verify every public CLI parses**

Run each entrypoint with `--help`:

```bash
python post_train/scripts/data/build_source.py --help
python post_train/scripts/data/build_teacher_pool.py --help
python post_train/scripts/data/build_sft_splits.py --help
python post_train/scripts/sft/train_full.py --help
python post_train/scripts/sft/train_lora.py --help
python post_train/scripts/sft/build_rft_data.py --help
python post_train/scripts/sft/train_rft.py --help
python post_train/scripts/dpo/build_dpo_data.py --help
python post_train/scripts/dpo/train_dpo.py --help
python post_train/scripts/grpo/train_grpo.py --help
python post_train/scripts/eval/evaluate_model.py --help
```

Expected: all commands exit 0 without loading a model.

- [ ] **Step 4: Check paths, links, placeholders, and whitespace**

Run:

```bash
rg -n "TBD|TODO|FIXME" \
  post_train/docs/remote_training_guide.md \
  post_train/README.md \
  post_train/configs/README.md \
  post_train/scripts/*/README.md
git diff --check
git status --short
```

Expected: no placeholders, no whitespace errors, and only intended
documentation changes before the final commit.

- [ ] **Step 5: Commit final documentation fixes if needed**

```bash
git add post_train/docs/remote_training_guide.md post_train/README.md post_train/configs/README.md post_train/scripts
git commit -m "docs: finalize legacy remote training tutorial"
```

Skip this commit when Step 4 finds no changes after the previous task commits.
