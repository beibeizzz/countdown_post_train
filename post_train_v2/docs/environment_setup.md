# Remote Environment Setup

This runbook creates an isolated post-training environment for the two
allocated A100 40 GB devices. Run commands from the repository root unless a
step explicitly changes directory.

## 1. Update the Repository

Clone the repository or update the existing checkout:

```bash
git clone <repository-url>
cd <repository-name>
```

or:

```bash
cd <repository-name>
git pull --ff-only
```

Confirm the environment files exist:

```bash
test -f post_train_v2/pyproject.toml
test -f post_train_v2/configs/environment/runtime-cu128.json
test -f post_train_v2/constraints-verl060-vllm091-cu128.txt
```

Do not copy or reuse the AgentFlow `.venv`.

## 2. Upload and Verify the Wheels

Download the exact official artifacts listed in
`post_train_v2/wheels/README.md`, then upload them to
`post_train_v2/wheels/` without renaming them.

Verify filenames and SHA-256 values before dependency resolution:

```bash
cd post_train_v2
python3 scripts/env/verify_artifacts.py \
  --manifest configs/environment/runtime-cu128.json \
  --wheels-dir wheels
```

The command must print two `OK` lines. Do not continue after a missing file or
hash mismatch.

## 3. Isolate the Runtime

Leave existing virtual or Conda environments and clear inherited build/runtime
paths:

```bash
deactivate 2>/dev/null || true
conda deactivate 2>/dev/null || true
unset PYTHONPATH CUDA_HOME LD_LIBRARY_PATH
hash -r
```

Do not export the CUDA 12.4 compiler toolkit into this runtime. The PyTorch and
vLLM wheels carry their selected CUDA runtime libraries; the host NVIDIA driver
provides device access.

## 4. Create the Exact Python Environment

From `post_train_v2`:

```bash
uv python install 3.11.15
uv venv --python 3.11.15 --seed
source .venv/bin/activate
python -V
python -c "import sys; print(sys.executable)"
```

Python must report `3.11.15`, and the executable must be inside
`post_train_v2/.venv`.

## 5. Resolve and Install

```bash
uv lock
uv sync --frozen
uv pip check
```

The project uses the configured TUNA mirror for ordinary packages, the
official PyTorch `cu128` index for the Torch family, and repository-local
official wheels for vLLM and Flash Attention.

Do not:

- install any `verl` extra;
- use `--no-deps` to bypass a resolver conflict;
- substitute a FALSE ABI Flash Attention wheel;
- compile or install Flash Attention from source.

If resolution fails, update the manifest and design after identifying the
actual conflict. Do not apply an undocumented override.

## 6. Run Static Tests

From the repository root:

```bash
cd ..
python -m pytest -q post_train_v2/tests/env
python -m pytest -q \
  post_train/tests/test_train_full_model_loader.py \
  post_train/tests/test_evaluate_model_loader.py \
  post_train/tests/test_flash_attention_entrypoints.py
```

## 7. Level 1 Runtime Gates

Return to `post_train_v2` and set model paths as needed:

```bash
cd post_train_v2
```

Runtime, package, GPU, ABI, topology, and Ray checks:

```bash
python scripts/env/check_runtime.py \
  --manifest configs/environment/runtime-cu128.json \
  --require-gpus 2 \
  --check-ray
```

Direct Flash Attention and Transformers training paths:

```bash
python scripts/env/smoke_flash_attention.py --device cuda:0

python scripts/env/smoke_transformers.py \
  --model-path ../post_train/model/qwen/qwen3-0.6b \
  --max-seq-length 64

python scripts/env/smoke_legacy_loader.py \
  --model-path ../post_train/model/qwen/qwen3-0.6b \
  --device cuda:0
```

NCCL:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
torchrun --standalone --nproc_per_node=2 scripts/env/smoke_nccl.py
```

The all-reduce result is a hard gate. Missing P2P/IPC exposure under HAMI is a
warning only.

vLLM TP1 and TP2:

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/env/smoke_vllm.py \
  --model-path ../post_train/model/qwen/qwen3-0.6b \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.5

CUDA_VISIBLE_DEVICES=0,1 \
python scripts/env/smoke_vllm.py \
  --model-path ../post_train/model/qwen/qwen3-0.6b \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.4
```

Two concurrent Qwen3-8B teacher engines:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
python scripts/env/smoke_teacher_dual_engine.py \
  --model-path ../post_train/model/qwen/qwen3-8b \
  --gpu-memory-utilization 0.8 \
  --timeout-seconds 600
```

TRL/PEFT constructors and adapter round-trip:

```bash
python scripts/env/smoke_trl_peft.py \
  --model-path ../post_train/model/qwen/qwen3-0.6b \
  --work-dir /tmp/post_train_v2_trl_peft_smoke
```

Full-model evaluation loader:

```bash
python scripts/env/smoke_eval_loader.py \
  --model-path ../post_train/model/qwen/qwen3-0.6b
```

LoRA evaluation loader, using the adapter produced above:

```bash
python scripts/env/smoke_eval_loader.py \
  --model-path /tmp/post_train_v2_trl_peft_smoke/adapter \
  --base-model-path ../post_train/model/qwen/qwen3-0.6b
```

Level 1 passes only when every hard gate above succeeds on the remote GPU
allocation.

## 8. Level 2 Deferred GRPO Gate

The verl optimizer-update gate is intentionally deferred until these files
exist:

- JSONL-to-verl-Parquet converter;
- custom Countdown reward adapter;
- two-GPU GRPO smoke configuration and launcher.

Level 2 must complete one two-question, two-rollout GRPO optimizer update with
FSDP2 and vLLM, emit reward/length metrics, and show correct Ray placement.
Environment installation alone does not establish GRPO training readiness.

## 9. Record the Resolved Environment

```bash
uv pip freeze > environment.lock.txt
git status --short uv.lock
```

Commit `uv.lock` after successful remote resolution. Do not commit `.venv`,
wheel binaries, or `environment.lock.txt`.
