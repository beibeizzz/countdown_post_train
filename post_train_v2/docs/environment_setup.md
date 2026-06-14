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

The project limits uv resolution to Linux x86_64 through
`tool.uv.environments`. This matches the CUDA wheel platform and prevents uv
from solving unrelated macOS or Windows dependency branches.

```bash
uv lock
uv sync --frozen
uv pip check --python .venv/bin/python
uv run --frozen python -V
uv run --frozen python -c "import sys; print(sys.executable)"
```

The project uses the configured TUNA mirror for ordinary packages, the
official PyTorch `cu128` index for the Torch family, and repository-local
official wheels for vLLM and Flash Attention.

The final Python commands must report Python 3.11.15 and the
`post_train_v2/.venv/bin/python` executable. Do not accept a `uv pip check`
result that names an inherited Conda environment.

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

The teacher smoke test maps each isolated child back to a physical GPU using
the CUDA Driver API (`libcuda.so.1`) plus `nvidia-smi`. It does not query UUIDs
from `libcudart`.

After the dual-engine gate passes, run the V2 coordinator smoke and
deterministic interruption/resume gate from the repository root with the same
active V2 virtual environment. These are Level 1 gates and must pass before
the remaining TRL/PEFT and evaluation-loader gates below.

```bash
cd ..
```

Cleanup and execution are separate commands:

```bash
rm -rf -- /tmp/post_train_v2_teacher_smoke
```

```bash
unset CUDA_VISIBLE_DEVICES
set -o pipefail
python post_train_v2/scripts/generation/build_teacher_pool.py \
  --config post_train_v2/configs/generation/teacher_rollout_2gpu_smoke.yaml \
  2>&1 | tee /tmp/post_train_v2_teacher_smoke.log
```

Run the copy-runnable validator in
`post_train_v2/scripts/generation/README.md`. Acceptance requires exactly
eight accepted rows, `completed: true`, `topology: dual_tp1`, coherent
accepted/rejected/processed counts, matching source/output/contract hashes,
an exact duplicate-free source prefix, and no remaining output lock or
transaction journal.

Run the copy-runnable runtime-metadata and orphan checker in the generation
README after both coordinator procedures. Acceptance requires child-reported
worker 0/1 runtime lines with distinct positive PIDs, visible devices `0`/`1`,
and the expected distinct `gpu0`/`gpu1` cache roots. Every smoke batch must
show each worker's result count equal to its shard size, a positive nonempty
count, and a finite nonnegative latency. Each run must show final
`exitcodes=(0, 0)` with its runtime PID pair, and `ps -p PID` must find no
recorded child after shutdown. Preserve both smoke logs with the remote
acceptance record.

The manifest/output validator cannot attribute rows to workers; the
coordinator log is the acceptance source for worker execution. Empty
responses are valid rejected production rows, but the smoke gates require at
least one nonempty response from each worker to prove both generation paths.

Follow the deterministic resume-smoke procedure in the generation README. It
creates an untracked `/tmp` config with batch size two, interrupts immediately
after the first committed batch, resumes without cleanup, requires
`resume processed` to be greater than zero, and runs the same validator
against the completed resume output. Its appended log must contain two valid
runtime/shutdown records, one for the interrupted run and one for the resumed
run, and all four recorded child processes must be gone.

A transaction journal left by an abnormal interruption is recovered
automatically on restart; never delete the journal manually. Use
`--recover-stale-lock` only after inspecting the lock with `cat`, checking its
PID with `ps`, and confirming that it belongs to this host and the PID is
dead. Detailed commands and the legacy-adoption boundary are in the generation
README.

Return to `post_train_v2`:

```bash
cd post_train_v2
```

TRL/PEFT constructors and adapter round-trip:

```bash
python scripts/env/smoke_trl_peft.py \
  --model-path ../post_train/model/qwen/qwen3-0.6b \
  --work-dir /tmp/post_train_v2_trl_peft_smoke
```

This smoke test performs one real SFT optimizer step and one real DPO
optimizer step. The DPO phase reloads a clean base model before TRL applies
its LoRA configuration.

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

Level 1 passes only when the dual-engine gate, coordinator smoke, deterministic
resume smoke, remaining TRL/PEFT and loader checks, and every other hard gate
above succeed on the remote GPU allocation.

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
