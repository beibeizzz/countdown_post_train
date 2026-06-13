# Flash Attention Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the old verl 0.7.1/vLLM 0.17.0/cu129 environment with one reproducible Python 3.11, PyTorch 2.7 cu128, Flash Attention 2.7.4.post1, vLLM 0.9.1, and verl 0.6.0 environment; force Flash Attention 2 across every existing `post_train` Transformers model-loading path; and add deterministic artifact and remote GPU acceptance tests.

**Architecture:** A JSON manifest is the canonical source for package versions, artifact names, URLs, and hashes. `pyproject.toml`, constraints, runtime requirements, documentation, and validation scripts must agree with that manifest. Static tests run without GPUs or downloaded wheels; Level 1 hardware and existing-loader gates run on the remote two-A100 allocation after both official wheels are uploaded. The Level 2 verl optimizer-update gate remains deferred until the Parquet converter, custom reward adapter, and GRPO configuration exist.

**Tech Stack:** Python 3.11, uv, PyTorch 2.7 cu128, Flash Attention 2.7.4.post1, vLLM 0.9.1, verl 0.6.0, Transformers 4.53.2, TRL 0.19.1, PEFT 0.15.2, pytest.

---

## File Structure

Create:

- `post_train_v2/configs/environment/runtime-cu128.json`: canonical versions and artifact metadata.
- `post_train_v2/scripts/env/verify_artifacts.py`: SHA-256 and filename verification.
- `post_train_v2/scripts/env/check_runtime.py`: package, CUDA, ABI, GPU, and Ray checks.
- `post_train_v2/scripts/env/smoke_nccl.py`: two-rank NCCL all-reduce and peer-access report.
- `post_train_v2/scripts/env/smoke_flash_attention.py`: direct BF16 Flash Attention forward/backward.
- `post_train_v2/scripts/env/smoke_transformers.py`: Qwen3 Transformers Flash Attention forward/backward.
- `post_train_v2/scripts/env/smoke_legacy_loader.py`: real old shared-loader
  CUDA BF16 forward/backward.
- `post_train_v2/scripts/env/smoke_vllm.py`: vLLM chat smoke test for TP 1 or TP 2.
- `post_train_v2/scripts/env/smoke_teacher_dual_engine.py`: concurrent
  Qwen3-8B TP1 engines, one process per GPU.
- `post_train_v2/tests/env/test_runtime_manifest.py`: manifest and dependency-file consistency.
- `post_train_v2/tests/env/test_verify_artifacts.py`: artifact verifier unit tests.
- `post_train_v2/tests/env/test_env_scripts.py`: syntax and CLI contract tests.
- `post_train_v2/tests/env/test_teacher_dual_engine.py`: child isolation,
  concurrent orchestration, timeout, and failure-propagation tests.

Rename:

- `post_train_v2/constraints-verl071-vllm017-cu129.txt`
  to `post_train_v2/constraints-verl060-vllm091-cu128.txt`.

Modify:

- `post_train_v2/pyproject.toml`
- `post_train_v2/requirements-runtime.txt`
- `post_train_v2/wheels/README.md`
- `post_train_v2/docs/environment_setup.md`
- `post_train_v2/environment.md`
- `post_train_v2/README.md`
- `post_train_v2/migration_plan.md`
- `post_train_v2/open_questions.md`
- `post_train_v2/.gitignore`
- `post_train/scripts/sft/train_full.py`
- `post_train/scripts/eval/evaluate_model.py`
- `post_train/tests/test_evaluate_model_loader.py`
- `post_train/tests/test_train_full_model_loader.py`
- `post_train/tests/test_training_loader_call_chains.py`

## Task 1: Add The Canonical Runtime Manifest

**Files:**

- Create: `post_train_v2/configs/environment/runtime-cu128.json`
- Create: `post_train_v2/tests/env/test_runtime_manifest.py`

- [ ] **Step 1: Write the failing manifest test**

Create a test that loads the JSON with the standard library and asserts the
selected core versions and artifacts:

```python
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "configs/environment/runtime-cu128.json"


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_core_versions_are_locked() -> None:
    data = load_manifest()
    assert data["python"] == "3.11.15"
    assert data["cuda_runtime"] == "12.8"
    assert data["packages"]["torch"] == "2.7.0"
    assert data["packages"]["flash-attn"] == "2.7.4.post1"
    assert data["packages"]["vllm"] == "0.9.1"
    assert data["packages"]["verl"] == "0.6.0"


def test_only_true_abi_flash_attention_is_allowed() -> None:
    artifact = load_manifest()["artifacts"]["flash-attn"]
    assert "cxx11abiTRUE" in artifact["filename"]
    assert "cxx11abiFALSE" not in artifact["filename"]
    assert len(artifact["sha256"]) == 64


def test_verl_extras_are_forbidden() -> None:
    assert load_manifest()["forbidden_requirements"] == [
        "verl[gpu]",
        "verl[trl]",
        "verl[vllm]",
    ]
```

- [ ] **Step 2: Run the test and confirm it fails**

Run:

```bash
pytest -q post_train_v2/tests/env/test_runtime_manifest.py
```

Expected: failure because `runtime-cu128.json` does not exist.

- [ ] **Step 3: Create the complete manifest**

Use this package map:

```json
{
  "schema_version": 1,
  "python": "3.11.15",
  "cuda_runtime": "12.8",
  "platform": {
    "os": "linux",
    "arch": "x86_64",
    "glibc_min": "2.28",
    "gpu_compute_capability": "8.0"
  },
  "packages": {
    "accelerate": "1.7.0",
    "codetiming": "1.4.0",
    "datasets": "3.6.0",
    "dill": "0.3.8",
    "flash-attn": "2.7.4.post1",
    "hydra-core": "1.3.2",
    "numpy": "1.26.4",
    "opencv-python-headless": "4.11.0.86",
    "packaging": "24.2",
    "pandas": "2.2.3",
    "peft": "0.15.2",
    "pyarrow": "19.0.1",
    "pybind11": "2.13.6",
    "pylatexenc": "2.10",
    "ray": "2.48.0",
    "safetensors": "0.5.3",
    "tensorboard": "2.19.0",
    "tensordict": "0.10.0",
    "tokenizers": "0.21.2",
    "torch": "2.7.0",
    "torchaudio": "2.7.0",
    "torchdata": "0.11.0",
    "torchvision": "0.22.0",
    "transformers": "4.53.2",
    "trl": "0.19.1",
    "verl": "0.6.0",
    "vllm": "0.9.1",
    "wandb": "0.21.4"
  },
  "dev_packages": {
    "pytest": "8.3.5"
  },
  "artifacts": {
    "vllm": {
      "filename": "vllm-0.9.1-cp38-abi3-manylinux1_x86_64.whl",
      "url": "https://github.com/vllm-project/vllm/releases/download/v0.9.1/vllm-0.9.1-cp38-abi3-manylinux1_x86_64.whl",
      "sha256": "28b99e8df39c7aaeda04f7e5353b18564a1a9d1c579691945523fc4777a1a8c8"
    },
    "flash-attn": {
      "filename": "flash_attn-2.7.4.post1+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl",
      "url": "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1%2Bcu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl",
      "sha256": "22013b8c74a63fc70e69be1e10ff02e4ad8fec84a43600bdca67b434ed417113"
    }
  },
  "forbidden_requirements": [
    "verl[gpu]",
    "verl[trl]",
    "verl[vllm]"
  ]
}
```

- [ ] **Step 4: Run the manifest test**

Run:

```bash
pytest -q post_train_v2/tests/env/test_runtime_manifest.py
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add post_train_v2/configs/environment/runtime-cu128.json post_train_v2/tests/env/test_runtime_manifest.py
git commit -m "test: define cu128 runtime manifest"
```

## Task 2: Replace The Dependency Baseline

**Files:**

- Modify: `post_train_v2/pyproject.toml`
- Rename: `post_train_v2/constraints-verl071-vllm017-cu129.txt`
  to `post_train_v2/constraints-verl060-vllm091-cu128.txt`
- Modify: `post_train_v2/requirements-runtime.txt`
- Modify: `post_train_v2/tests/env/test_runtime_manifest.py`

- [ ] **Step 1: Add failing consistency tests**

Extend the test file to:

- parse `pyproject.toml` with `tomllib`;
- convert dependency strings into package/version pairs;
- assert every manifest package appears with the same exact version;
- assert the Torch source index ends in `/cu128`;
- assert vLLM and Flash Attention use the exact local artifact paths;
- reject any dependency beginning with `verl[`;
- assert constraints and runtime requirements contain no old version tokens.

The local source assertions must be:

```python
sources = project["tool"]["uv"]["sources"]
assert sources["vllm"]["path"].endswith(
    "wheels/vllm-0.9.1-cp38-abi3-manylinux1_x86_64.whl"
)
assert sources["flash-attn"]["path"].endswith(
    "wheels/flash_attn-2.7.4.post1+cu12torch2.7"
    "cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
)
```

- [ ] **Step 2: Run the tests and confirm old versions fail**

Run:

```bash
pytest -q post_train_v2/tests/env/test_runtime_manifest.py
```

Expected: failures mentioning cu129, verl 0.7.1, vLLM 0.17.0, or missing
Flash Attention source.

- [ ] **Step 3: Rewrite `pyproject.toml`**

Requirements:

- `requires-python = "==3.11.*"`;
- every direct dependency uses `==`;
- put `pytest==8.3.5` in a `dev` dependency group;
- use `ray[default,cgraph]==2.48.0`;
- use base `verl==0.6.0`, never a verl extra;
- include `flash-attn==2.7.4.post1` directly;
- rename the explicit Torch index to `pytorch-cu128`;
- set its URL to `https://download.pytorch.org/whl/cu128`;
- map Torch, TorchVision, and TorchAudio to that index;
- map vLLM and Flash Attention to their local wheel paths;
- retain `package = false`;
- keep the exact OpenCV override only if uv resolution still needs it.

- [ ] **Step 4: Rename and rewrite constraints**

The new file must contain the exact package matrix from the JSON manifest and
must not mention cu129, vLLM 0.17.0, verl 0.7.1, Torch 2.10, or Flash
Attention 2.8.3.

- [ ] **Step 5: Rewrite runtime requirements**

Keep only runtime packages, exact pins, and comments explaining:

- Torch comes from the official cu128 index;
- vLLM and Flash Attention come from repository-local official wheels;
- base verl is intentional because its extras conflict with this environment.

- [ ] **Step 6: Run the consistency tests**

Run:

```bash
pytest -q post_train_v2/tests/env/test_runtime_manifest.py
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add post_train_v2/pyproject.toml post_train_v2/requirements-runtime.txt post_train_v2/constraints-verl060-vllm091-cu128.txt post_train_v2/tests/env/test_runtime_manifest.py
git rm post_train_v2/constraints-verl071-vllm017-cu129.txt
git commit -m "build: pin verl 0.6 cu128 runtime"
```

## Task 3: Implement Artifact Integrity Verification

**Files:**

- Create: `post_train_v2/scripts/env/verify_artifacts.py`
- Create: `post_train_v2/tests/env/test_verify_artifacts.py`
- Modify: `post_train_v2/.gitignore`

- [ ] **Step 1: Write verifier unit tests**

Tests must cover:

- a matching temporary file succeeds;
- a wrong hash raises a clear error;
- a missing artifact raises `FileNotFoundError`;
- `--manifest` and `--wheels-dir` are accepted CLI options.

Keep hashing logic injectable through a small function:

```python
def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
```

- [ ] **Step 2: Confirm tests fail**

Run:

```bash
pytest -q post_train_v2/tests/env/test_verify_artifacts.py
```

Expected: import failure because `verify_artifacts.py` does not exist.

- [ ] **Step 3: Implement the verifier**

The CLI must:

1. load `runtime-cu128.json`;
2. locate every declared artifact under `wheels/`;
3. compare the exact filename;
4. calculate SHA-256;
5. print one `OK <filename> <sha256>` line per artifact;
6. return nonzero on any mismatch.

It must not download files or accept approximate filenames.

- [ ] **Step 4: Preserve wheel ignore rules**

Ensure `.gitignore` contains:

```gitignore
wheels/*
!wheels/README.md
```

Do not add exceptions for the large binary wheels.

- [ ] **Step 5: Run tests**

Run:

```bash
pytest -q post_train_v2/tests/env/test_verify_artifacts.py
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add post_train_v2/scripts/env/verify_artifacts.py post_train_v2/tests/env/test_verify_artifacts.py post_train_v2/.gitignore
git commit -m "feat: verify runtime wheel integrity"
```

## Task 4: Add Base Runtime And Distributed Smoke Scripts

**Files:**

- Create: `post_train_v2/scripts/env/check_runtime.py`
- Create: `post_train_v2/scripts/env/smoke_nccl.py`
- Create: `post_train_v2/tests/env/test_env_scripts.py`

- [ ] **Step 1: Write static script tests**

The tests must:

- compile every `scripts/env/*.py` with `py_compile`;
- verify each script supports `--help` without importing GPU libraries at
  module import time where avoidable;
- verify `check_runtime.py` contains expected versions loaded from the JSON
  manifest rather than a second hard-coded version map.
- force the P2P result to false and verify `smoke_nccl.py` emits a warning
  and exits with status zero after a successful NCCL collective.

- [ ] **Step 2: Confirm tests fail**

Run:

```bash
pytest -q post_train_v2/tests/env/test_env_scripts.py
```

Expected: failures for missing scripts.

- [ ] **Step 3: Implement `check_runtime.py`**

The script must:

- read the manifest;
- use `importlib.metadata.version` for every package;
- compare `packaging.version.Version(installed).base_version` with each
  manifest version;
- assert `sys.version_info[:3] == (3, 11, 15)`;
- separately assert Torch reports complete version `2.7.0+cu128`,
  `torch.version.cuda == "12.8"`, and CXX11 ABI `True`;
- assert exactly two CUDA devices when `--require-gpus 2` is passed;
- print GPU name, compute capability, total memory, and free memory;
- initialize local Ray only with `--check-ray`;
- assert Ray reports exactly two GPU resources;
- always shut Ray down in `finally`.

- [ ] **Step 4: Implement `smoke_nccl.py`**

The script must:

- require execution under `torchrun`;
- initialize NCCL;
- bind each rank to `LOCAL_RANK`;
- all-reduce `[rank + 1]` and assert the result equals 3 for world size 2;
- report CUDA peer-access and IPC capability on rank zero;
- emit a clear warning, rather than failing, when HAMI does not expose
  P2P/IPC;
- exit with status zero when P2P is false but the NCCL collective succeeds;
- allocate a small tensor on each local device and synchronize;
- destroy the process group in `finally`.

- [ ] **Step 5: Run static tests**

Run:

```bash
pytest -q post_train_v2/tests/env/test_env_scripts.py
```

Expected: all tests pass on the development machine without a GPU.

- [ ] **Step 6: Commit**

```bash
git add post_train_v2/scripts/env/check_runtime.py post_train_v2/scripts/env/smoke_nccl.py post_train_v2/tests/env/test_env_scripts.py
git commit -m "test: add runtime and nccl smoke checks"
```

## Task 5: Add Flash Attention And Transformers Smoke Tests

**Files:**

- Create: `post_train_v2/scripts/env/smoke_flash_attention.py`
- Create: `post_train_v2/scripts/env/smoke_transformers.py`
- Create: `post_train_v2/scripts/env/smoke_legacy_loader.py`
- Modify: `post_train_v2/tests/env/test_env_scripts.py`

- [ ] **Step 1: Add failing CLI contract tests**

Assert:

- `smoke_flash_attention.py --help` exposes `--device`;
- `smoke_transformers.py --help` exposes `--model-path` and
  `--max-seq-length`;
- `smoke_legacy_loader.py --help` exposes `--model-path` and `--device`;
- both scripts compile without syntax errors.

- [ ] **Step 2: Implement direct Flash Attention smoke**

Use `flash_attn.flash_attn_func` with BF16 tensors shaped:

```python
q = torch.randn(2, 32, 4, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
k = torch.randn_like(q, requires_grad=True)
v = torch.randn_like(q, requires_grad=True)
output = flash_attn_func(q, k, v, causal=True)
output.float().square().mean().backward()
```

Assert output is finite and all three gradients exist and are finite.

- [ ] **Step 3: Implement Transformers Qwen3 smoke**

Load with:

```python
AutoModelForCausalLM.from_pretrained(
    args.model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    trust_remote_code=True,
)
```

Assert:

```python
assert model.config._attn_implementation == "flash_attention_2"
```

Explicitly call `model.to("cuda:0")`, and move all tokenized inputs and
labels to that same device before loss computation.

Tokenize one short Countdown prompt, run a teacher-forced loss, call
`loss.backward()`, and assert at least one trainable gradient is finite. Keep
the default sequence length at 64 to limit smoke-test memory.

- [ ] **Step 4: Implement the real old shared-loader smoke**

Call `train_full.load_model_and_tokenizer` without mocks, verify Flash
Attention 2 and BF16, and complete a CUDA BF16 forward/backward step with
model, inputs, and labels on the selected device.

- [ ] **Step 5: Run static tests**

Run:

```bash
pytest -q post_train_v2/tests/env/test_env_scripts.py
```

Expected: all tests pass without executing CUDA code.

- [ ] **Step 6: Commit**

```bash
git add post_train_v2/scripts/env/smoke_flash_attention.py post_train_v2/scripts/env/smoke_transformers.py post_train_v2/scripts/env/smoke_legacy_loader.py post_train_v2/tests/env/test_env_scripts.py
git commit -m "test: validate flash attention training path"
```

## Task 6: Force Flash Attention In Existing Transformers Loaders

**Files:**

- Modify: `post_train/scripts/sft/train_full.py`
- Modify: `post_train/scripts/eval/evaluate_model.py`
- Modify: `post_train/tests/test_evaluate_model_loader.py`
- Create: `post_train/tests/test_train_full_model_loader.py`
- Create: `post_train/tests/test_training_loader_call_chains.py`

- [ ] **Step 1: Write failing shared-loader tests**

Add focused tests around
`post_train/scripts/sft/train_full.py::load_model_and_tokenizer` that mock
`AutoModelForCausalLM.from_pretrained` and assert:

- `attn_implementation="flash_attention_2"` is always passed;
- `torch_dtype=torch.bfloat16` is always passed;
- Full SFT is covered directly through this shared loader;
- the existing Full SFT behavior remains unchanged.

Use the dedicated `post_train/tests/test_train_full_model_loader.py` file for
all shared-loader tests.

- [ ] **Step 2: Add static and call-chain reuse tests**

Inspect imports and mock the shared loader at each entry point to prove that
`train_lora.py`, `train_rft.py`, `train_dpo.py`, and legacy `train_grpo.py`
continue to invoke `train_full.load_model_and_tokenizer`. Fail if any entry
point adds its own `AutoModelForCausalLM.from_pretrained` path or stops using
the shared loader.

- [ ] **Step 3: Extend evaluation loader tests**

Extend the existing
`post_train/tests/test_evaluate_model_loader.py` before changing production
code. Mock model loading and assert that both paths pass
`attn_implementation="flash_attention_2"` and
`torch_dtype=torch.bfloat16`:

- adapter evaluation when loading the base model before applying PEFT;
- merged/full-model evaluation when loading the evaluated model directly.

The tests must preserve existing assertions about adapter detection, merge
behavior, dtype, device placement, and return values.

- [ ] **Step 4: Run the focused tests and confirm failure**

Run:

```bash
pytest -q \
  post_train/tests/test_train_full_model_loader.py \
  post_train/tests/test_training_loader_call_chains.py \
  post_train/tests/test_evaluate_model_loader.py
```

Expected: assertions fail because the loading calls do not yet pass the
Flash Attention and BF16 arguments.

- [ ] **Step 5: Update the shared training loader**

Pass:

```python
attn_implementation="flash_attention_2",
torch_dtype=torch.bfloat16,
```

in the shared `AutoModelForCausalLM.from_pretrained` call in
`train_full.py`. Confirm by import and call-chain inspection that Full SFT,
LoRA, RFT, DPO, and legacy GRPO all reuse this loader; do not duplicate the
setting in each caller unless a caller bypasses the shared loader.

- [ ] **Step 6: Update both evaluation loading branches**

Pass the same exact two arguments in:

- the adapter base-model `from_pretrained` call;
- the merged/full-model `from_pretrained` call.

Do not change adapter merge semantics or introduce a fallback attention
implementation. Flash Attention 2 is mandatory in this environment.

- [ ] **Step 7: Run focused and related tests**

Run the two focused modules, substituting the actual shared-loader test path
if it differs:

```bash
pytest -q \
  post_train/tests/test_train_full_model_loader.py \
  post_train/tests/test_training_loader_call_chains.py \
  post_train/tests/test_evaluate_model_loader.py
```

Then run any existing SFT, LoRA, RFT, DPO, GRPO, and evaluation tests that
import these loaders. Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add \
  post_train/scripts/sft/train_full.py \
  post_train/scripts/eval/evaluate_model.py \
  post_train/tests/test_train_full_model_loader.py \
  post_train/tests/test_training_loader_call_chains.py \
  post_train/tests/test_evaluate_model_loader.py
git commit -m "feat: require flash attention in model loaders"
```

## Task 7: Add vLLM, Teacher, TRL, And PEFT Smoke Tests

**Files:**

- Create: `post_train_v2/scripts/env/smoke_vllm.py`
- Create: `post_train_v2/scripts/env/smoke_teacher_dual_engine.py`
- Create: `post_train_v2/scripts/env/smoke_trl_peft.py`
- Modify: `post_train_v2/tests/env/test_env_scripts.py`
- Create: `post_train_v2/tests/env/test_teacher_dual_engine.py`

- [ ] **Step 1: Add failing CLI tests**

Assert:

- vLLM script accepts `--model-path`, `--tensor-parallel-size`, and
  `--gpu-memory-utilization`;
- Teacher dual-engine script accepts `--model-path`,
  `--gpu-memory-utilization`, and `--timeout-seconds`;
- TRL/PEFT script accepts `--model-path` and `--work-dir`.

Add unit tests for the Teacher orchestrator using mocked child execution.
They must prove that:

- both child processes are started before either is awaited;
- child 0 receives only `CUDA_VISIBLE_DEVICES=0` and child 1 only
  `CUDA_VISIBLE_DEVICES=1`;
- timeout, nonzero exit, OOM text, empty output, and device-isolation
  violations produce a hard failure;
- identical physical GPU UUIDs or PCI bus IDs produce a hard failure even
  when the children report different logical CUDA indices;
- a successful pair returns both structured reports.

- [ ] **Step 2: Implement vLLM chat smoke**

Construct:

```python
LLM(
    model=args.model_path,
    trust_remote_code=True,
    tensor_parallel_size=args.tensor_parallel_size,
    gpu_memory_utilization=args.gpu_memory_utilization,
    max_model_len=256,
)
```

Call `llm.chat` with nested user messages and:

```python
chat_template_kwargs={"enable_thinking": False}
```

Use deterministic sampling with at most 32 generated tokens. Print the
generated text and fail if it is empty.

- [ ] **Step 3: Implement the Teacher dual-engine smoke**

The parent process must launch two concurrent child processes with isolated
environments:

```text
child 0: CUDA_VISIBLE_DEVICES=0, tensor_parallel_size=1
child 1: CUDA_VISIBLE_DEVICES=1, tensor_parallel_size=1
```

Before launching children, use standard-library `subprocess.run` to execute:

```bash
nvidia-smi --query-gpu=index,uuid,pci.bus_id --format=csv,noheader,nounits
```

Build an index-to-UUID/PCI mapping without adding a Python dependency.
Missing/malformed output or unmappable launch indices fail Level 1.

Each child must load local Qwen3-8B, call `LLM.chat` with
`chat_template_kwargs={"enable_thinking": False}`, and report structured
metadata containing:

- its assigned and inherited `CUDA_VISIBLE_DEVICES`;
- `torch.cuda.device_count()` and current device inside the child;
- the visible GPU name and total/free memory before and after model loading;
- the non-empty generated output.

The parent must start both children before waiting, enforce a timeout,
collect both results, and fail on OOM, nonzero exit, empty output, duplicate
physical identity, unexpected visible-device count, or evidence that a child
accessed the other GPU. The parent matches launch indices 0 and 1 against the
`nvidia-smi` mapping and requires different UUIDs and PCI bus IDs. The script
must clean up both processes on failure.

- [ ] **Step 4: Implement TRL and PEFT constructor smoke**

The script must:

- load local Qwen3-0.6B using `flash_attention_2`;
- create and attach a small LoRA configuration;
- save the adapter under `--work-dir`;
- reload and merge it;
- create a two-row in-memory SFT dataset;
- construct `SFTConfig` and `SFTTrainer` without training;
- create a two-row preference dataset;
- construct `DPOConfig` and `DPOTrainer` without training;
- use signatures verified against TRL 0.19.1, not newer documentation.

- [ ] **Step 5: Run static tests**

Run:

```bash
pytest -q post_train_v2/tests/env/test_env_scripts.py
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add post_train_v2/scripts/env/smoke_vllm.py post_train_v2/scripts/env/smoke_teacher_dual_engine.py post_train_v2/scripts/env/smoke_trl_peft.py post_train_v2/tests/env/test_env_scripts.py post_train_v2/tests/env/test_teacher_dual_engine.py
git commit -m "test: add vllm teacher trl and peft smoke checks"
```

## Task 8: Rewrite Environment Documentation

**Files:**

- Modify: `post_train_v2/wheels/README.md`
- Modify: `post_train_v2/docs/environment_setup.md`
- Modify: `post_train_v2/environment.md`
- Modify: `post_train_v2/README.md`
- Modify: `post_train_v2/migration_plan.md`
- Modify: `post_train_v2/open_questions.md`
- Modify: `post_train_v2/tests/env/test_runtime_manifest.py`

- [ ] **Step 1: Add documentation consistency tests**

Search the active environment documentation and fail if it contains:

```text
verl 0.7.1
vLLM 0.17.0
torch 2.10
cu129
Flash Attention 2.8.3
constraints-verl071-vllm017-cu129.txt
```

The rejected-alternatives section in the dated design document is exempt.

- [ ] **Step 2: Rewrite the wheel README**

Document:

- exact two filenames;
- exact official URLs;
- exact SHA-256 values;
- that only the TRUE ABI Flash Attention wheel is accepted;
- that both binaries remain ignored by Git.

- [ ] **Step 3: Rewrite the installation runbook**

Required order:

1. clone or pull the repository;
2. upload both exact wheels;
3. run `verify_artifacts.py`;
4. deactivate AgentFlow and clear `PYTHONPATH`, `CUDA_HOME`, and
   `LD_LIBRARY_PATH`;
5. run `uv python install 3.11.15`;
6. run `uv venv --python 3.11.15 --seed`;
7. run `uv lock`;
8. run `uv sync --frozen`;
9. run `uv pip check`;
10. run base runtime, Flash Attention, Transformers, NCCL, Ray, vLLM TP 1,
    vLLM TP 2, Teacher 8B dual-engine, and TRL/PEFT checks;
11. treat CUDA P2P/IPC results as HAMI diagnostics rather than hard gates;
12. run the Level 2 verl integration gate only after the Parquet converter,
    reward adapter, and GRPO configuration exist;
13. save `uv pip freeze` as an uncommitted diagnostic artifact.

Explicitly prohibit:

- copying AgentFlow `.venv`;
- exporting the CUDA 12.4 toolkit into the runtime;
- installing any verl extras;
- using the FALSE ABI wheel;
- installing Flash Attention from source.

- [ ] **Step 4: Update project and migration documentation**

State consistently:

- SFT/LoRA/RFT/DPO use Transformers/TRL with Flash Attention 2;
- existing Full SFT, LoRA, RFT, DPO, legacy GRPO, and evaluation
  `AutoModelForCausalLM` paths require Flash Attention 2;
- GRPO uses verl 0.6.0 FSDP2 and vLLM 0.9.1;
- vLLM uses its own attention backend;
- configuration keys must be checked against tag v0.6.0;
- the full verl batch smoke remains a pre-training integration gate, not a
  prerequisite for generating `uv.lock`.

- [ ] **Step 5: Resolve obsolete open questions**

Mark the environment-selection questions resolved with the accepted matrix.
Retain only questions unrelated to the environment migration.

- [ ] **Step 6: Run consistency tests**

Run:

```bash
pytest -q post_train_v2/tests/env/test_runtime_manifest.py
```

Expected: all tests pass and no active document contains the old baseline.

- [ ] **Step 7: Commit**

```bash
git add post_train_v2/wheels/README.md post_train_v2/docs/environment_setup.md post_train_v2/environment.md post_train_v2/README.md post_train_v2/migration_plan.md post_train_v2/open_questions.md post_train_v2/tests/env/test_runtime_manifest.py
git commit -m "docs: replace post-training environment runbook"
```

## Task 9: Resolve And Validate On The Remote GPU Host

**Files:**

- Create on remote and commit: `post_train_v2/uv.lock`
- Do not commit: `post_train_v2/environment.lock.txt`

- [ ] **Step 1: Verify artifacts before resolution**

Run from `post_train_v2`:

```bash
uv run --no-project python scripts/env/verify_artifacts.py \
  --manifest configs/environment/runtime-cu128.json \
  --wheels-dir wheels
```

Expected: two `OK` lines with the manifest hashes.

- [ ] **Step 2: Create the independent interpreter and environment**

```bash
deactivate 2>/dev/null || true
conda deactivate 2>/dev/null || true
unset PYTHONPATH CUDA_HOME LD_LIBRARY_PATH

uv python install 3.11.15
uv venv --python 3.11.15 --seed
source .venv/bin/activate
```

Expected: `python -V` reports Python 3.11.15 and `sys.executable` points into
`post_train_v2/.venv`.

- [ ] **Step 3: Resolve and synchronize**

```bash
uv lock
uv sync --frozen
uv pip check
```

Expected: lock succeeds and `uv pip check` reports no broken requirements.
If uv reports a real dependency conflict, update the design and manifest;
do not apply `--no-deps` or an undocumented override.

- [ ] **Step 4: Run the base checks**

```bash
python scripts/env/check_runtime.py \
  --manifest configs/environment/runtime-cu128.json \
  --require-gpus 2 \
  --check-ray

python scripts/env/smoke_flash_attention.py --device cuda:0

python scripts/env/smoke_transformers.py \
  --model-path ../post_train/model/qwen/qwen3-0.6b \
  --max-seq-length 64

python scripts/env/smoke_legacy_loader.py \
  --model-path ../post_train/model/qwen/qwen3-0.6b \
  --device cuda:0
```

Expected: exact versions, CUDA 12.8, ABI true, two A100 devices, two Ray GPU
resources, finite Flash Attention gradients, and a finite Qwen3 loss and
gradient through both the standalone Transformers path and the real old
shared-loader path.

- [ ] **Step 5: Run NCCL and topology checks**

```bash
CUDA_VISIBLE_DEVICES=0,1 \
torchrun --standalone --nproc_per_node=2 scripts/env/smoke_nccl.py
```

Expected: both ranks report the all-reduced value 3.0. Record peer-access
and IPC status. Under HAMI, unavailable P2P/IPC emits a warning and does not
fail this gate. The NCCL all-reduce itself is a Level 1 hard gate.

- [ ] **Step 6: Run vLLM checks**

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

Expected: both runs generate non-empty text with thinking disabled.

- [ ] **Step 7: Run the Teacher 8B dual-engine check**

```bash
CUDA_VISIBLE_DEVICES=0,1 \
python scripts/env/smoke_teacher_dual_engine.py \
  --model-path ../post_train/model/qwen/qwen3-8b \
  --gpu-memory-utilization 0.8 \
  --timeout-seconds 600
```

Expected: GPU 0 and GPU 1 concurrently host separate TP1 engines, both
complete `LLM.chat`, and the report contains each process's
`CUDA_VISIBLE_DEVICES`, actual visible/current device, memory observations,
and output. Any OOM, timeout, process failure, empty output, or cross-device
leakage fails the Level 1 gate.

- [ ] **Step 8: Run TRL and PEFT checks**

```bash
python scripts/env/smoke_trl_peft.py \
  --model-path ../post_train/model/qwen/qwen3-0.6b \
  --work-dir /tmp/post_train_v2_trl_peft_smoke
```

Expected: SFTTrainer and DPOTrainer construct successfully and the LoRA
adapter is saved, reloaded, and merged.

- [ ] **Step 9: Specify the deferred Level 2 verl integration gate**

After the JSONL-to-verl-Parquet converter and Countdown reward adapter are
implemented and a verl GRPO configuration exists, run one two-question,
two-rollout GRPO batch with:

```text
post_train_v2/configs/verl/grpo_countdown_smoke.yaml
post_train_v2/scripts/verl/launch_grpo.py
post_train_v2/tests/integration/test_verl_grpo_smoke.py
```

The future integration test must invoke the launch script with the named
configuration. Its executable verification command must be:

```bash
pytest -q post_train_v2/tests/integration/test_verl_grpo_smoke.py \
  --verl-config post_train_v2/configs/verl/grpo_countdown_smoke.yaml
```

The launch script will apply:

```text
actor_rollout_ref.actor.strategy=fsdp2
actor_rollout_ref.rollout.name=vllm
trainer.n_gpus_per_node=2
trainer.nnodes=1
```

Acceptance requires:

- one completed optimizer update;
- non-empty reward and response-length metrics;
- recorded Ray placement and actor/rollout GPU assignments;
- no CUDA OOM, NCCL error, or weight-synchronization failure.

These three files are future deliverables for the GRPO implementation stage,
not files created in the current environment migration. This gate is
intentionally deferred. Do not claim GRPO training readiness until the
executable command passes.

- [ ] **Step 10: Record and commit the lock**

```bash
uv pip freeze > environment.lock.txt
git add uv.lock
git commit -m "build: lock cu128 training environment"
```

Do not commit `.venv`, wheels, or `environment.lock.txt`.

- [ ] **Step 11: Return to the repository root**

```bash
cd ..
```

Confirm the current directory is the repository root before Task 10.

## Task 10: Final Cross-Repository Review

**Files:**

- Review changed files under `post_train_v2/`
- Review changed files under `post_train/`

- [ ] **Step 1: Run static tests**

```bash
pytest -q post_train_v2/tests/env
pytest -q \
  post_train/tests/test_train_full_model_loader.py \
  post_train/tests/test_training_loader_call_chains.py \
  post_train/tests/test_evaluate_model_loader.py
```

Expected: all tests pass.

- [ ] **Step 2: Scan for stale baseline references**

```bash
rg -n \
  "constraints-verl071|pytorch-cu129|vllm-0\.17\.0|verl==0\.7\.1|torch==2\.10\.0|flash-attn==2\.8\.3" \
  post_train_v2 \
  -g '!docs/superpowers/specs/2026-06-12-flash-attention-environment-design.md' \
  -g '!docs/superpowers/plans/2026-06-12-flash-attention-environment-implementation.md'
```

Expected: no matches.

- [ ] **Step 3: Scan for forbidden verl extras and source builds**

```bash
rg -n "verl\[(gpu|trl|vllm)\]|flash-attn.*no-build-isolation|flash-attention.*tar\.gz" post_train_v2
```

Expected: matches only in documentation that explicitly prohibits those
operations.

- [ ] **Step 4: Review the complete diff**

```bash
git diff --check -- post_train_v2 post_train
git diff --stat -- post_train_v2 post_train
git status --short -- post_train_v2 post_train
```

Expected: no whitespace errors, no wheel binaries staged, and only intended
environment migration files changed.

- [ ] **Step 5: Commit any review corrections**

```bash
git add post_train_v2 post_train
git commit -m "chore: finalize flash attention environment migration"
```
