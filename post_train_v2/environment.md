# Runtime Environment Baseline

## 1. Corrected Current Environment

The implementation target is the following existing Python 3.11 environment:

| Package | Current version |
| --- | --- |
| Python | 3.11.15 |
| verl | 0.7.1 |
| vLLM | 0.20.1 |
| Transformers | 5.9.0 |
| TRL | not installed |
| PEFT | 0.19.1 |
| Accelerate | 1.13.0 |
| PyTorch | 2.11.0 |
| TorchVision | 0.26.0 |
| TorchAudio | 2.11.0 |
| Datasets | 4.8.5 |
| Tokenizers | 0.22.2 |
| Safetensors | 0.7.0 |
| Flash Attention | not installed |
| DeepSpeed | not installed |

The previous Python 3.12 / PyTorch 2.8 / vLLM 0.10.2 baseline was based on
incorrect environment information and is no longer applicable.

Additional inspection results:

- `torch.version.cuda` is `13.0`.
- `torch.cuda.device_count()` returned `0`.
- `uv pip check` reported 10 incompatibilities in the AgentFlow environment.

The zero GPU count means this process was not running with visible GPU
devices. It does not prove that CUDA 13.0 or NCCL works on the eventual
two-GPU training allocation.

The dependency check confirms that the AgentFlow environment is already
internally inconsistent. Examples include AgentFlowKit requiring vLLM 0.8.5
while vLLM 0.20.1 is installed, and xFormers requiring PyTorch 2.6.0 while
PyTorch 2.11.0 is installed. This strengthens the requirement to avoid
in-place modification.

## 2. Compatibility Finding

The current environment cannot yet be accepted as a supported verl GRPO
runtime.

verl 0.7.1 declares a vLLM optional dependency range ending at 0.12.0 in its
package metadata. More importantly, the verl 0.7.1 release notes and stable
Dockerfile use:

- vLLM 0.17.0
- PyTorch 2.10.0
- TorchVision 0.25.0
- TorchAudio 2.10.0
- Transformers below 5 through vLLM 0.17.0
- Flash Attention 2.8.3

The installed vLLM 0.20.1 requires:

- PyTorch 2.11.0
- TorchVision 0.26.0
- TorchAudio 2.11.0

The installed Torch family is internally consistent with vLLM 0.20.1, but
vLLM 0.20.1 is newer than the rollout engine tested and documented by verl
0.7.1. verl imports succeeding does not prove compatibility: rollout worker
construction, weight refit, CUDA IPC, sleep/wake behavior, and engine
configuration are API-sensitive.

V2 will therefore not target the unverified combination:

```text
verl 0.7.1 + vLLM 0.20.1 + PyTorch 2.11.0
```

## 3. Recommended Reproducible Baseline

The recommended V2 baseline follows the official verl 0.7.1 stable vLLM
image:

| Package | Target version | Action |
| --- | --- | --- |
| Python | 3.11.15 | Keep; verl and TRL support Python 3.11 |
| verl | 0.7.1 | Keep |
| vLLM | 0.17.0 cu128 wheel | Downgrade from 0.20.1 |
| PyTorch | 2.10.0 cu128 | Downgrade from 2.11.0 |
| TorchVision | 0.25.0 cu128 | Downgrade from 0.26.0 |
| TorchAudio | 2.10.0 cu128 | Downgrade from 2.11.0 |
| Transformers | 4.57.6 | Downgrade from 5.9.0 |
| TRL | 0.27.0 | Install for DPO |
| PEFT | 0.19.1 | Keep |
| Accelerate | 1.13.0 | Keep |
| Datasets | 4.8.5 | Keep |
| Tokenizers | 0.22.2 | Keep |
| Safetensors | 0.7.0 | Keep |
| Flash Attention | 2.8.3 | Install |
| DeepSpeed | not installed | Keep absent |

Supporting verl packages:

| Package | Target |
| --- | --- |
| Ray | 2.48.0 |
| TensorDict | 0.10.0 |
| PyArrow | 19.0.1 |
| NumPy | 1.26.4 |
| Hydra Core | 1.3.2 |
| TorchData | 0.11.0 |

TRL 0.27.0 is selected because the verl 0.7.1 stable vLLM Dockerfile installs
that version explicitly. Its declared requirements include Python 3.11,
Transformers 4.56.2 or newer, Accelerate 1.4.0 or newer, and Datasets 3.0.0
or newer. The proposed versions satisfy those constraints.

Transformers 4.57.6 is selected because:

- vLLM 0.17.0 requires Transformers 4.56.0 or newer and below 5;
- TRL 0.27.0 requires Transformers 4.56.2 or newer;
- Tokenizers 0.22.2 satisfies Transformers 4.57.6's 0.22 through 0.23 range.

## 4. Environment Isolation Requirement

Do not downgrade the existing AgentFlow environment in place until it is
confirmed that AgentFlow itself does not require vLLM 0.20.1, Transformers
5.9.0, or PyTorch 2.11.0.

The recommended approach is a separate project environment:

```text
AgentFlow/.venv                 existing application environment
post_train_v2/.venv            post-training environment
```

Model checkpoints and datasets are filesystem artifacts and can be shared
between the two environments.

If cluster policy prevents a second virtual environment, the in-place
downgrade requires explicit approval after `uv pip freeze` and AgentFlow
dependency inspection.

Given the reported `uv pip check` failures, in-place modification is no
longer recommended even if it is technically permitted.

## 5. Libraries Not Required

The confirmed architecture does not require:

- DeepSpeed: Trainer stages use two-GPU DDP.
- Apex: not needed for the FSDP GRPO path.
- Megatron-LM or mbridge: GRPO uses FSDP/FSDP2.
- SGLang: rollout uses vLLM.
- A learned reward model: Countdown uses the exact rule validator.
- `math-verify`: the project retains its Fraction/AST solver validation.

Flash Attention 2.8.3 is recommended by the official stable image. It will be
compiled and installed separately after the selected PyTorch/CUDA wheel
combination is installed.

Compilation requires more than a CUDA-enabled PyTorch wheel:

- `nvcc` must exist;
- `CUDA_HOME` must point to a compatible CUDA toolkit;
- the GPU architecture must be supported;
- `ninja` and compiler build tools must be available.

Flash Attention 2.8.3 requires CUDA 12.0 or newer and supports Ampere, Ada,
and Hopper for FlashAttention-2. The exact GPU model remains unknown because
the current process saw zero GPUs.

## 6. Version and Mirror Configuration

The environment is defined by:

- `pyproject.toml`
- `constraints-verl071-vllm017-cu128.txt`
- `requirements-runtime.txt`

`pyproject.toml` uses:

- TUNA as the default mirror for ordinary PyPI packages;
- the official PyTorch cu128 index only for Torch, TorchVision, and
  TorchAudio.

CUDA packages should not be resolved from a general-purpose PyPI mirror.
The PyTorch index is explicit so unrelated dependencies cannot be selected
from it.

### Why cu128

The node has:

- NVIDIA driver 580.105.08, supporting CUDA 13.0 runtime compatibility;
- two A100 SXM4 devices exposed as approximately 40 GB each by HAMI;
- system `nvcc` 12.4;
- glibc 2.35.

The driver can run CUDA 12.8 binaries. Using cu128 keeps the runtime in the
same CUDA 12 major series as the local 12.4 compiler. Building a CUDA
extension with nvcc 12.4 against a cu128 Torch build can still produce a
minor-version warning and must be smoke-tested, but it avoids the unsupported
CUDA 12 compiler versus CUDA 13 Torch major-version mismatch.

A100 is an Ampere GPU and is supported by FlashAttention-2.

## 7. Remaining Environment Inspection

Run this in the current remote environment:

```bash
uv run --no-sync python - <<'PY'
import torch
from importlib.metadata import PackageNotFoundError, version

print("torch:", torch.__version__)
print("torch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    print(f"GPU {index}:", torch.cuda.get_device_name(index))

packages = [
    "numpy",
    "pandas",
    "pyarrow",
    "ray",
    "tensordict",
    "torchdata",
    "hydra-core",
    "wandb",
    "tensorboard",
    "codetiming",
    "dill",
    "packaging",
    "pybind11",
    "pylatexenc",
    "huggingface-hub",
    "triton",
]

for package in packages:
    try:
        print(f"{package:18} {version(package)}")
    except PackageNotFoundError:
        print(f"{package:18} NOT INSTALLED")
PY

nvidia-smi
nvcc --version
ldd --version | head -n 1
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-<unset>}"
uname -m
gcc --version | head -n 1
g++ --version | head -n 1
ninja --version
uv pip freeze > post_train_v2-environment-before.txt
uv pip check
```

Already confirmed:

- driver: 580.105.08;
- driver CUDA capability: 13.0;
- GPU: two A100 SXM4 allocations, about 40 GB each;
- system compiler toolkit: CUDA 12.4;
- glibc: 2.35;
- current AgentFlow environment is metadata-inconsistent.

Still needed before building Flash Attention:

- `torch._C._GLIBCXX_USE_CXX11_ABI` from the new Torch 2.10 environment;
- confirmation that the new environment sees both GPUs.

Confirmed build-host details:

- uv 0.11.16;
- architecture: x86_64;
- GCC/G++ 11.4.0;
- Ninja 1.13.0;
- about 3.2 TB free on the project filesystem;
- `CUDA_VISIBLE_DEVICES` is unset;
- nvcc is provided by the shared `llm-26` conda environment.
- the CUDA 12.4 toolkit prefix contains `include/cuda_runtime.h`;
- the toolkit contains `lib/libcudart.so.12`.

An unset `CUDA_VISIBLE_DEVICES` normally leaves all devices available, but
the earlier Python process reported zero visible GPUs. Visibility must be
checked again from the new environment inside the actual training job.

## 8. Clean uv Environment Installation

Run from the repository root without activating AgentFlow's `.venv`.

### Step 1: create the isolated environment

```bash
cd post_train_v2
uv venv .venv --python 3.11 --seed
source .venv/bin/activate
```

### Step 2: download and upload the vLLM wheel

Download this file on a networked machine:

```text
https://github.com/vllm-project/vllm/releases/download/v0.17.0/vllm-0.17.0%2Bcu128-cp38-abi3-manylinux_2_35_x86_64.whl
```

Upload it to:

```text
post_train_v2/wheels/vllm-0.17.0+cu128-cp38-abi3-manylinux_2_35_x86_64.whl
```

`pyproject.toml` declares this local wheel as the vLLM package source.

### Step 3: synchronize the mirrored Python and cu128 Torch stack

```bash
uv sync
```

Ordinary packages are downloaded from TUNA. Torch packages are downloaded
from the official PyTorch cu128 index configured in `pyproject.toml`.

uv resolves the local vLLM wheel's dependencies. The explicit OpenCV override
keeps NumPy 1.26 for verl's declared constraint. The wheel requires glibc
2.35, which is available.

### Step 4: build and install Flash Attention separately

An official Flash Attention 2.8.3 wheel should be used only when every wheel
tag matches:

- Python CPython 3.11;
- Linux x86_64;
- PyTorch 2.10;
- CUDA 12;
- the same C++11 ABI value as the installed Torch.

The official 2.8.3 release page is:

```text
https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.3
```

No official 2.8.3 release wheel has been confirmed for PyTorch 2.10. The
published wheel names must be inspected before download; do not substitute a
Torch 2.8 wheel. A wheel with a mismatched Torch version may import
unsuccessfully or fail at runtime.

If no exact Torch 2.10 wheel is present, download the source archive instead:

```text
https://github.com/Dao-AILab/flash-attention/archive/refs/tags/v2.8.3.tar.gz
```

Upload it to `post_train_v2/wheels/`, then build a reusable wheel on the
remote node:

```bash
export CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(which nvcc)")")")"
export TORCH_CUDA_ARCH_LIST="8.0"
export MAX_JOBS=4

echo "$CUDA_HOME"
test -f "$CUDA_HOME/include/cuda_runtime.h"
find "$CUDA_HOME" -name 'libcudart.so*' -print -quit

uv pip install ninja packaging wheel
uv pip wheel \
  wheels/flash-attention-2.8.3.tar.gz \
  --no-build-isolation \
  --wheel-dir wheels/built
uv pip install wheels/built/flash_attn-2.8.3-*.whl
```

This produces the manually installable wheel requested for the project. It
can be archived and reused only on machines with the same Python, PyTorch,
CUDA major/minor compatibility, platform, and C++ ABI.

On the inspected host, this should resolve from:

```text
/inspire/hdd/project/fdu-aidake-cfff/public/.conda/envs/llm-26/bin/nvcc
```

`CUDA_HOME` must be the toolkit prefix, not its `bin` directory. If the
header and `libcudart` checks fail, the conda package is not a complete CUDA
development toolkit and a complete CUDA 12.x toolkit module must be loaded
before compiling Flash Attention.

The inspected toolkit passed both checks, so this prefix is the selected
Flash Attention build toolkit:

```text
/inspire/hdd/project/fdu-aidake-cfff/public/.conda/envs/llm-26
```

### Step 5: validate and lock

```bash
uv pip check
uv pip freeze > environment.lock.txt
```

`uv pip check` may report the intentionally overridden vLLM/OpenCV lower
bound. The project pins OpenCV 4.11 with NumPy 1.26 for a text-only workload.
No other incompatibility is acceptable.

## 9. Mandatory Compatibility Gates

Before core implementation begins:

1. Import every pinned package.
2. Start a two-rank NCCL `torchrun` smoke job.
3. Load local Qwen3-0.6B with Transformers 4.57.6.
4. Apply the Qwen chat template with `enable_thinking=false`.
5. Load local Qwen3-8B with vLLM 0.17.0 and run one chat batch.
6. Instantiate TRL 0.27.0 `DPOConfig` and `DPOTrainer`.
7. Save, reload, and merge a PEFT 0.19.1 LoRA adapter.
8. Initialize Ray with exactly two GPUs.
9. Run a minimal verl 0.7.1 FSDP plus vLLM GRPO batch.
10. Save and reload a verl actor checkpoint as a Hugging Face model.

## 10. Implementation Policy

After the compatibility gates pass, V2 code will target:

- Python 3.11.15
- verl 0.7.1
- vLLM 0.17.0
- PyTorch 2.10.0
- Transformers 4.57.6
- TRL 0.27.0
- PEFT 0.19.1
- Accelerate 1.13.0
- Datasets 4.8.5

The implementation must not include broad signature-based compatibility
fallbacks for vLLM 0.20 or Transformers 5. Those are different runtime
targets and require a separately reviewed migration.

## 11. Upstream References

- verl 0.7.1 release:
  https://github.com/verl-project/verl/releases/tag/v0.7.1
- verl 0.7.1 dependency declarations:
  https://github.com/verl-project/verl/blob/v0.7.1/setup.py
- verl 0.7.1 stable vLLM Dockerfile:
  https://github.com/verl-project/verl/blob/v0.7.1/docker/Dockerfile.stable.vllm
- vLLM 0.17.0 CUDA requirements:
  https://github.com/vllm-project/vllm/blob/v0.17.0/requirements/cuda.txt
- vLLM 0.17.0 common requirements:
  https://github.com/vllm-project/vllm/blob/v0.17.0/requirements/common.txt
- vLLM 0.20.1 CUDA requirements:
  https://github.com/vllm-project/vllm/blob/v0.20.1/requirements/cuda.txt
- TRL 0.27.0 dependency declarations:
  https://github.com/huggingface/trl/blob/v0.27.0/pyproject.toml
- Transformers 4.57.6 dependency declarations:
  https://github.com/huggingface/transformers/blob/v4.57.6/setup.py
