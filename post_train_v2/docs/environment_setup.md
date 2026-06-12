# Remote Environment Setup

## Required Order

Set up the environment only after the latest repository revision containing
`post_train_v2/pyproject.toml` has been transferred to the remote machine.

The local vLLM wheel is declared by a repository-relative path, and the uv
indexes and exact package versions are defined in the project files.

## 1. Transfer or Update the Repository

Using Git:

```bash
cd /path/to/remote/workspace
git clone <repository-url>
cd <repository-name>
```

For an existing clone:

```bash
cd /path/to/remote/repository
git pull --ff-only
```

Confirm:

```bash
test -f post_train_v2/pyproject.toml
test -f post_train_v2/environment.md
test -f post_train_v2/constraints-verl071-vllm017-cu128.txt
```

Do not copy the old AgentFlow `.venv`.

## 2. Transfer Manual Artifacts

Download on a networked machine:

### vLLM 0.17.0 cu128

```text
https://github.com/vllm-project/vllm/releases/download/v0.17.0/vllm-0.17.0%2Bcu128-cp38-abi3-manylinux_2_35_x86_64.whl
```

### Flash Attention 2.8.3 source

```text
https://github.com/Dao-AILab/flash-attention/archive/refs/tags/v2.8.3.tar.gz
```

Upload as:

```text
post_train_v2/wheels/vllm-0.17.0+cu128-cp38-abi3-manylinux_2_35_x86_64.whl
post_train_v2/wheels/flash-attention-2.8.3.tar.gz
```

Verify:

```bash
ls -lh post_train_v2/wheels/
sha256sum post_train_v2/wheels/*
```

## 3. Leave the AgentFlow Environment

```bash
conda deactivate || true
deactivate 2>/dev/null || true
unset PYTHONPATH
hash -r
```

The CUDA compiler remains available through its absolute toolkit path:

```bash
export CUDA_HOME=/inspire/hdd/project/fdu-aidake-cfff/public/.conda/envs/llm-26
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
```

Validate:

```bash
"$CUDA_HOME/bin/nvcc" --version
test -f "$CUDA_HOME/include/cuda_runtime.h"
test -f "$CUDA_HOME/lib/libcudart.so.12"
```

## 4. Create the uv Environment

```bash
cd post_train_v2
rm -rf .venv
uv venv .venv --python 3.11 --seed
source .venv/bin/activate
```

The `rm -rf` command is appropriate only for a newly created or disposable
`post_train_v2/.venv`. Do not apply it to the AgentFlow environment.

Confirm interpreter isolation:

```bash
which python
python -V
python -c "import sys; print(sys.executable); print(sys.prefix)"
```

The executable must point into `post_train_v2/.venv`.

## 5. Resolve and Install the Runtime

`pyproject.toml` uses:

- TUNA for ordinary PyPI packages;
- the official PyTorch cu128 index for the Torch family;
- the uploaded local file for vLLM.

Run:

```bash
uv lock
uv sync --frozen
```

If TUNA has not synchronized a required package, temporarily retry with PyPI
as the default index:

```bash
uv sync --frozen \
  --default-index https://pypi.org/simple
```

Do not replace the explicit PyTorch cu128 index.

## 6. Validate the Base Runtime

```bash
uv pip check
```

Then:

```bash
python - <<'PY'
from importlib.metadata import version
import torch

packages = [
    "verl",
    "vllm",
    "transformers",
    "trl",
    "peft",
    "accelerate",
    "datasets",
    "tokenizers",
    "safetensors",
    "ray",
    "tensordict",
    "pyarrow",
    "numpy",
]

for package in packages:
    print(f"{package:15} {version(package)}")

print("torch", torch.__version__)
print("torch CUDA", torch.version.cuda)
print("CXX11 ABI", torch._C._GLIBCXX_USE_CXX11_ABI)
print("CUDA available", torch.cuda.is_available())
print("GPU count", torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    properties = torch.cuda.get_device_properties(index)
    print(index, properties.name, properties.total_memory // 1024**3, "GiB")
PY
```

Expected core versions:

```text
verl         0.7.1
vllm         0.17.0
torch        2.10.0+cu128
transformers 4.57.6
trl          0.27.0
peft         0.19.1
```

GPU validation must report two devices in the actual training allocation.
Stop if it reports zero.

## 7. Build Flash Attention

Set the build environment:

```bash
export CUDA_HOME=/inspire/hdd/project/fdu-aidake-cfff/public/.conda/envs/llm-26
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="8.0"
export MAX_JOBS=4
```

Install build tools:

```bash
uv pip install ninja packaging wheel
```

Build a reusable wheel:

```bash
mkdir -p wheels/built
uv pip wheel \
  wheels/flash-attention-2.8.3.tar.gz \
  --no-build-isolation \
  --wheel-dir wheels/built
```

Install the generated wheel:

```bash
uv pip install wheels/built/flash_attn-2.8.3-*.whl
```

Validate:

```bash
python - <<'PY'
import flash_attn
import torch

print("flash_attn", flash_attn.__version__)
print("torch", torch.__version__)
print("torch CUDA", torch.version.cuda)
print("CUDA available", torch.cuda.is_available())
PY
```

The system compiler is CUDA 12.4 while Torch uses CUDA 12.8. This is a minor
version difference within CUDA 12. The build may emit a warning. Stop if the
build reports a CUDA major-version mismatch or an undefined-symbol import
error.

## 8. Two-GPU NCCL Smoke Test

Create a temporary script:

```bash
cat > /tmp/post_train_v2_nccl_smoke.py <<'PY'
import os
import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank()
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
value = torch.tensor([rank + 1.0], device=f"cuda:{local_rank}")
dist.all_reduce(value)
print(
    f"rank={rank} local_rank={local_rank} "
    f"device={torch.cuda.get_device_name(local_rank)} value={value.item()}"
)
dist.destroy_process_group()
PY

torchrun --standalone --nproc_per_node=2 /tmp/post_train_v2_nccl_smoke.py
```

Both ranks must print `value=3.0`.

## 9. vLLM Smoke Test

Use the local Qwen3-0.6B model first:

```bash
python - <<'PY'
from vllm import LLM, SamplingParams

model_path = "../post_train/model/qwen/qwen3-0.6b"
llm = LLM(
    model=model_path,
    trust_remote_code=True,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.5,
)
outputs = llm.chat(
    messages=[[{"role": "user", "content": "Return <answer> 1+1 </answer>."}]],
    sampling_params=SamplingParams(temperature=0.0, max_tokens=32),
    chat_template_kwargs={"enable_thinking": False},
)
print(outputs[0].outputs[0].text)
PY
```

Run the Qwen3-8B/two-GPU generation topology only after the 0.6B smoke test
passes.

## 10. Lock and Record

```bash
uv pip check
uv pip freeze > environment.lock.txt
git status --short uv.lock
```

Commit `uv.lock`. Do not commit `.venv`, downloaded wheels, or
`environment.lock.txt`.
