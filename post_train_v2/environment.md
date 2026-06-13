# Runtime Baseline

## Selected Matrix

`configs/environment/runtime-cu128.json` is the canonical source of truth.
Dependency files, documentation, and validation scripts must match it.

| Component | Version |
| --- | --- |
| Python | 3.11.15 |
| PyTorch | 2.7.0+cu128 |
| TorchVision | 0.22.0+cu128 |
| TorchAudio | 2.7.0+cu128 |
| Flash Attention | 2.7.4.post1, CXX11 ABI TRUE |
| vLLM | 0.9.1 |
| verl | 0.6.0, base package only |
| Transformers | 4.53.2 |
| TRL | 0.19.1 |
| PEFT | 0.15.2 |
| Accelerate | 1.7.0 |
| Datasets | 3.6.0 |
| Tokenizers | 0.21.2 |
| Ray | 2.48.0 |
| TensorDict | 0.10.0 |
| NumPy | 1.26.4 |
| PyArrow | 19.0.1 |

All direct and development dependencies are exact pins. The complete package
set is in the manifest and `constraints-verl060-vllm091-cu128.txt`.

## Host Contract

The target allocation is:

- Linux x86_64 with glibc 2.35;
- NVIDIA driver 580.105.08;
- two visible A100 devices, each limited to 40 GB;
- compute capability 8.0;
- one process namespace exposing both devices;
- Ray and NCCL permitted.

The driver may advertise CUDA 13.0 while the selected binary runtime is CUDA
12.8. That is valid driver backward compatibility. The CUDA 12.4 `nvcc`
available in the old Conda environment is not part of this runtime.

## Why a Separate Environment Is Required

The inspected AgentFlow environment contains incompatible application pins and
binary packages. Reusing or incrementally mutating it risks mixing unrelated
OpenAI/Anthropic, vLLM, PyTorch, xFormers, NumPy, and CUDA constraints.

`post_train_v2` therefore owns an independent uv environment. It must not
inherit AgentFlow's `.venv`, `PYTHONPATH`, `CUDA_HOME`, or `LD_LIBRARY_PATH`.

## Artifact Policy

Two manually transferred official wheels are required:

- vLLM 0.9.1 ABI3 wheel;
- Flash Attention 2.7.4.post1 CPython 3.11 wheel built for Torch 2.7,
  CUDA 12, and CXX11 ABI TRUE.

Their exact filenames, URLs, and SHA-256 hashes are stored in the manifest.
`scripts/env/verify_artifacts.py` must pass before `uv lock` or `uv sync`.

No source build is supported by this baseline. A differently named wheel,
FALSE ABI wheel, or hash mismatch is a hard failure.

## Attention Backend Boundary

All existing Transformers model-loading paths used by Full SFT, LoRA SFT,
RFT, DPO, legacy GRPO, and evaluation pass:

```python
attn_implementation="flash_attention_2"
torch_dtype=torch.bfloat16
```

This requirement is covered by unit tests. The remote forward/backward smoke
tests remain mandatory before Level 1 can pass. vLLM uses its own attention
backend and is validated separately.

## Framework Boundary

- Full SFT, LoRA SFT, RFT, and DPO remain on Transformers/TRL with two-rank
  DDP.
- Teacher, RFT, and DPO data generation use vLLM.
- GRPO migrates to verl 0.6.0 with FSDP2 and vLLM rollout.
- PPO, a critic, and a learned reward model are out of scope.
- Do not install `verl` extras; the project pins the integration dependencies
  independently.

Configuration keys for GRPO must be checked against the verl v0.6.0 source
tag. Current `main` examples or another release are not an API contract for
this project.

## Acceptance Levels

### Level 1: environment and existing paths

Required on the remote two-GPU allocation:

- exact package, Python, CUDA runtime, and CXX11 ABI checks;
- two A100 devices and two Ray GPU resources;
- direct Flash Attention BF16 forward/backward;
- Transformers Qwen3 BF16 forward/backward with Flash Attention 2;
- existing shared training loader forward/backward;
- NCCL two-rank all-reduce;
- vLLM TP1 and TP2 generation;
- two concurrent independent Qwen3-8B teacher engines;
- TRL SFT/DPO constructor and PEFT adapter round-trip;
- full and LoRA evaluation loader generation.

NCCL and vLLM TP2 are hard gates. HAMI P2P/IPC visibility is diagnostic and
may warn without failing.

### Level 2: verl optimizer update

Deferred until the Parquet converter, Countdown reward adapter, and GRPO
configuration exist. It must execute one real FSDP2 plus vLLM optimizer update
and verify metrics, placement, and checkpoint/export behavior.

Passing Level 1 does not imply Level 2 readiness.

## Installation

Use `docs/environment_setup.md` as the executable remote runbook.

Primary upstream references:

- <https://download.pytorch.org/whl/cu128/>
- <https://github.com/vllm-project/vllm/releases/tag/v0.9.1>
- <https://github.com/Dao-AILab/flash-attention/releases/tag/v2.7.4.post1>
- <https://github.com/verl-project/verl/releases/tag/v0.6.0>
