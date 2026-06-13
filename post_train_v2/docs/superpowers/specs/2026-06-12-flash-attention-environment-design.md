# Flash Attention Environment Design

## Goal

Define one reproducible Linux environment for the Countdown post-training
project that supports:

- two NVIDIA A100 GPUs exposed with approximately 40 GiB each;
- Transformers/TRL SFT, LoRA, RFT, and DPO;
- the existing `post_train` Transformers loaders used by Full SFT, LoRA,
  RFT, DPO, legacy GRPO, and evaluation;
- verl FSDP/FSDP2 GRPO with vLLM rollout;
- mandatory Flash Attention from an exact prebuilt wheel;
- Python 3.11 and uv-based dependency management.

## Selected Baseline

| Package | Version |
| --- | --- |
| Python | 3.11.15, installed independently by uv |
| CUDA wheel runtime | 12.8 |
| torch | 2.7.0+cu128 |
| torchvision | 0.22.0+cu128 |
| torchaudio | 2.7.0+cu128 |
| flash-attn | 2.7.4.post1 |
| vllm | 0.9.1 |
| verl | 0.6.0 |
| transformers | 4.53.2 |
| trl | 0.19.1 |
| peft | 0.15.2 |
| accelerate | 1.7.0 |
| datasets | 3.6.0 |
| tokenizers | 0.21.2 |
| numpy | 1.26.4 |
| pyarrow | 19.0.1 |
| opencv-python-headless | 4.11.0.86 |
| ray | 2.48.0 |
| tensordict | 0.10.0 |
| torchdata | 0.11.0 |
| hydra-core | 1.3.2 |
| wandb | 0.21.4 |
| tensorboard | 2.19.0 |
| codetiming | 1.4.0 |
| dill | 0.3.8 |
| pandas | 2.2.3 |
| pybind11 | 2.13.6 |
| pylatexenc | 2.10 |
| packaging | 24.2 |
| safetensors | 0.5.3 |
| pytest (development) | 8.3.5 |

Every direct project dependency must be pinned to an exact version.
`uv.lock` must freeze all transitive dependencies. Datasets 3.6.0 requires
`dill<0.3.9`, which is why the baseline uses dill 0.3.8.

## Compatibility Rationale

The NVIDIA 580.105.08 driver can execute CUDA 12.8 binaries. A100 is an
Ampere `sm80` device supported by PyTorch, vLLM, and FlashAttention-2.
glibc 2.35 and Linux x86_64 satisfy the selected binary wheels.

vLLM 0.9.1 pins:

- torch 2.7.0;
- torchvision 0.22.0;
- torchaudio 2.7.0.

verl 0.6.0 declares vLLM compatibility through 0.9.1 and its release moved
the maintained vLLM environment to 0.9.1. The selected Hugging Face stack
satisfies TRL 0.19.1 and vLLM 0.9.1 constraints without requiring
Transformers 5.

NumPy remains at 1.26.4 because verl requires NumPy below 2. OpenCV is pinned
to 4.11.0.86 because newer OpenCV 4.13 metadata requires NumPy 2 on supported
Python versions.

## Flash Attention Binary

Flash Attention is mandatory and must not be compiled with the available
CUDA 12.4 toolkit.

PyTorch 2.7 official Linux wheels use CXX11 ABI 1. The only accepted Flash
Attention artifact is:

```text
flash_attn-2.7.4.post1+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
```

Official SHA-256:

```text
22013b8c74a63fc70e69be1e10ff02e4ad8fec84a43600bdca67b434ed417113
```

After installing the locked environment, verify:

```bash
python - <<'PY'
import torch
assert torch.__version__.startswith("2.7.0+cu128"), torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
assert torch._C._GLIBCXX_USE_CXX11_ABI is True
print("Torch and ABI checks passed")
PY
```

The FALSE ABI wheel is not an alternative for this environment.

## Dependency Installation Design

The uv project will use:

- the configured PyPI mirror for ordinary packages;
- the official PyTorch cu128 index exclusively for the Torch family;
- the repository-local official CUDA 12.8 vLLM wheel:
  `vllm-0.9.1-cp38-abi3-manylinux1_x86_64.whl`;
- the repository-local TRUE ABI Flash Attention wheel named above.

Official artifact URLs:

```text
https://github.com/vllm-project/vllm/releases/download/v0.9.1/vllm-0.9.1-cp38-abi3-manylinux1_x86_64.whl
https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1%2Bcu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
```

The vLLM wheel SHA-256 is:

```text
28b99e8df39c7aaeda04f7e5353b18564a1a9d1c579691945523fc4777a1a8c8
```

Both local wheels must be declared in `[tool.uv.sources]`. Flash Attention
must be a direct project dependency so a later `uv sync` cannot remove it.
Wheel hashes must be checked before `uv lock`.

Install base `verl==0.6.0` only. Do not install `verl[gpu]`, `verl[trl]`, or
`verl[vllm]`:

- `verl[gpu]` can trigger a Flash Attention source build;
- `verl[trl]` constrains TRL to 0.9.6 or earlier;
- vLLM is already pinned explicitly to the selected local wheel.

TRL 0.19.1 is used by the independent Transformers/TRL training path. The
verl GRPO path must not import or rely on TRL.

Downloaded wheels remain excluded from Git. Their expected filenames and
official download locations are documented under `wheels/README.md`.

Python must not be borrowed from the AgentFlow virtual environment. Create
the interpreter and virtual environment independently:

```bash
uv python install 3.11.15
uv venv --python 3.11.15 --seed
```

The old `verl 0.7.1 + vLLM 0.17.0 + torch 2.10 cu129` constraints file will
be replaced, not retained as an alternative environment.

## Project API Constraints

The rewritten training code must target the selected package APIs:

- use the vLLM 0.9.1 `LLM.chat` interface;
- pass `chat_template_kwargs={"enable_thinking": False}`;
- use TRL 0.19.1 SFT and DPO configuration fields;
- use verl 0.6.0 configuration keys and worker interfaces;
- set `attn_implementation="flash_attention_2"` on every
  `AutoModelForCausalLM.from_pretrained` path used by the existing
  `post_train` Full SFT, LoRA, RFT, DPO, legacy GRPO, and evaluation code;
- enforce that setting in the shared
  `post_train/scripts/sft/train_full.py::load_model_and_tokenizer` loader,
  which is reused by Full SFT, LoRA, RFT, DPO, and legacy GRPO;
- require that shared loader to pass `torch_dtype=torch.bfloat16`;
- enforce the same setting in both evaluation loading branches in
  `post_train/scripts/eval/evaluate_model.py`: the adapter base-model branch
  and the merged/full-model branch;
- require both evaluation branches to pass `torch_dtype=torch.bfloat16`;
- do not copy configuration keys from verl 0.7.1 or current `main` without
  verification against tag `v0.6.0`.

The installed `flash-attn` package accelerates the Transformers training
path. vLLM uses its own attention kernels and backend selection; installing
`flash-attn` does not make it vLLM's attention implementation.

The data schema, solver, prompts, reward semantics, fixed evaluation set, and
model paths are unchanged by this environment migration.

## Validation Gates

Acceptance is split into two levels so the environment migration does not
claim to validate GRPO components that do not exist yet.

### Level 1: Environment And Existing-Code Hard Gates

The scripts and static tests for this level are implemented. The remote GPU
gates have not yet been executed; all hard gates below must pass before the
environment is accepted:

1. `uv pip check` reports no dependency conflicts.
2. Python reports `sys.version_info[:3] == (3, 11, 15)`, and package
   versions match using `Version(installed).base_version`.
3. PyTorch reports CUDA 12.8 and two visible A100 devices.
4. Torch separately reports complete version `2.7.0+cu128`,
   `torch.version.cuda == "12.8"`, CXX11 ABI 1, and matching wheel hashes.
5. Flash Attention imports and runs a BF16 forward/backward CUDA test.
6. Transformers loads local Qwen3-0.6B with
   `attn_implementation="flash_attention_2"`, confirms the selected
   implementation in the model configuration, and completes a BF16
   forward/backward step.
7. Tests prove that the existing shared training loader and both evaluation
   branches always pass `attn_implementation="flash_attention_2"` and
   `torch_dtype=torch.bfloat16`. Full SFT
   is covered directly, while static import/call-chain tests prove that
   `train_lora.py`, `train_rft.py`, `train_dpo.py`, and legacy
   `train_grpo.py` continue to reuse
   `train_full.load_model_and_tokenizer`. A Level 1 GPU smoke calls this real
   shared loader and completes a CUDA BF16 forward/backward step.
8. A two-rank NCCL all-reduce succeeds.
9. Ray reports exactly two GPU resources.
10. vLLM loads Qwen3-0.6B and executes `LLM.chat` with thinking disabled.
11. vLLM loads a two-GPU smoke topology with `tensor_parallel_size=2`.
12. Two independent Qwen3-8B vLLM processes run concurrently: GPU 0 and GPU
    1 each host a separate `tensor_parallel_size=1` engine and each completes
    `LLM.chat` without OOM or cross-device leakage. The test records each
    process's `CUDA_VISIBLE_DEVICES`, actual CUDA device, visible and free
    memory and generated output. The parent uses standard-library
    `subprocess` to run `nvidia-smi
    --query-gpu=index,uuid,pci.bus_id --format=csv,noheader,nounits`, maps
    launch indices to physical UUID/PCI identities, and requires both to
    differ. Missing `nvidia-smi` or an unmappable index fails Level 1.
13. TRL constructs minimal SFT and DPO trainers.
14. PEFT creates, saves, reloads, and merges a LoRA adapter.

Under HAMI, CUDA peer-to-peer access or IPC may be unavailable even when the
supported distributed topology works. P2P/IPC checks therefore emit
diagnostic warnings and are not hard failures. NCCL all-reduce and vLLM TP2
remain Level 1 hard gates. A unit test must force P2P to false and verify a
warning is emitted while the successful NCCL smoke exits with status zero.

### Level 2: GRPO Integration Hard Gate

This level is deferred until the JSONL-to-verl-Parquet converter, Countdown
custom reward adapter, and verl GRPO configuration have been implemented.
It must then run a minimal two-GPU FSDP2 plus vLLM GRPO batch and complete one
optimizer update. Acceptance requires recorded Ray placement, actor and
rollout GPU assignments, observed free memory, reward and response-length
metrics, and no CUDA OOM, NCCL, weight-synchronization, or optimizer error.

The future GRPO implementation must provide:

- `post_train_v2/configs/verl/grpo_countdown_smoke.yaml`;
- `post_train_v2/scripts/verl/launch_grpo.py`;
- `post_train_v2/tests/integration/test_verl_grpo_smoke.py`.

Its executable Level 2 verification command will be:

```bash
pytest -q post_train_v2/tests/integration/test_verl_grpo_smoke.py \
  --verl-config post_train_v2/configs/verl/grpo_countdown_smoke.yaml
```

The current environment migration must not claim that Level 2 has passed.
These future files are specified for the later GRPO implementation stage and
are not implemented in the current migration. GRPO training must not begin
until Level 2 passes.

## Files To Update

The implementation will update:

- `post_train_v2/configs/environment/runtime-cu128.json` as the canonical
  machine-readable version and artifact manifest;
- `post_train_v2/pyproject.toml`;
- the version constraints file, renamed for the new baseline;
- `post_train_v2/requirements-runtime.txt`;
- `post_train_v2/wheels/README.md`;
- `post_train_v2/docs/environment_setup.md`;
- `post_train_v2/environment.md`;
- `post_train_v2/README.md`;
- `post_train_v2/migration_plan.md`;
- `post_train_v2/open_questions.md`;
- new environment validation scripts under `post_train_v2/scripts/env/`;
- `post_train_v2/scripts/env/smoke_teacher_dual_engine.py`;
- tests for artifact metadata and validation-script behavior;
- a focused Teacher dual-engine orchestration test covering GPU isolation,
  concurrency, timeout, and failure propagation;
- `post_train/scripts/sft/train_full.py`;
- `post_train/scripts/eval/evaluate_model.py`;
- `post_train/tests/test_evaluate_model_loader.py`;
- `post_train/tests/test_train_full_model_loader.py` as the dedicated shared
  training-loader test;
- a static import/call-chain test covering `train_lora.py`, `train_rft.py`,
  `train_dpo.py`, and legacy `train_grpo.py`.

The migration changes only the existing model-loading behavior in the old
training and evaluation code. It does not otherwise rewrite the training
loops, data flow, reward logic, or rollout implementation.

## Rejected Alternatives

### verl 0.7.1, vLLM 0.17.0, CUDA 13

This provides official Python 3.12/Torch 2.10 Flash Attention and vLLM
wheels, but the published dependency metadata does not close cleanly:

- verl requires NumPy below 2;
- vLLM 0.17 requires OpenCV 4.13 or newer;
- OpenCV 4.13 requires NumPy 2;
- verl's published vLLM extra still caps vLLM at 0.12.0.

It would require dependency overrides and would not satisfy the clean
`uv pip check` requirement.

### Building Flash Attention locally

The available compiler is CUDA 12.4 while the runtime baseline would be CUDA
12.8 or 12.9. Because an exact official binary is available for the selected
Torch 2.7/Python 3.11 stack, local compilation adds risk without benefit.
