# Environment Validation Scripts

These scripts implement the static and remote GPU acceptance checks for the
pinned runtime in `../../configs/environment/runtime-cu128.json`.

Run `python <script> --help` for arguments.

| Script | Purpose |
| --- | --- |
| `verify_artifacts.py` | Verify exact wheel filenames and SHA-256 hashes before installation. |
| `check_runtime.py` | Check package pins, Python, CUDA runtime, ABI, A100 devices, P2P/IPC diagnostics, and optional Ray resources. |
| `smoke_flash_attention.py` | Run direct BF16 Flash Attention forward/backward. |
| `smoke_transformers.py` | Run Qwen3 Transformers forward/backward with Flash Attention 2. |
| `smoke_legacy_loader.py` | Exercise the existing shared training model loader. |
| `smoke_eval_loader.py` | Exercise full-model or LoRA evaluation loading and generation. |
| `smoke_nccl.py` | Run a two-rank NCCL all-reduce under `torchrun`. |
| `smoke_vllm.py` | Run vLLM chat generation with TP1 or TP2 and thinking disabled. |
| `smoke_teacher_dual_engine.py` | Launch two isolated concurrent one-GPU Qwen3-8B teacher engines. |
| `smoke_trl_peft.py` | Validate PEFT save/reload/merge and run one SFT and one DPO training step. |

The complete command order and acceptance criteria are in
`../../docs/environment_setup.md`.
