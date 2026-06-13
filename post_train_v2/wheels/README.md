# Local Runtime Wheels

Place exactly these two official release artifacts in this directory:

| Package | Filename | SHA-256 |
| --- | --- | --- |
| vLLM 0.9.1 | `vllm-0.9.1-cp38-abi3-manylinux1_x86_64.whl` | `28b99e8df39c7aaeda04f7e5353b18564a1a9d1c579691945523fc4777a1a8c8` |
| Flash Attention 2.7.4.post1 | `flash_attn-2.7.4.post1+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl` | `22013b8c74a63fc70e69be1e10ff02e4ad8fec84a43600bdca67b434ed417113` |

Official URLs:

- <https://github.com/vllm-project/vllm/releases/download/v0.9.1/vllm-0.9.1-cp38-abi3-manylinux1_x86_64.whl>
- <https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1%2Bcu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl>

Only the Flash Attention wheel built with `cxx11abiTRUE` is accepted. Do not
substitute a FALSE ABI wheel and do not build Flash Attention from source.

Verify both files before creating the environment:

```bash
cd post_train_v2
python3 scripts/env/verify_artifacts.py \
  --manifest configs/environment/runtime-cu128.json \
  --wheels-dir wheels
```

The wheel files are intentionally ignored by Git. Only this README is tracked.
