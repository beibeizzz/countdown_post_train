# Local Wheels

This directory is for manually transferred binary/source artifacts that are
not downloaded through the default package mirror.

Expected artifacts:

- `vllm-0.17.0+cu128-cp38-abi3-manylinux_2_35_x86_64.whl`
- `flash-attention-2.8.3.tar.gz`, if no exact compatible Flash Attention
  wheel exists
- a locally built `flash_attn-2.8.3-*.whl`

Do not commit large wheel or source archive files to Git.

