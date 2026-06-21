# Legacy Post-Train 远程训练指南

本文只适用于现有 `post_train/` 项目，不适用于 `post_train_v2/` 的双 GPU
分布式流程。所有命令默认从仓库根目录执行。

## 1. 硬件执行模型

当前 `post_train/` 是单 GPU 实现：

- Teacher、RFT 和 DPO 数据生成各创建一个 vLLM 引擎，默认
  `tensor_parallel_size=1`。
- Full SFT、LoRA SFT、RFT 和 DPO 使用普通 Transformers/TRL Trainer
  入口，没有 DDP、FSDP、DeepSpeed 或 `torchrun` 拓扑。
- Legacy GRPO 在一个进程中同时持有可训练 Transformers 模型和一个 TP=1
  vLLM rollout 引擎。

因此，一张可用 CUDA GPU 是必要条件。建议先固定 GPU 0：

```bash
export CUDA_VISIBLE_DEVICES=0
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
```

如果远程机器提供第二张 GPU，可以独立测试 GPU 1，但 NCCL、P2P 和双卡训练
不是本项目的验收条件。GPU 1 测试失败不会否定 GPU 0 的单卡流程，但在修复前
不得把 GPU 1 用于本项目。

## 2. 验证现有 uv 环境

本节只审计已经激活的 uv 环境，不默认创建、删除或同步环境。

### 2.1 确认解释器身份

```bash
pwd
which uv
uv --version
which python
python -V
python -c "import sys; print(sys.executable); print(sys.path)"
uv pip check
```

验收标准：

- `which python` 和 `sys.executable` 指向预期的 uv 虚拟环境。
- Python 版本能够被当前 PyTorch、Transformers、TRL 和 vLLM 组合支持。
- `uv pip check` 没有与训练依赖有关的冲突。

不要因为 `uv pip check` 输出了无关应用的冲突就直接重装全部环境。先确认冲突包
是否会被 `post_train/` 导入。

### 2.2 记录关键包版本

```bash
python - <<'PY'
from importlib.metadata import PackageNotFoundError, version

packages = [
    "torch",
    "transformers",
    "trl",
    "peft",
    "accelerate",
    "datasets",
    "vllm",
    "flash-attn",
    "pandas",
    "pyarrow",
    "PyYAML",
    "tokenizers",
    "safetensors",
    "wandb",
    "pytest",
]

for package in packages:
    try:
        print(f"{package:15} {version(package)}")
    except PackageNotFoundError:
        print(f"{package:15} NOT INSTALLED")
PY
```

训练必需：`torch`、`transformers`、`trl`、`peft`、`accelerate`、
`datasets`、`vllm`、`flash-attn`、`pandas`、`pyarrow`、`PyYAML`、
`tokenizers` 和 `safetensors`。

`wandb` 只在启用在线监控时需要，`pytest` 只在运行测试时需要。版本号不要求与
`post_train_v2/` 完全一致，最终以导入和模型运行 gate 为准。

### 2.3 导入 gate

```bash
python - <<'PY'
import accelerate
import datasets
import flash_attn
import pandas
import peft
import pyarrow
import safetensors
import tokenizers
import torch
import transformers
import trl
import vllm
import yaml

print("OK: required imports succeeded")
PY
```

导入成功只证明 Python 包可见，不代表 CUDA、Flash Attention 或 vLLM 可以加载
本地模型。

## 3. CUDA 与 GPU gate

### 3.1 驱动和设备信息

```bash
nvidia-smi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
```

注意：`nvidia-smi` 显示的 CUDA 版本是驱动最高兼容版本，不等于 PyTorch 实际
编译使用的 CUDA runtime。

### 3.2 PyTorch、BF16 和真实显存分配

```bash
CUDA_VISIBLE_DEVICES=0 python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("visible GPUs:", torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    print(
        index,
        torch.cuda.get_device_name(index),
        torch.cuda.get_device_capability(index),
    )
print("BF16 supported:", torch.cuda.is_bf16_supported())

x = torch.empty(256 * 1024 * 1024, dtype=torch.float32, device="cuda")
torch.cuda.synchronize()
print("allocated bytes:", torch.cuda.memory_allocated())
print("memory info:", torch.cuda.mem_get_info())
del x
torch.cuda.empty_cache()
print("released memory info:", torch.cuda.mem_get_info())
PY
```

验收标准：

- `torch.cuda.is_available()` 为 `True`。
- 可见设备数量为 1，因为命令已经固定 GPU 0。
- BF16 可用。
- 1 GiB 实际分配成功，并且脚本正常退出。

如果机器有第二张 GPU，再独立执行：

```bash
CUDA_VISIBLE_DEVICES=1 python - <<'PY'
import torch

print(torch.cuda.get_device_name(0))
x = torch.empty(256 * 1024 * 1024, dtype=torch.float32, device="cuda")
torch.cuda.synchronize()
print("allocated bytes:", torch.cuda.memory_allocated())
del x
PY
```

在 HAMI/vGPU 环境中，不能忽略 `host pid is error`、显存使用量不变化或
`cudaDeviceGetUuid` 等异常；这些异常意味着该 GPU 的进程或显存记账不可信。

## 4. Flash Attention 2 模型 gate

当前 Full SFT、LoRA、RFT、DPO、GRPO 的 Transformers 模型加载以及独立评估
都会强制使用 `attn_implementation="flash_attention_2"`。必须进行真实模型
加载，而不是只执行 `import flash_attn`。

```bash
CUDA_VISIBLE_DEVICES=0 python - <<'PY'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

path = "post_train/model/qwen/qwen3-0.6b"
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
).to("cuda")

print("tokenizer:", type(tokenizer).__name__)
print("attention:", model.config._attn_implementation)
print("dtype:", next(model.parameters()).dtype)
print("device:", next(model.parameters()).device)
PY
```

验收标准：attention 为 `flash_attention_2`，权重为 BF16，设备为 CUDA。

## 5. Qwen3-8B vLLM 单卡 gate

使用项目自己的 `VLLMGenerator`，这样 smoke 与生产流程使用相同的 Qwen chat
template，并显式关闭 thinking。

```bash
mkdir -p /tmp/post-train-vllm-cache

CUDA_VISIBLE_DEVICES=0 \
VLLM_CACHE_ROOT=/tmp/post-train-vllm-cache \
python - <<'PY'
from post_train.src.countdown.generation import GenerationConfig, VLLMGenerator

generator = VLLMGenerator(
    "post_train/model/qwen/qwen3-8b",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.8,
    max_model_len=512,
)
result = generator.generate(
    ["Using the numbers [1, 1, 1, 1], create an equation equal to 4."],
    GenerationConfig(
        max_new_tokens=64,
        temperature=0.0,
        top_p=1.0,
        enable_thinking=False,
    ),
)
print(result[0])
PY
```

验收标准：引擎初始化成功、输出非空、没有 CUDA/HAMI 错误，cache 指向 `/tmp`。
运行下一个训练阶段前，使用 `nvidia-smi` 确认没有遗留 vLLM 进程。

## 6. 失败后的定向修复原则

只有出现明确失败时才考虑安装或调整版本：

| 失败类型 | 下一步 |
| --- | --- |
| `NOT INSTALLED` | 只安装缺失 distribution，并重新运行导入 gate |
| `uv pip check` 冲突 | 确认冲突包是否属于训练依赖，再调整单个依赖组合 |
| CUDA unavailable | 检查容器 GPU 挂载、驱动和 PyTorch CUDA build |
| Flash Attention 导入失败 | 检查 Python、PyTorch、CUDA、CXX11 ABI 与 wheel/build 是否匹配 |
| Qwen3-0.6B FA2 加载失败 | 保留完整 traceback，先解决 FA2，不要退回 eager attention |
| vLLM 初始化失败 | 检查模型路径、显存、cache、残留进程和 vLLM/PyTorch 兼容性 |

不要默认执行以下操作：

- 删除现有 uv 环境；
- 无审查地运行 `uv sync`；
- 套用 `post_train_v2/` 的固定版本锁；
- 为了绕过 gate 而移除 Flash Attention 2；
- 把双 GPU/NCCL 作为旧版项目的前置条件。
