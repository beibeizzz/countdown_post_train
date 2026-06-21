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

## 7. 正式运行前检查

### 7.1 模型和原始数据

`post_train/configs/data_build.yaml` 的原始数据位于仓库根目录 `datasets/`，
不是 `post_train/data/`：

```bash
test -f datasets/raw_train.parquet
test -f datasets/raw_test.json
test -f post_train/model/qwen/qwen3-0.6b/config.json
test -f post_train/model/qwen/qwen3-8b/config.json

du -sh datasets post_train/model/qwen/qwen3-0.6b post_train/model/qwen/qwen3-8b
df -h .
nvidia-smi
ps -ef | grep -E 'python|vllm' | grep -v grep || true
```

任一 `test -f` 失败时先修复路径，不要启动训练。进入新阶段前确保没有上一个
vLLM/Trainer 进程继续占用 GPU。

### 7.2 配置快照

```bash
RUN_TAG=$(date +%Y%m%d-%H%M%S)
mkdir -p post_train/outputs/run_configs/$RUN_TAG
cp post_train/configs/*.yaml post_train/outputs/run_configs/$RUN_TAG/
echo $RUN_TAG
```

重点检查：模型路径、训练数据、输出目录、`max_new_tokens`、batch size、
gradient accumulation、W&B 开关以及所有 `enable_thinking: false`。

## 8. 构造 solver-backed 基础数据

### 8.1 隔离 smoke

`build_source.py` 支持 `--limit`，并允许通过复制配置改变输出目录：

```bash
mkdir -p /tmp/post_train_smoke/configs /tmp/post_train_smoke/data/processed
python - <<'PY'
from pathlib import Path
import yaml

config = yaml.safe_load(Path(post_train/configs/data_build.yaml).read_text())
config[output_dir] = /tmp/post_train_smoke/data/processed
Path(/tmp/post_train_smoke/configs/data_build.yaml).write_text(
    yaml.safe_dump(config, sort_keys=False)
)
PY
python post_train/scripts/data/build_source.py \
  --config /tmp/post_train_smoke/configs/data_build.yaml \
  --limit 100
wc -l /tmp/post_train_smoke/data/processed/*.jsonl
python -m json.tool /tmp/post_train_smoke/data/processed/manifest.json | head -n 80
```

### 8.2 正式构造

```bash
python post_train/scripts/data/build_source.py \
  --config post_train/configs/data_build.yaml
```

| 文件 | 用途 |
| --- | --- |
| `post_train/data/processed/source_all.jsonl` | 所有 solver-backed 规范样本 |
| `train_pool.jsonl` | 排除固定验证集后的 Teacher 候选池 |
| `val_200.jsonl` | 固定 200 条验证集 |
| `val_eval_50.jsonl` | 每 100 optimizer steps 使用的固定 50 条 |
| `test_with_solver_answers.jsonl` | solver 补全答案的测试集 |
| `unsolved_train.jsonl` | solver 未找到答案的原始训练样本 |
| `manifest.json` | 数据计数、配置和 schema 信息 |

`train_source.jsonl`、`val.jsonl`、`eval_subset.jsonl` 和 `test.jsonl` 是兼容
别名。

```bash
wc -l \
  post_train/data/processed/train_pool.jsonl \
  post_train/data/processed/val_200.jsonl \
  post_train/data/processed/val_eval_50.jsonl \
  post_train/data/processed/test_with_solver_answers.jsonl
python -m json.tool post_train/data/processed/manifest.json | head -n 100
```

## 9. 构造 Teacher accepted pool

这一阶段使用 Qwen3-8B、单个 TP=1 vLLM 引擎，按输入顺序 rollout 一次，直到
累计 20,000 条正确样本。

```bash
export CUDA_VISIBLE_DEVICES=0
python post_train/scripts/data/build_teacher_pool.py \
  --config post_train/configs/teacher_rollout.yaml
```

输出固定在：

- `post_train/data/teacher_rollouts/teacher_accepted_20k.jsonl`
- `post_train/data/teacher_rollouts/teacher_rejected.jsonl`
- `post_train/data/teacher_rollouts/manifest.json`

脚本支持从 accepted/rejected 文件继续处理尚未出现的输入 ID，并使用
`.teacher_pool.lock` 防止并发写入。只有确认锁对应进程已经不存在时才使用：

```bash
python post_train/scripts/data/build_teacher_pool.py \
  --config post_train/configs/teacher_rollout.yaml \
  --recover-stale-lock
```

不要把 `post_train_v2` Teacher 产物复制到该目录。旧版入口会拒绝 V2-owned
manifest 或 transaction marker。Teacher 脚本没有 `--limit` 或可配置输出目录，
因此不要用正式目录做短 smoke；使用第 5 节的 vLLM gate 代替。

```bash
wc -l post_train/data/teacher_rollouts/teacher_accepted_20k.jsonl
python -m json.tool post_train/data/teacher_rollouts/manifest.json | head -n 120
```

## 10. 构造 SFT 8k 和 GRPO 4k

前置条件是 accepted pool 至少包含 8,000 条样本：

```bash
python post_train/scripts/data/build_sft_splits.py \
  --config post_train/configs/data_build.yaml
```

输出：

- `post_train/data/sft/sft_train_8k.jsonl`
- `post_train/data/grpo/grpo_train_4k.jsonl`
- `post_train/data/sft/manifest.json`

```bash
wc -l \
  post_train/data/sft/sft_train_8k.jsonl \
  post_train/data/grpo/grpo_train_4k.jsonl
python -m json.tool post_train/data/sft/manifest.json
```

该脚本的输出目录是代码常量，没有隔离 smoke 输出参数。不要使用小 accepted
文件覆盖正式 split；先完成 accepted pool，再执行一次正式分层采样。

## 11. Full SFT

Full SFT 对 Qwen3-0.6B 做 response 全量监督，包括 Teacher response 中可能存在的
有用推理。最大序列长度为 256，模型加载强制 Flash Attention 2 和 BF16。

### 11.1 隔离两步 smoke

```bash
mkdir -p /tmp/post_train_smoke/configs /tmp/post_train_smoke/outputs/sft/full
python - <<'PY'
from pathlib import Path
import yaml

config = yaml.safe_load(Path(post_train/configs/sft_full.yaml).read_text())
config[output_dir] = /tmp/post_train_smoke/outputs/sft/full
config[report_to] = None
Path(/tmp/post_train_smoke/configs/sft_full.yaml).write_text(
    yaml.safe_dump(config, sort_keys=False)
)
PY
CUDA_VISIBLE_DEVICES=0 python post_train/scripts/sft/train_full.py \
  --config /tmp/post_train_smoke/configs/sft_full.yaml \
  --max-steps 2
```

### 11.2 正式训练

```bash
CUDA_VISIBLE_DEVICES=0 python post_train/scripts/sft/train_full.py \
  --config post_train/configs/sft_full.yaml
```

主要产物：

- `post_train/outputs/sft/full/checkpoint-*`
- `post_train/outputs/sft/full/eval/step_*`
- `post_train/outputs/sft/full/final/`

`final/` 是可直接由 `AutoModelForCausalLM.from_pretrained()` 加载的完整模型。

## 12. LoRA SFT

```bash
mkdir -p /tmp/post_train_smoke/configs /tmp/post_train_smoke/outputs/sft/lora
python - <<'PY'
from pathlib import Path
import yaml

config = yaml.safe_load(Path(post_train/configs/sft_lora.yaml).read_text())
config[output_dir] = /tmp/post_train_smoke/outputs/sft/lora
config[report_to] = None
Path(/tmp/post_train_smoke/configs/sft_lora.yaml).write_text(
    yaml.safe_dump(config, sort_keys=False)
)
PY
CUDA_VISIBLE_DEVICES=0 python post_train/scripts/sft/train_lora.py \
  --config /tmp/post_train_smoke/configs/sft_lora.yaml \
  --max-steps 2

CUDA_VISIBLE_DEVICES=0 python post_train/scripts/sft/train_lora.py \
  --config post_train/configs/sft_lora.yaml
```

LoRA 的 `final/` 是 adapter，不是独立完整模型。评估时需要 adapter config 中的
base model 路径，或显式传 `--base-model-path`。

## 13. RFT

RFT 分两步：先多次 rollout 并保留正确 response，再复用 Full SFT trainer 训练。

```bash
CUDA_VISIBLE_DEVICES=0 python post_train/scripts/sft/build_rft_data.py \
  --config post_train/configs/rft.yaml
CUDA_VISIBLE_DEVICES=0 python post_train/scripts/sft/train_rft.py \
  --config post_train/configs/rft.yaml
```

当前 `rft.yaml` 的 `base_model_path` 是本地 Qwen3-8B。运行前必须确认这符合当前
实验意图；如果目标是用 Full SFT 0.6B 做拒绝采样，应复制配置后显式改为
`post_train/outputs/sft/full/final`。本文不静默修改默认 YAML。

安全 smoke 必须同时改写 `accepted_output` 和 `output_dir`：

```bash
mkdir -p /tmp/post_train_smoke/data /tmp/post_train_smoke/outputs/sft/rft
python - <<'PY'
from pathlib import Path
import yaml

config = yaml.safe_load(Path(post_train/configs/rft.yaml).read_text())
config[accepted_output] = /tmp/post_train_smoke/data/rft_accepted.jsonl
config[output_dir] = /tmp/post_train_smoke/outputs/sft/rft
config[train][report_to] = None
Path(/tmp/post_train_smoke/configs/rft.yaml).write_text(
    yaml.safe_dump(config, sort_keys=False)
)
PY
CUDA_VISIBLE_DEVICES=0 python post_train/scripts/sft/build_rft_data.py \
  --config /tmp/post_train_smoke/configs/rft.yaml \
  --limit 8
CUDA_VISIBLE_DEVICES=0 python post_train/scripts/sft/train_rft.py \
  --config /tmp/post_train_smoke/configs/rft.yaml \
  --max-steps 2
```

## 14. DPO

DPO 数据构造使用 Qwen3-8B，为 SFT chosen response 生成 forced-wrong 和高温
rollout rejected，并按五类错误过滤，目标约 6,000 对。

### 14.1 隔离 smoke

```bash
mkdir -p /tmp/post_train_smoke/data/dpo /tmp/post_train_smoke/outputs/dpo
python - <<'PY'
from pathlib import Path
import yaml

data = yaml.safe_load(Path(post_train/configs/dpo_data.yaml).read_text())
data[output_dir] = /tmp/post_train_smoke/data/dpo
Path(/tmp/post_train_smoke/configs/dpo_data.yaml).write_text(
    yaml.safe_dump(data, sort_keys=False)
)

train = yaml.safe_load(Path(post_train/configs/dpo_train.yaml).read_text())
train[train_data] = /tmp/post_train_smoke/data/dpo/dpo_train.jsonl
train[output_dir] = /tmp/post_train_smoke/outputs/dpo
train[report_to] = None
Path(/tmp/post_train_smoke/configs/dpo_train.yaml).write_text(
    yaml.safe_dump(train, sort_keys=False)
)
PY
CUDA_VISIBLE_DEVICES=0 python post_train/scripts/dpo/build_dpo_data.py \
  --config /tmp/post_train_smoke/configs/dpo_data.yaml \
  --limit 8
CUDA_VISIBLE_DEVICES=0 python post_train/scripts/dpo/train_dpo.py \
  --config /tmp/post_train_smoke/configs/dpo_train.yaml \
  --max-steps 2
```

如果 8 条 chosen 无法产生可训练 pair，应增大 `--limit`，不要降低 correctness
过滤标准。

### 14.2 正式 DPO

```bash
CUDA_VISIBLE_DEVICES=0 python post_train/scripts/dpo/build_dpo_data.py \
  --config post_train/configs/dpo_data.yaml
CUDA_VISIBLE_DEVICES=0 python post_train/scripts/dpo/train_dpo.py \
  --config post_train/configs/dpo_train.yaml
```

数据输出位于 `post_train/data/dpo/`，模型输出位于
`post_train/outputs/dpo/`。

## 15. Legacy GRPO

当前 GRPO 不是 verl/FSDP 实现。它在同一 Python 进程中加载一个可训练的
Qwen3-0.6B Transformers 模型和一个 `tensor_parallel_size=1` 的 vLLM rollout
模型。两者共享唯一可见 GPU，因此运行前必须清空残留进程并检查显存。

默认 `kl_coeff=0.0`、每个 prompt rollout 4 条、每次 rollout 更新 policy 两次、
每 20 步同步/保存、每 100 步固定评估。

### 15.1 隔离两步 smoke

```bash
python - <<'PY'
from pathlib import Path
import yaml

config = yaml.safe_load(Path(post_train/configs/grpo.yaml).read_text())
config[output_dir] = /tmp/post_train_smoke/outputs/grpo
config[report_to] = None
Path(/tmp/post_train_smoke/configs/grpo.yaml).write_text(
    yaml.safe_dump(config, sort_keys=False)
)
PY
CUDA_VISIBLE_DEVICES=0 python post_train/scripts/grpo/train_grpo.py \
  --config /tmp/post_train_smoke/configs/grpo.yaml \
  --max-steps 2
```

### 15.2 正式训练

```bash
CUDA_VISIBLE_DEVICES=0 python post_train/scripts/grpo/train_grpo.py \
  --config post_train/configs/grpo.yaml
```

```bash
tail -n 5 post_train/outputs/grpo/metrics.jsonl
find post_train/outputs/grpo -maxdepth 2 -type d -name 'checkpoint-*'
test -d post_train/outputs/grpo/final
```

指标包括 reward、accuracy、format、loss、KL、response length、截断数和 group
reward 标准差。`compute_entropy: false` 时不会计算 entropy。

## 16. W&B 监控

默认 `report_to: null`，不会创建 W&B run。需要启用时：

```bash
python -c "import wandb; print(wandb.__version__)"
wandb login
```

在对应配置中设置 `report_to: wandb`、项目、分组、run name 和
`logging_steps`。Full/LoRA/RFT/DPO 使用 Trainer 集成；legacy GRPO 每步保留
本地 `metrics.jsonl`，并在启用时同步到 W&B。离线 evaluator 不上传 W&B。

## 17. 恢复和阶段验收

### 17.1 Trainer checkpoint

```bash
find post_train/outputs -maxdepth 4 -type d -name 'checkpoint-*' | sort
```

当前训练 CLI 没有公开 `--resume-from-checkpoint` 参数。不要假定 Trainer 会自动
选择 checkpoint；恢复正式训练前先核对脚本行为和目标输出目录，避免覆盖已有
结果。

### 17.2 Teacher lock

```bash
ls -la post_train/data/teacher_rollouts
ps -ef | grep build_teacher_pool.py | grep -v grep || true
```

只有锁对应进程不存在时才使用 `--recover-stale-lock`。

### 17.3 每阶段完成条件

- 数据阶段：目标 JSONL 行数合理，manifest 可解析。
- Teacher：accepted 达到 20,000，manifest 记录 completed 状态。
- Trainer：`final/config.json` 或 adapter config 存在，固定评估目录可读。
- DPO：pair/category 数量满足配置约束。
- GRPO：`metrics.jsonl` 持续写入，checkpoint 和 `final/` 存在。
- 所有模型：必须通过第 18 节的独立评估，而不是只看训练 loss。

## 18. 独立评估矩阵

`evaluate_model.py` 对完整模型和 LoRA adapter 使用相同的固定数据、chat
template、256-token 上限和 solver 判定。每个模型必须使用独立输出目录。

### 18.1 Base 与完整模型

```bash
CUDA_VISIBLE_DEVICES=0 python post_train/scripts/eval/evaluate_model.py \
  --config post_train/configs/eval.yaml \
  --model-path post_train/model/qwen/qwen3-0.6b \
  --output-dir post_train/data/eval/base_0_6b

CUDA_VISIBLE_DEVICES=0 python post_train/scripts/eval/evaluate_model.py \
  --config post_train/configs/eval.yaml \
  --model-path post_train/outputs/sft/full/final \
  --output-dir post_train/data/eval/sft_full

CUDA_VISIBLE_DEVICES=0 python post_train/scripts/eval/evaluate_model.py \
  --config post_train/configs/eval.yaml \
  --model-path post_train/outputs/sft/rft/final \
  --output-dir post_train/data/eval/rft

CUDA_VISIBLE_DEVICES=0 python post_train/scripts/eval/evaluate_model.py \
  --config post_train/configs/eval.yaml \
  --model-path post_train/outputs/dpo/final \
  --output-dir post_train/data/eval/dpo

CUDA_VISIBLE_DEVICES=0 python post_train/scripts/eval/evaluate_model.py \
  --config post_train/configs/eval.yaml \
  --model-path post_train/outputs/grpo/final \
  --output-dir post_train/data/eval/grpo
```

如果某个可选阶段尚未生成 `final/`，跳过该命令并在实验记录中标记为未训练，
不要把缺失模型记录为 accuracy 0。

### 18.2 LoRA adapter

```bash
CUDA_VISIBLE_DEVICES=0 python post_train/scripts/eval/evaluate_model.py \
  --config post_train/configs/eval.yaml \
  --model-path post_train/outputs/sft/lora/final \
  --base-model-path post_train/model/qwen/qwen3-0.6b \
  --output-dir post_train/data/eval/sft_lora
```

评估器通过 `adapter_config.json` 自动识别 LoRA。如果 adapter metadata 中没有
可用 base model 路径，必须显式传入 `--base-model-path`。

### 18.3 快速 smoke 与输出解释

```bash
CUDA_VISIBLE_DEVICES=0 python post_train/scripts/eval/evaluate_model.py \
  --config post_train/configs/eval.yaml \
  --model-path post_train/outputs/sft/full/final \
  --output-dir /tmp/post_train_smoke/eval/sft_full \
  --limit 10
```

每个输出目录包含：

- `eval_samples.jsonl`：prompt、完整 response、提取表达式、格式、correctness、
  token 数和截断状态；
- `eval_metrics.json`：`accuracy`、`format_rate`、
  `valid_expression_rate`、`avg_generated_tokens`、
  `max_generated_tokens` 和 `truncated_count`。

比较模型时优先看 accuracy，其次看 format rate、截断数量和平均输出长度。不能
只比较 Trainer loss。
