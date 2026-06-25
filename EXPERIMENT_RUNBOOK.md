# Legacy post_train 正式实验流程

> 适用范围:旧版单 GPU `post_train/` 子项目,不涉及 `post_train_v2/` 的双 GPU 分布式流程。
> 所有命令默认在**仓库根目录** `/root/autodl-tmp/post_train/countdown_post_train` 下执行。
> 本流程已通过 7 阶段环境/硬件/gate/smoke 验收,代码侧无阻塞(含 GRPO 单卡 OOM 修复)。

---

## 0. 前置说明

### 0.1 环境基线(验收时确认)

| 项 | 值 |
|---|---|
| Python | 3.12.13 (`/root/autodl-tmp/.venv/bin/python`) |
| uv | 0.11.18 |
| PyTorch | 2.8.0+cu128 |
| Transformers | 4.57.1 |
| TRL | 1.5.1 |
| PEFT | 0.19.1 |
| vLLM | 0.10.2 |
| Flash Attention | 2.8.3 |
| GPU | NVIDIA GeForce RTX 4090,47.37 GiB,compute 8.9 |
| 可见 GPU | 1(`CUDA_VISIBLE_DEVICES=0` 固定) |
| wandb | 未安装(可选,默认 `report_to: null`) |

### 0.2 通用约定(每条命令前生效)

```bash
source /root/autodl-tmp/.venv/bin/activate
export OMP_NUM_THREADS=1            # 修复本机无效的 OMP_NUM_THREADS=0
export CUDA_VISIBLE_DEVICES=0       # legacy 全流程单 GPU
cd /root/autodl-tmp/post_train/countdown_post_train
```

- **不要并发跑两个 GPU 任务**(Teacher / SFT / DPO / GRPO 均独占单卡)。
- 每阶段完成后用 `nvidia-smi` 确认无残留进程再进入下一阶段。
- 长任务(Teacher、各训练)建议在 `tmux` / `screen` 中运行,断网不中断。
- 正式跑前按指南 7.2 节存档配置快照(见阶段 0)。

### 0.3 流水线依赖图

```
Source(已有) → Teacher(8b, 20k) → Splits(8k / 4k)
                                   │
                                   ├─→ Full SFT(0.6b) ──┬─→ DPO ──→ eval
                                   ├─→ LoRA(0.6b)        └─→ GRPO ──→ eval
                                   └─→ RFT(注意 base_model) ──→ eval

eval: base / sft_full / sft_lora / rft / dpo / grpo 各独立评估
```

### 0.4 阶段耗时与产物概览

| 阶段 | 模型 | 产物 | 预估耗时 | 可中断续跑 |
|---|---|---|---|---|
| 1 Teacher | 8b vLLM | `teacher_accepted_20k.jsonl`(20000) | **最长**(300 prompt→20k accepted) | 是(按 ID 续) |
| 2 Splits | CPU | `sft_train_8k.jsonl` / `grpo_train_4k.jsonl` | 秒级 | 否(幂等重跑) |
| 3 Full SFT | 0.6b | `outputs/sft/full/final/` | 中(3 epochs) | 否(无公开 resume) |
| 4 LoRA | 0.6b | `outputs/sft/lora/final/`(adapter) | 中 | 否 |
| 5 RFT | 8b 或 0.6b | `rft_accepted.jsonl` + `outputs/sft/rft/final/` | 中(含 rollout) | 数据阶段是 |
| 6 DPO | 0.6b(+8b 数据) | `data/dpo/` + `outputs/dpo/final/` | 中 | 数据阶段是 |
| 7 GRPO | 0.6b(+0.6b rollout) | `outputs/grpo/{metrics.jsonl,final/}` | 中长(500 步) | 否 |
| 8 Eval | 各 final | `data/eval/*/eval_metrics.json` | 短 | — |

---

## 阶段 0:正式运行前检查 + 配置快照

```bash
source /root/autodl-tmp/.venv/bin/activate
export OMP_NUM_THREADS=1
cd /root/autodl-tmp/post_train/countdown_post_train
export CUDA_VISIBLE_DEVICES=0

# 0.1 前置文件检查(任一缺失则先修复路径,不要启动训练)
test -f datasets/raw_train.parquet
test -f datasets/raw_test.json
test -f post_train/model/qwen/qwen3-0.6b/config.json
test -f post_train/model/qwen/qwen3-8b/config.json
test -f post_train/data/processed/train_pool.jsonl   # teacher 输入(300 行)

# 0.2 确认 GPU 空闲、无残留 vLLM/Trainer 进程
nvidia-smi
ps -ef | grep -E 'python|vllm|EngineCore' | grep -v grep || echo "clean"

# 0.3 配置快照(指南 7.2)
RUN_TAG=$(date +%Y%m%d-%H%M%S)
mkdir -p post_train/outputs/run_configs/$RUN_TAG
cp post_train/configs/*.yaml post_train/outputs/run_configs/$RUN_TAG/
echo "RUN_TAG=$RUN_TAG"

# 0.4 磁盘空间(当前约 23GB 可用;0.6b 训练产物都很小,够用)
df -h .
```

**通过标准**:0.1 全部存在;0.2 无残留进程;0.3 快照已存。

---

## 阶段 1:构造 Teacher accepted pool(8b vLLM,目标 20000 条)

> ⚠️ **全流程最久的一步**。输入仅 300 条 prompt,需 8b 反复 rollout 直到累计 20000 条正确样本。
> 8b 单引擎(无 policy 共享),48GB 足够;支持**断点续跑**(从 accepted/rejected 文件继续未完成 ID)。
> 强烈建议在 `tmux` / `screen` 中运行。

```bash
source /root/autodl-tmp/.venv/bin/activate
export OMP_NUM_THREADS=1
cd /root/autodl-tmp/post_train/countdown_post_train
export CUDA_VISIBLE_DEVICES=0

# 1.1 正式生成(可中断;重跑同命令自动续未完成 ID)
python post_train/scripts/data/build_teacher_pool.py \
  --config post_train/configs/teacher_rollout.yaml
```

若中途中断后重启,提示 stale lock 时再带恢复参数(仅当确认锁对应进程已不存在):

```bash
python post_train/scripts/data/build_teacher_pool.py \
  --config post_train/configs/teacher_rollout.yaml \
  --recover-stale-lock
```

**验收**:

```bash
wc -l post_train/data/teacher_rollouts/teacher_accepted_20k.jsonl   # 应 = 20000
python -m json.tool post_train/data/teacher_rollouts/manifest.json | head -n 120
nvidia-smi   # 确认进程退出、显存释放
```

**通过标准**:accepted = 20000 行;manifest 记录 completed;无残留进程。

---

## 阶段 2:构造 SFT 8k 与 GRPO 4k splits

> 前置:阶段 1 accepted ≥ 8000(已 20000)。输出目录为代码常量,无 `--limit` / 隔离输出参数。
> CPU 任务,可与阶段 1 收尾重叠。

```bash
source /root/autodl-tmp/.venv/bin/activate
export OMP_NUM_THREADS=1
cd /root/autodl-tmp/post_train/countdown_post_train

python post_train/scripts/data/build_sft_splits.py \
  --config post_train/configs/data_build.yaml
```

**验收**:

```bash
wc -l post_train/data/sft/sft_train_8k.jsonl      # 应 = 8000
wc -l post_train/data/grpo/grpo_train_4k.jsonl    # 应 = 4000
python -m json.tool post_train/data/sft/manifest.json
```

**通过标准**:8k = 8000、4k = 4000;manifest 可解析。

---

## 阶段 3:Full SFT(Qwen3-0.6B,正式)

> FA2 + BF16,seq_len 256,3 epochs。产物 `final/` 是可直接由 `AutoModelForCausalLM.from_pretrained()` 加载的完整模型,**DPO / GRPO 依赖它**。

```bash
source /root/autodl-tmp/.venv/bin/activate
export OMP_NUM_THREADS=1
cd /root/autodl-tmp/post_train/countdown_post_train
export CUDA_VISIBLE_DEVICES=0

python post_train/scripts/sft/train_full.py \
  --config post_train/configs/sft_full.yaml
```

**验收**:

```bash
test -d post_train/outputs/sft/full/final
test -f post_train/outputs/sft/full/final/config.json
test -f post_train/outputs/sft/full/final/model.safetensors
find post_train/outputs/sft/full -maxdepth 2 -type d -name 'checkpoint-*' | sort
nvidia-smi
```

**通过标准**:`final/config.json` 与 `model.safetensors` 存在;无残留进程。

---

## 阶段 4:LoRA SFT(Qwen3-0.6B,正式)

> 单卡须与阶段 3 串行。产物 `final/` 是 **adapter**(非完整模型),评估时需 base model 路径。

```bash
source /root/autodl-tmp/.venv/bin/activate
export OMP_NUM_THREADS=1
cd /root/autodl-tmp/post_train/countdown_post_train
export CUDA_VISIBLE_DEVICES=0

python post_train/scripts/sft/train_lora.py \
  --config post_train/configs/sft_lora.yaml
```

**验收**:

```bash
test -f post_train/outputs/sft/lora/final/adapter_model.safetensors
test -f post_train/outputs/sft/lora/final/adapter_config.json
# adapter_config.json 的 base_model_name_or_path 应指向 qwen3-0.6b
grep base_model_name_or_path post_train/outputs/sft/lora/final/adapter_config.json
nvidia-smi
```

**通过标准**:adapter 产物存在;base 路径正确。

---

## 阶段 5:RFT(可选,先确认 base_model 意图)

> ⚠️ **决策点**:`rft.yaml` 的 `base_model_path` 默认是 **Qwen3-8B**。运行前必须确认这符合实验意图。
> 若目标是用 Full SFT 0.6b 做拒绝采样,应复制配置后改为 `post_train/outputs/sft/full/final`。
> 本流程默认按正式配置(8b)给出;若改 0.6b,复制 `rft.yaml` 到新路径并改 `base_model_path`,其余命令不变。

**5a 构造 RFT accepted 数据**(默认 8b rollout):

```bash
source /root/autodl-tmp/.venv/bin/activate
export OMP_NUM_THREADS=1
cd /root/autodl-tmp/post_train/countdown_post_train
export CUDA_VISIBLE_DEVICES=0

python post_train/scripts/sft/build_rft_data.py \
  --config post_train/configs/rft.yaml
```

**5b 训练 RFT**(复用 Full SFT trainer,映射 `rft.yaml` 的 `train` 段):

```bash
python post_train/scripts/sft/train_rft.py \
  --config post_train/configs/rft.yaml
```

**验收**:

```bash
wc -l post_train/data/sft/rft_accepted.jsonl
test -d post_train/outputs/sft/rft/final
test -f post_train/outputs/sft/rft/final/model.safetensors
nvidia-smi
```

---

## 阶段 6:DPO 数据 + 训练

> 数据构建用 8b vLLM 生成 rejected(forced-wrong + 高温 rollout,五类过滤,目标 ~6000 对);
> chosen 来自 `sft_train_8k.jsonl`(阶段 2 产物)。训练从 `sft/full/final` 出发(依赖阶段 3)。

**6a 构造 DPO pairs**:

```bash
source /root/autodl-tmp/.venv/bin/activate
export OMP_NUM_THREADS=1
cd /root/autodl-tmp/post_train/countdown_post_train
export CUDA_VISIBLE_DEVICES=0

python post_train/scripts/dpo/build_dpo_data.py \
  --config post_train/configs/dpo_data.yaml
```

**验收(数据)**:

```bash
python -m json.tool post_train/data/dpo/manifest.json
wc -l post_train/data/dpo/dpo_train.jsonl
# 检查 category 计数;勿为凑数降低 correctness 过滤标准
```

**6b DPO 训练**:

```bash
python post_train/scripts/dpo/train_dpo.py \
  --config post_train/configs/dpo_train.yaml
```

**验收**:

```bash
test -f post_train/outputs/dpo/final/model.safetensors
test -f post_train/outputs/dpo/final/config.json
nvidia-smi
```

---

## 阶段 7:Legacy GRPO(已修复单卡 OOM)

> 0.6b policy + 0.6b vLLM rollout 共享唯一可见 GPU。
> `rollout_gpu_memory_utilization: 0.45` / `rollout_max_model_len: 1024` 已写入 `grpo.yaml`
> (本会话修复:`train_grpo.py` 原未给 rollout 引擎传显存上限,vLLM 默认 0.9 导致 OOM)。
> 初始模型 `sft/full/final`(依赖阶段 3)。默认 `kl_coeff=0.0`,group 4,每 rollout 更新 2 次,
> 每 20 步同步/保存,每 100 步固定评估(读 `eval.yaml` 的 `eval_subset`)。

```bash
source /root/autodl-tmp/.venv/bin/activate
export OMP_NUM_THREADS=1
cd /root/autodl-tmp/post_train/countdown_post_train
export CUDA_VISIBLE_DEVICES=0

# 7.1 启动前清残留(指南 15 节强制;双引擎对残留显存极敏感)
ps -ef | grep -E 'vllm|EngineCore|train_grpo' | grep -v grep || echo "clean"
nvidia-smi

# 7.2 正式训练
python post_train/scripts/grpo/train_grpo.py \
  --config post_train/configs/grpo.yaml
```

**验收**:

```bash
tail -n 5 post_train/outputs/grpo/metrics.jsonl
find post_train/outputs/grpo -maxdepth 2 -type d -name 'checkpoint-*'
test -d post_train/outputs/grpo/final
nvidia-smi   # 确认双引擎(policy + rollout)都退出
```

**通过标准**:`metrics.jsonl` 持续写入(reward / accuracy / format / loss / KL / response length / 截断数 / group reward std);`final/` 存在;无残留进程。

---

## 阶段 8:独立评估矩阵

> 每个 final 模型独立评估,强制 FA2 + `enable_thinking=false`,solver 判定 `<answer>...</answer>`。
> `--output-dir` 各自独立,**勿覆盖**。未训练的阶段跳过并在记录中标记"未训练",**不要记 accuracy 0**。

```bash
source /root/autodl-tmp/.venv/bin/activate
export OMP_NUM_THREADS=1
cd /root/autodl-tmp/post_train/countdown_post_train
export CUDA_VISIBLE_DEVICES=0


# 8.1b base 8b
python post_train/scripts/eval/evaluate_model.py \
  --config post_train/configs/eval_8b.yaml \
  --model-path post_train/model/qwen/qwen3-8b \
  --output-dir post_train/data/eval/base_8b  

# 8.1 base 0.6b
python post_train/scripts/eval/evaluate_model.py \
  --config post_train/configs/eval.yaml \
  --model-path post_train/model/qwen/qwen3-0.6b \
  --output-dir post_train/data/eval/base_0_6b

# 8.2 Full SFT
python post_train/scripts/eval/evaluate_model.py \
  --config post_train/configs/eval.yaml \
  --model-path post_train/outputs/sft/full/final \
  --output-dir post_train/data/eval/sft_full

# 8.3 LoRA adapter(自动从 adapter_config.json 推导 base;远程路径不可用时显式传 --base-model-path)
python post_train/scripts/eval/evaluate_model.py \
  --config post_train/configs/eval.yaml \
  --model-path post_train/outputs/sft/lora/final \
  --base-model-path post_train/model/qwen/qwen3-0.6b \
  --output-dir post_train/data/eval/sft_lora

# 8.4 RFT(若阶段 5 已做)
python post_train/scripts/eval/evaluate_model.py \
  --config post_train/configs/eval.yaml \
  --model-path post_train/outputs/sft/rft/final \
  --output-dir post_train/data/eval/rft

# 8.5 DPO
python post_train/scripts/eval/evaluate_model.py \
  --config post_train/configs/eval.yaml \
  --model-path post_train/outputs/dpo/final \
  --output-dir post_train/data/eval/dpo

# 8.6 GRPO
python post_train/scripts/eval/evaluate_model.py \
  --config post_train/configs/eval.yaml \
  --model-path post_train/outputs/grpo/final \
  --output-dir post_train/data/eval/grpo
```

**输出内容**(每个 output-dir):

- `eval_samples.jsonl`:prompt、完整 response、提取表达式、format、correctness、token 数、截断状态。
- `eval_metrics.json`:`accuracy`、`format_rate`、`valid_expression_rate`、`avg_generated_tokens`、`max_generated_tokens`、`truncated_count`。

**比较口径**:优先看 `accuracy`,其次 `format_rate`、`truncated_count`、`avg_generated_tokens`。**不能只看训练 loss。**

---

## 阶段 9:阶段验收总览

```bash
# Trainer checkpoints
find post_train/outputs -maxdepth 4 -type d -name 'checkpoint-*' | sort

# 各 final 存在性
for d in sft/full sft/lora sft/rft dpo grpo; do
  test -d post_train/outputs/$d/final && echo "OK: $d/final" || echo "MISSING: $d/final"
done

# eval 指标汇总
for m in base_0_6b sft_full sft_lora rft dpo grpo; do
  echo "=== $m ==="
  cat post_train/data/eval/$m/eval_metrics.json 2>/dev/null || echo "(未评估)"
done
```

---

## 关键注意事项

1. **Teacher 是瓶颈**:300 prompt → 20000 accepted,8b 生成耗时长。建议 `tmux` 后台跑,支持中断续跑。这是整个流程最久的一步。
2. **串行单卡**:阶段 1 / 3 / 4 / 5 / 6 / 7 都独占 GPU,必须串行;阶段 2(CPU)可与阶段 1 收尾重叠。
3. **RFT base_model 决策点**(阶段 5 前):`rft.yaml` 默认 8b,需确认是用 8b 还是用 SFT 0.6b final 做拒绝采样。若改 0.6b,复制配置改 `base_model_path`,勿静默修改默认 YAML。
4. **DPO chosen 数据**:`build_dpo_data` 用 `sft_train_8k.jsonl` 的 chosen,rejected 由 8b 生成。若 8 条 chosen 产不出 pair,指南建议**增大输入而非降低 correctness 过滤标准**——正式用全量 8k 不会有此问题。
5. **GRPO 启动前必清残留进程**(双引擎共享单卡,对残留显存极敏感)。
6. **磁盘**:当前约 23GB 可用。8b 已在盘,0.6b 训练产物都很小;teacher 20k JSONL 约几十 MB。够用,阶段间可清理 `/tmp/post_train_smoke`(smoke 产物)释放空间。
7. **W&B**:默认 `report_to: null`,不创建 run。如需监控:先 `pip install wandb`(清华镜像)再 `wandb login`,然后在对应配置设 `report_to: wandb`。离线评估器不上传。
8. **恢复**:Trainer CLI 无公开 `--resume-from-checkpoint`;恢复正式训练前先核对脚本行为与目标输出目录,避免覆盖已有结果。Teacher 可按 ID 续跑(见阶段 1)。

---

## 附:本次会话已做的代码/配置变更(供回顾)

| 文件 | 变更 | 原因 |
|---|---|---|
| `post_train/scripts/grpo/train_grpo.py` | 主流程与 `sync_rollout_model` 的 `VLLMGenerator` 调用从 cfg 读取 `rollout_gpu_memory_utilization` / `rollout_max_model_len` 并传入 | 原调用未传显存上限,vLLM 默认 0.9 在 48GB 单卡与 0.6b policy 共存时 OOM |
| `post_train/configs/grpo.yaml` | 新增 `rollout_gpu_memory_utilization: 0.45` / `rollout_max_model_len: 1024` | 配套上面修复,约束 rollout 引擎显存 |

环境层(上一轮会话):`transformers 5.9.0 → 4.57.1`(与 vllm 0.10.2 在 vLLM 推理路径兼容)、`setuptools → 79.0.1`(满足 vllm `<80`)、`post_train/model/qwen/{qwen3-0.6b,qwen3-8b}` 符号链接指向 `temp/model/Qwen3-*`。pin 文件已同步。

---

## 附:一键评估全部六个模型(`run_all_evals.sh`)

阶段 8 的六条 `evaluate_model.py` 命令已封装进串行脚本 `post_train/scripts/eval/run_all_evals.sh`,依次在 **500 行 held-out test** 上评估 base / SFT-full / SFT-LoRA / RFT / DPO / GRPO 六个模型,每个模型结果写到 `post_train/data/eval/<name>/eval_metrics.json`,全部跑完后汇总打印指标。

> 注意:`evaluate_model.py` 现版本读取 `eval.yaml` 的 `test_data`(500 行 test,已验证与所有训练集零 ID 重叠),**不再**用 `val_eval_50`(50 行验证子集,仅供训练期 callback 周期评估)。推理走批量化(`batch_size: 32`,左 padding),比原串行快约 9×;greedy + 左 padding 有少数样本文本分叉(accuracy 判定 0 翻转),需逐条可复现时加 `--no-batch`。

### 环境准备

```bash
source /root/autodl-tmp/.venv/bin/activate
export OMP_NUM_THREADS=1
cd /root/autodl-tmp/post_train/countdown_post_train
export CUDA_VISIBLE_DEVICES=0
```

### 用法

```bash
# 跑全部六个模型(已评估的自动跳过,可安全重复执行)
bash post_train/scripts/eval/run_all_evals.sh

# 冒烟:每个模型只评估前 50 行,几分钟内出结果
bash post_train/scripts/eval/run_all_evals.sh --smoke

# 强制串行(batch=1),与历史逐条结果完全一致
bash post_train/scripts/eval/run_all_evals.sh --no-batch

# 指定批大小(覆盖 eval.yaml 的 batch_size)
bash post_train/scripts/eval/run_all_evals.sh --batch-size 16

# 只跑部分模型
bash post_train/scripts/eval/run_all_evals.sh --models sft_full,dpo,grpo

# 强制重新评估(忽略已有 eval_metrics.json)
REEVAL=1 bash post_train/scripts/eval/run_all_evals.sh
```

### 选项说明

| 选项 | 作用 |
|---|---|
| `--smoke` | 每个模型仅 `--limit 50`,快速验证流程 |
| `--no-batch` | 传 `--no-batch`,退回串行逐条推理(完全可复现) |
| `--batch-size N` | 覆盖 `eval.yaml` 的 `batch_size`(默认 32) |
| `--models a,b,c` | 只评估指定模型子集,名称见下表 |
| `REEVAL=1` | 环境变量,跳过"已评估则 skip"逻辑,强制重跑 |
| `-h` / `--help` | 打印用法 |

### 六个模型与输出目录

| 名称 | 模型路径 | 输出目录 |
|---|---|---|
| `base_0_6b` | `post_train/model/qwen/qwen3-0.6b` | `post_train/data/eval/base_0_6b` |
| `sft_full` | `post_train/outputs/sft/full/final` | `post_train/data/eval/sft_full` |
| `sft_lora` | `post_train/outputs/sft/lora/final`(显式传 `--base-model-path` 0.6b) | `post_train/data/eval/sft_lora` |
| `rft` | `post_train/outputs/sft/rft/final` | `post_train/data/eval/rft` |
| `dpo` | `post_train/outputs/dpo/final` | `post_train/data/eval/dpo` |
| `grpo` | `post_train/outputs/grpo/final` | `post_train/data/eval/grpo` |

### 行为说明

- **串行独占 GPU**:六模型依次跑,不并行;某模型失败不影响后续模型,退出码汇总(全部成功=0,有失败=1)。
- **模型路径不存在则跳过**该模型并在汇总标记失败(例如 RFT 阶段未做时 `rft/final` 缺失)。
- **LoRA** 显式传 `--base-model-path post_train/model/qwen/qwen3-0.6b`,避免 `adapter_config.json` 中绝对路径在符号链接/远程场景解析问题。
- **汇总输出**:脚本末尾打印每个模型的 `eval_metrics.json`(accuracy / format_rate / valid_expression_rate / avg_generated_tokens / truncated_count)。

### 后台运行(推荐)

六模型 × 500 行 batch 推理,单卡约 4–6 分钟;串行 `--no-batch` 约 30–40 分钟。长跑建议 `tmux`:

```bash
tmux new -s eval
bash post_train/scripts/eval/run_all_evals.sh 2>&1 | tee post_train/data/eval/run_all.log
# Ctrl+B D 脱离; tmux a -t eval 重连
```

### 查看结果

```bash
# 单个模型
cat post_train/data/eval/sft_full/eval_metrics.json

# 汇总(脚本末尾会自动打印,也可手动)
for m in base_0_6b sft_full sft_lora rft dpo grpo; do
  echo "=== $m ==="; cat post_train/data/eval/$m/eval_metrics.json 2>/dev/null || echo "(未评估)"
done

# 逐条样本
head -1 post_train/data/eval/sft_full/eval_samples.jsonl | python3 -m json.tool
```
