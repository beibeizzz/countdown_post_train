# Countdown Post-Training

> 本 README 描述 **post_train v1**(单 GPU legacy 流程)。仓库内另有 `post_train_v2/`(双 GPU 分布式),暂不在本说明覆盖范围内。

Countdown(24 点式)算术推理任务的后训练实验工程:在 Qwen3 上,通过 Teacher 蒸馏 → SFT → RFT/DPO/GRPO 的渐进式后训练链路,让小模型(0.6B)学会用给定数字、四则运算凑出目标值,并以可验证的 `<answer> 表达式 </answer>` 格式作答。

---

## 1. 背景

大模型在数学推理上能力强,但存在两个工程痛点:

1. **输出格式不可验证**:模型倾向于写自然语言多步推导(`A=19, 19+17=36`),难以程序化判定对错。
2. **小模型能力不足**:0.6B base 模型虽能套 `<answer>` 标签,但生成的表达式非法(多步等式 / 未用给定数字),准确率为 0;8B base 又因长篇生成未对齐到任务格式而被截断。

本项目的目标是:**用后训练把输出规范对齐到「单一合法算术表达式 + answer 标签」,并让 0.6B 在该格式下达到可用准确率**,同时验证 RFT / DPO / GRPO 三种范式相对纯 SFT 的增益。

---

## 2. 任务定义

给定一组数字和一个目标值,要求模型用**每个数字恰好一次**、仅用 `+ - * /` 和括号,构造一个等于目标值的表达式。

**Prompt 示例**:

```
Using the numbers [79, 17, 60], create an equation that equals 36.
Use each number exactly once. Only use +, -, *, / and parentheses.
Do not use any other numbers. Keep the response concise.
Finally return <answer> equation </answer>.
```

**期望输出**: `<answer> (79 - 60) + 17 </answer>`

**判定逻辑**(`post_train/src/countdown/validation.py`):
- 提取 `<answer>...</answer>` 内的表达式;
- `ast.parse(mode="eval")` 解析为单一算术表达式(拒绝多步等式、自然语言);
- 用 `Fraction` 精确求值,校验:**所用数字多重集 == 给定数字** 且 **值 == target**。

---

## 3. 思路

渐进式后训练,每一阶段解决一个具体短板:

| 阶段 | 解决的问题 | 方法 |
|---|---|---|
| Teacher 蒸馏 | 缺高质量、格式合规的训练数据 | 用 8B 模型 rollout,过滤出 20000 条**正确且格式合规**的样本 |
| SFT(全参 / LoRA) | 0.6B 不会输出合法单表达式 | 在 teacher 数据上监督学习对齐输出格式 |
| RFT(拒绝采样微调) | SFT 数据多样性不足 | 对 SFT 模型/teacher 多次采样,只保留**正确**的 rollout 再训 |
| DPO | 模型会答对但易产出错误表达式 | 构造 (chosen 正确, rejected 错误) 对,偏好优化纠错 |
| GRPO | 纯模仿缺乏探索、难突破准确率上限 | 用规则奖励做组相对策略优化,鼓励正确且简洁的生成 |

**核心观察(来自 eval)**:
- 0.6B base accuracy=0 不是「不会算」,而是输出多步等式不被验证器接受 → SFT 对齐格式后即大幅提升。
- 8B base(0.246)反不如训练后的 0.6B(~0.38),因其长篇生成未对齐任务格式、被截断 → 说明**格式对齐比堆参数更关键**。

---

## 4. 工程实现

### 4.1 目录结构

```
post_train/
├── configs/            # 各阶段 YAML 配置
├── scripts/
│   ├── data/           # teacher pool / SFT-GRPO splits / DPO pairs 数据构建
│   ├── sft/            # full / lora / rft 训练 + RFT 数据构建
│   ├── dpo/            # DPO 数据构建 + 训练
│   ├── grpo/           # GRPO 训练
│   └── eval/           # evaluate_model.py + run_all_evals.sh
├── src/countdown/      # 核心库: config / generation(vLLM) / validation / eval / io
├── data/               # 数据与 eval 产物(大文件见 .gitignore)
└── outputs/            # 训练产物(权重见 .gitignore)
```

### 4.2 关键模块

- **`generation.py` — `VLLMGenerator`**:封装 vLLM `llm.chat` 批量推理,数据构建(teacher/RFT/DPO)与 GRPO rollout 共用,内部连续批处理。
- **`validation.py`**:基于 `ast` + `Fraction` 的硬规则验证器,贯穿数据过滤、奖励计算、eval 评分,保证「训练 reward」与「评测 accuracy」口径一致。
- **`eval.py`**:`score_generation` / `aggregate_eval_rows`,产出 accuracy / format_rate / valid_expression_rate / token 数 / 截断数。


---

## 5. 使用流程

> 环境:Python 3.12 / PyTorch 2.8 / vLLM 0.10.2 / Transformers 4.57.1,单卡 RTX 4090(48GB)。详细环境与验收见 [`EXPERIMENT_RUNBOOK.md`](EXPERIMENT_RUNBOOK.md)。

### 5.0 通用前置

```bash
source /root/autodl-tmp/.venv/bin/activate
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0
cd /root/autodl-tmp/post_train/countdown_post_train
# 平台 git/网络加速(按需)
source /etc/network_turbo
```

### 5.1 Teacher 蒸馏(8B,目标 20000 条正确样本)

```bash
python post_train/scripts/data/build_teacher_pool.py \
  --config post_train/configs/teacher_rollout.yaml
```

### 5.2 切分 SFT 8k / GRPO 4k / RFT 2k

```bash
python post_train/scripts/data/build_sft_splits.py \
  --config post_train/configs/data_build.yaml
```

### 5.3 Full SFT(0.6B)

```bash
python post_train/scripts/sft/train_full.py --config post_train/configs/sft_full.yaml
```

### 5.4 LoRA SFT(0.6B,产物为 adapter)

```bash
python post_train/scripts/sft/train_lora.py --config post_train/configs/sft_lora.yaml
```

### 5.5 RFT

```bash
# 5a 构建 RFT accepted 数据(默认 8B rollout,2k prompt × 4 采样)
python post_train/scripts/sft/build_rft_data.py --config post_train/configs/rft.yaml
# 5b 训练(复用 SFT trainer)
python post_train/scripts/sft/train_rft.py --config post_train/configs/rft.yaml
```

### 5.6 DPO

```bash
# 6a 构建 (chosen, rejected) 对
python post_train/scripts/dpo/build_dpo_data.py --config post_train/configs/dpo_data.yaml
# 6b 训练(从 sft/full/final 出发)
python post_train/scripts/dpo/train_dpo.py --config post_train/configs/dpo_train.yaml
```

### 5.7 GRPO(0.6B policy + 0.6B vLLM rollout 共享单卡)

```bash
python post_train/scripts/grpo/train_grpo.py --config post_train/configs/grpo.yaml
```

### 5.8 评估矩阵

```bash
# 一键评估 base / sft_full / sft_lora / rft / dpo / grpo 六个模型
bash post_train/scripts/eval/run_all_evals.sh
```

每个模型在 500 行 held-out test 上评估,产出 `post_train/data/eval/<name>/eval_metrics.json`,核心指标为 `accuracy`(其次 `format_rate` / `truncated_count` / `avg_generated_tokens`)。

---

