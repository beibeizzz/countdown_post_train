# countdown_post_train / post_train 一天阅读指南

基于本地仓库审查结果编写，仓库分支 `main`，HEAD `63665a68bc33ef97da0da85793382710cf3e91ff`。重点范围是用户 URL 指向的 legacy 单 GPU 子项目 `post_train/`；`post_train_v2/` 是双 GPU、verl、Manifest V2 的重构版本，作为对比和后续扩展阅读。

## 1. 项目一句话概括

`post_train` 是一个面向 Countdown 算术推理任务的 Qwen3 后训练实验工程：用 Qwen3-8B teacher 生成可验证答案，再将 Qwen3-0.6B 经过 Full SFT、LoRA SFT、RFT、DPO、GRPO 等阶段提升到能输出 `<answer>...</answer>` 内合法表达式的模型。

## 2. 项目目标与核心问题

核心问题不是单纯“模型会不会算”，而是小模型能否在严格格式下输出单个、可程序验证的算术表达式。

- 输入：`datasets/raw_train.parquet` 和 `datasets/raw_test.json`，字段是题目 ID、`nums`/`numbers`、`target`。
- 输出：训练数据 JSONL、teacher accepted/rejected pool、SFT/RFT/DPO/GRPO 模型或 adapter、评估样本与指标。
- 主要实验对象：Qwen3-0.6B target model；Qwen3-8B teacher/generation model。
- 判分标准：`post_train/src/countdown/validation.py` 用 `<answer>` 抽取、AST allow-list、`Fraction` 精确计算、数字多重集匹配、目标值相等来判断。
- 当前评估结论来自已提交的 `post_train/data/eval/*/eval_metrics.json`：base 0.6B accuracy 0.0，base 8B 0.246，SFT full 0.352，LoRA 0.384，RFT 0.376，DPO 0.372，GRPO 0.436。

## 3. 与 agentflow 的关系

本仓库没有直接 import 或依赖 `agentflow`，`rg "agentflow|AgentFlow"` 只发现文档引用。联系主要在工程习惯层面：Python 包目录、脚本入口、YAML 配置、README/runbook、pytest、manifest、可复现实验记录。区别是 `agentflow` 偏应用/工作流编排，本项目是模型后训练实验系统，新增核心知识是 Transformers/TRL/PEFT/vLLM/GRPO/reward/evaluation/checkpoint。

注意：`post_train_v2/environment.md` 明确说明不要复用 AgentFlow 的 `.venv`、`PYTHONPATH`、`CUDA_HOME`、`LD_LIBRARY_PATH`，因为 CUDA、PyTorch、vLLM、xFormers、NumPy 等二进制约束不兼容。

## 4. 项目整体架构

根目录：

- `datasets/`：原始数据，`raw_train.parquet`、`raw_test.json`。
- `post_train/`：legacy 单 GPU主实现，本指南重点。
- `post_train_v2/`：双 GPU DDP + verl 重构版，保留同一任务语义但重做分布式、artifact、pipeline。
- `README.md`、`EXPERIMENT_RUNBOOK.md`：legacy 端到端流程和远程运行指南。
- `requirements.txt`：legacy 运行栈，Python 3.12、PyTorch 2.8、Transformers 4.57.1、TRL 1.5.1、PEFT 0.19.1、vLLM 0.10.2、Flash Attention 2.8.3。

`post_train/` 内部：

- `configs/`：每个阶段一个 YAML。
- `src/countdown/`：任务语义与通用工具，最值得先读。
- `scripts/data/`：source、teacher pool、SFT/GRPO/RFT split。
- `scripts/sft/`：Full SFT、LoRA SFT、RFT 数据与训练。
- `scripts/dpo/`：DPO pair 构造和 TRL DPO 训练。
- `scripts/grpo/`：legacy 自写 GRPO 训练环。
- `scripts/eval/`：独立评估入口和一键评估矩阵。
- `data/`、`outputs/`：部分小产物和评估结果已提交；大数据和权重多数未提交。
- `tests/`：纯 Python 单测，GPU/vLLM/full training smoke 留给远程环境。

## 5. 完整数据与训练链路

真实主链路如下：

```text
datasets/raw_train.parquet, raw_test.json
  -> build_source.py: solver 生成 gold_expr/prompt/bucket/source pool
  -> build_teacher_pool.py: Qwen3-8B vLLM rollout
  -> teacher_accepted_20k.jsonl / teacher_rejected.jsonl
  -> build_sft_splits.py: SFT 8k, GRPO 4k, RFT prompts 2k
  -> train_full.py / train_lora.py
  -> build_rft_data.py -> train_rft.py
  -> build_dpo_data.py -> train_dpo.py
  -> train_grpo.py
  -> evaluate_model.py 或 run_all_evals.sh
  -> eval_samples.jsonl / eval_metrics.json
```

当前本地仓库只包含部分产物：`processed` 里有 val/test/eval subset，但没有 `train_pool.jsonl`；`teacher_rollouts` 只有 manifest；`data/sft` 只有 `sft_train_2k.jsonl`；`data/grpo` 有 `grpo_train_4k.jsonl`；`data/dpo` 只有 manifest；模型权重目录 `post_train/model/` 不存在。

## 6. 技术栈分析

| 技术 | 在项目中解决的问题 | 关键文件/函数 | 流程位置 | 阅读深度 | 相对 agentflow 新增 |
|---|---|---|---|---|---|
| Python 工程结构 | 分离任务语义、入口脚本、配置、产物 | `src/countdown/*`, `scripts/*` | 全流程 | 通读即可 | 部分复用 |
| YAML/argparse | 阶段参数和 CLI 入口 | `configs/*.yaml`, `load_yaml_config`, `parse_args` | 每阶段入口 | 精读路径解析 | 部分复用 |
| pandas/pyarrow | parquet 原始训练题库读取 | `build_source.py` | source build | 会跑即可 | 新增一点 |
| solver/AST/Fraction | 构造 gold expression 和统一判分 | `solver.py`, `validation.py` | 数据、reward、eval | 必须精读 | 新增 |
| Bucketing/sampling | 难度分桶、固定 seed 分层抽样 | `bucketing.py`, `sampling.py` | source/split | 精读 | 部分新增 |
| Transformers | 模型加载、SFT、生成评估 | `train_full.py`, `evaluate_model.py` | SFT/eval | 必须理解 | 新增 |
| Chat template | Qwen3 对话输入、thinking off | `apply_chat_template`, `apply_chat_template_compat` | 训练/推理 | 精读 | 新增 |
| SFT loss mask | prompt mask 为 -100，只训练 response | `encode_prompt_response` | SFT/RFT | 必须精读 | 新增 |
| PEFT/LoRA | adapter 训练，低显存微调 | `train_lora.py` | LoRA SFT | 精读 target modules | 新增 |
| TRL DPO | preference pair 优化 | `train_dpo.py` | DPO | 理解输入格式 | 新增 |
| vLLM | teacher/RFT/DPO/GRPO rollout | `VLLMGenerator` | 数据生成/RL | 理解 batch/metadata | 新增 |
| 自写 GRPO | group rollout、reward、advantage、policy loss | `train_grpo.py` | RL 阶段 | 必须精读 | 新增 |
| reward function | 格式奖励、正确性奖励、长度惩罚 | `compute_rewards` | GRPO | 必须精读 | 新增 |
| 混合精度/FA2 | BF16 + Flash Attention 2 | `load_model_and_tokenizer` | 训练/eval | 知道约束 | 新增 |
| checkpoint | final/export、periodic eval | `save_model`, `save_checkpoint` | 训练输出 | 理解产物形式 | 新增 |
| W&B | 可选日志 | `wandb_utils.py` | 训练/GRPO | 快速浏览 | 部分复用 |
| 测试 | 任务语义、训练参数、入口兼容性 | `post_train/tests/*` | 验证 | 按模块读 | 部分复用 |

## 7. 前置知识清单

前置知识不计入一天阅读时间。建议总补充 5.5 到 7 小时。

| 前置知识 | 为什么需要 | 对应文件 | 掌握程度 | 建议练习 | 时间 | agentflow 覆盖 |
|---|---|---|---|---|---|---|
| Transformers causal LM | 理解模型加载、Trainer、generate | `train_full.py`, `evaluate_model.py` | 能解释输入/labels/generate | 加载一个 tiny causal LM 做一次 generate | 45m | 否 |
| tokenizer/chat template | Qwen3 prompt/assistant 拼接靠它 | `generation.py`, `train_full.py` | 能打印 rendered prompt | 打印一条 SFT prompt/full text | 30m | 否 |
| SFT loss mask | 项目只训练 response tokens | `encode_prompt_response` | 能解释 -100 | 手动看 labels 前后段 | 35m | 否 |
| Full FT vs LoRA/PEFT | 区分完整模型和 adapter | `train_lora.py`, eval loader | 能解释 adapter 加载 | 查看 `adapter_config.json` 预期字段 | 35m | 否 |
| TRL DPO | preference pair 格式和 beta | `train_dpo.py` | 能解释 prompt/chosen/rejected | 构造 2 条 pair 转 Dataset | 45m | 否 |
| GRPO 基本思想 | group-relative reward/advantage | `train_grpo.py` | 能解释 group std 和 advantage | 用 5 个 reward 手算 advantage | 50m | 否 |
| rule reward | 不用 learned reward model | `validation.py`, `compute_rewards` | 能追踪错误类别 | 测 5 个表达式错误类别 | 40m | 否 |
| vLLM batch rollout | teacher/RFT/DPO/GRPO 都依赖 | `generation.py` | 知道 SamplingParams 和 metadata | mock 或少量 prompts 生成 | 40m | 否 |
| GPU 显存/batch/accumulation | 运行训练避免 OOM | `configs/*.yaml` | 会算 effective batch | 改 smoke config 跑 2 step | 40m | 否 |
| bf16/fp16/Flash Attention | 当前环境硬约束 | `requirements.txt`, loaders | 能定位安装问题 | 读 `EXPERIMENT_RUNBOOK.md` 环境段 | 35m | 否 |
| JSON/JSONL/parquet | 数据格式贯穿全链路 | `build_source.py`, `io.py` | 能抽样检查 schema | head 一行并 json pretty | 25m | 部分 |
| YAML/CLI 参数传递 | 每阶段入口都靠 config | `config.py`, scripts | 能追踪路径解析 | 改 copy 后 smoke | 25m | 是 |
| checkpoint/adapter/merge | 评估不同产物形式 | `evaluate_model.py` | 能区分 full vs LoRA | 读 loader 分支 | 30m | 否 |
| 训练日志/指标 | 结果解释不能只看 loss | `eval.py`, eval outputs | 能解释 accuracy/format/truncation | 汇总 eval_metrics | 30m | 部分 |

## 8. 一天阅读总时长

正式阅读建议 9.5 小时，不含前置知识和环境安装。若要实际跑 GPU smoke，再额外预留 1 到 2 小时；完整 teacher 或训练不适合塞进一天阅读。

## 9. 一天分阶段阅读计划

### 阶段 1：09:00-10:00 建立全局认识

- 目标：明确项目定位、目录结构、主链路、产物缺失状态。
- 精读：`README.md`, `post_train/README.md`, `EXPERIMENT_RUNBOOK.md` 的 0-8 阶段。
- 通读：`post_train/configs/README.md`, `post_train/scripts/*/README.md`。
- 快速浏览：`post_train_v2/README.md`, `post_train_v2/analysis.md`。
- 跳过：`docs/superpowers/plans/*` 的任务细节。
- 命令：`rg --files post_train`, `git status --short`, `Get-ChildItem post_train/data -Recurse -Depth 2`。
- 最小实验：读 3 个 `eval_metrics.json`，写一张 base/SFT/GRPO 对比小表。
- 自检：为什么 0.6B base `format_rate` 高但 accuracy 0？为什么 8B base 被 0.6B post-trained 模型超过？
- 笔记产出：一页主流程图和关键目录表。

### 阶段 2：10:00-12:00 数据流水线

- 精读顺序：`prompts.py` -> `solver.py` -> `validation.py` -> `bucketing.py` -> `sampling.py` -> `build_source.py` -> `build_teacher_pool.py` -> `build_sft_splits.py` -> `build_rft_data.py` -> `build_dpo_data.py`。
- 重点函数：`build_solution_prompt`, `solve_countdown`, `validate_countdown_expression`, `assign_bucket`, `stratified_sample`, `build_solved_record`, `build_teacher_payload`, `process_teacher_responses`, `select_dpo_pairs`。
- 命令：`python post_train/scripts/data/build_source.py --config post_train/configs/data_build.yaml --limit 20`，建议先复制 config 并改 output 到临时目录，避免覆盖正式产物。
- 最小实验：手动验证 `<answer>(79-60)+17</answer>`；打印 `sft_train_2k.jsonl` 第一行；检查 `data/dpo/manifest.json` 的 category counts。
- 自检：source record 和 accepted SFT record 差哪些字段？DPO 为什么实际只有 1889 pairs？
- 笔记产出：四种数据 schema：source、accepted、DPO pair、GRPO prompt。

### 阶段 3：13:00-14:30 监督训练流程

- 精读：`train_full.py`, `train_lora.py`, `train_rft.py`, `sft_full.yaml`, `sft_lora.yaml`, `rft.yaml`。
- 重点函数：`normalize_sft_config`, `encode_prompt_response`, `DataCollatorForCausalSFT`, `load_model_and_tokenizer`, `build_training_arguments`, `run_sft_training`, `resolve_lora_target_modules`, `apply_lora`。
- 命令：`python post_train/scripts/sft/train_full.py --config <smoke-copy.yaml> --max-steps 2`。
- 最小实验：在不训练的情况下用 mock tokenizer 或阅读测试 `test_train_full.py`，确认 prompt label 为 -100。
- 自检：RFT 为什么复用 Full SFT trainer？LoRA final 和 Full SFT final 在评估加载上有什么差别？
- 笔记产出：Full/LoRA/RFT 三者输入、输出、checkpoint 形式对比。

### 阶段 4：14:30-16:30 偏好优化与强化学习

- 精读：`build_dpo_data.py`, `train_dpo.py`, `train_grpo.py`, `dpo_data.yaml`, `dpo_train.yaml`, `grpo.yaml`。
- 重点函数：`classify_rejected`, `build_route_requests`, `build_candidates`, `select_dpo_pairs`, `format_dpo_record_for_trl`, `build_dpo_trainer`, `compute_rewards`, `group_relative_advantages`, `rollout_batch`, `sequence_policy_loss`, `sync_rollout_model`, `train_grpo`。
- 命令：`python post_train/scripts/dpo/build_dpo_data.py --config <smoke-copy.yaml> --limit 8`；`python post_train/scripts/grpo/train_grpo.py --config <smoke-copy.yaml> --max-steps 2`。
- 最小实验：对 rewards `[1.5, 0.5, 0.5, 1.5, 0]` 手算 group advantage；用 `compute_rewards` 验证格式、正确性、长度惩罚。
- 自检：legacy GRPO 的 `clip_eps` 实际裁剪的是 advantage，不是 policy ratio；为什么 `kl_coeff` 只能为 0？
- 笔记产出：DPO pair 选择规则和 GRPO 训练环伪代码。

### 阶段 5：16:30-17:45 推理、评估与结果

- 精读：`evaluate_model.py`, `eval.py`, `run_all_evals.sh`, `eval.yaml`, `eval_8b.yaml`。
- 重点函数：`load_model_and_tokenizer`, `generation_kwargs`, `generate_one`, `evaluate_rows_batched`, `score_generation`, `aggregate_eval_rows`。
- 命令：`bash post_train/scripts/eval/run_all_evals.sh --smoke --models base_0_6b,sft_full`，需要模型路径存在。
- 最小实验：读取 `post_train/data/eval/*/eval_metrics.json`，按 accuracy 排序。
- 自检：训练中的 periodic eval 用 50 条，独立 eval 用 500 条 test，这两个路径在哪里分叉？
- 笔记产出：指标解释和已提交结果表。

### 阶段 6：18:00-19:30 调用链串联与总结

- 精读：本日笔记、`post_train_v2/analysis.md`、`post_train_v2/migration_plan.md` 中 legacy 风险段。
- 追踪：任选一个真实命令，从 YAML 到脚本入口、数据加载、模型加载、Trainer/rollout、输出文件。
- 命令：`python -m pytest post_train/tests/test_validation.py -q`，再选择数据/GRPO/DPO相关单测。
- 最小实验：写 15 行总结，说明如何新增一种 reward 或修改 DPO pair 选择。
- 自检：能否白板讲出 Teacher -> DPO -> GRPO 的数据和模型流？
- 笔记产出：面试讲稿、风险清单、下一步实验清单。

## 10. 文件级阅读顺序

必须精读：

| 文件 | 职责 | 上游输入 | 下游输出 | 重点问题 | 是否运行 | agentflow 相似性 |
|---|---|---|---|---|---|---|
| `post_train/src/countdown/prompts.py` | 标准 prompt 和 DPO forced-wrong prompt | numbers/target/chosen | prompt/messages | prompt 是否约束 answer 标签 | 否 | 通用 prompt 工具相似 |
| `post_train/src/countdown/validation.py` | 统一判分 | response/expression | `ValidationResult` | AST 支持哪些语法，错误类别如何定义 | 是 | 新增 |
| `post_train/src/countdown/solver.py` | source gold_expr 求解 | raw numbers/target | expression/meta | 搜索如何处理重复数字和 Fraction | 是 | 新增 |
| `post_train/src/countdown/generation.py` | vLLM 和 chat template 封装 | prompts/config | texts/metadata | `enable_thinking` 和 token_count 如何传递 | GPU | 新增 |
| `post_train/scripts/data/build_source.py` | raw -> normalized source | parquet/json | processed JSONL | val/test 切分和 bucket | CPU | 数据脚本相似 |
| `post_train/scripts/data/build_teacher_pool.py` | teacher rollout/filter/resume | train_pool + 8B | accepted/rejected/manifest | lock、断点、V2 状态拒绝 | GPU | 新增 |
| `post_train/scripts/data/build_sft_splits.py` | accepted -> SFT/GRPO/RFT splits | accepted pool | sft/grpo/rft JSONL | 三个 split 是否互斥？实际是独立抽样 | CPU | 数据脚本相似 |
| `post_train/scripts/sft/train_full.py` | Full SFT | SFT/RFT rows + 0.6B | full model | response-only label mask | GPU | 新增 |
| `post_train/scripts/sft/train_lora.py` | LoRA SFT | SFT rows + base | adapter | target modules auto 解析 | GPU | 新增 |
| `post_train/scripts/sft/build_rft_data.py` | RFT rollout/filter | 2k prompts + rollout model | rft accepted/rejected | 默认 base 是 8B，不是 SFT 0.6B | GPU | 新增 |
| `post_train/scripts/dpo/build_dpo_data.py` | DPO pair 构造 | chosen SFT + 8B candidates | pairs/candidates/manifest | rejected category 优先级 | GPU | 新增 |
| `post_train/scripts/dpo/train_dpo.py` | TRL DPO 训练 | prompt/chosen/rejected | full model | TRL record 如何格式化 | GPU | 新增 |
| `post_train/scripts/grpo/train_grpo.py` | legacy GRPO | GRPO prompts + SFT full | metrics/checkpoints/final | reward、advantage、policy loss、vLLM sync | GPU | 新增 |
| `post_train/scripts/eval/evaluate_model.py` | 独立评估 | model + test data | samples/metrics | LoRA loader、batched left padding | GPU | 新增 |
| `post_train/src/countdown/eval.py` | scoring 聚合 | generation rows | metrics | accuracy/format/truncation | 是 | 部分 |

需要通读：

- `post_train/src/countdown/config.py`, `io.py`, `wandb_utils.py`, `output_lock.py`：工程基础设施。
- `post_train/configs/*.yaml`：每个阶段的默认参数和路径。
- `post_train/scripts/*/README.md`：运行约束和 smoke 提醒。
- `post_train/tests/test_*.py`：按模块看行为契约。

快速浏览：

- `post_train/docs/remote_training_guide.md`、`EXPERIMENT_RUNBOOK.md`：远程运行环境和验收。
- `post_train/data/eval/*/eval_metrics.json`：结果表。
- `post_train/outputs/*/eval/*/eval_metrics.json`：训练中周期评估。

暂时跳过：

- `post_train/docs/superpowers/plans/*` 的实现计划细节。
- 未提交的大权重和缓存产物。

`post_train_v2/` 阅读顺序：

1. `post_train_v2/README.md`：确认 V2 目标。
2. `post_train_v2/analysis.md`：legacy 行为分析，尤其 GRPO 风险。
3. `post_train_v2/environment.md`：为什么不能复用 AgentFlow 环境。
4. `post_train_v2/src/countdown/*`, `src/data/schema.py`, `src/artifacts/manifest.py`, `src/generation/parallel_vllm.py`, `src/training/*`, `verl/data/conversion.py`, `verl/rewards/countdown_reward.py`, `verl/launch/train_grpo.py`：只在完成 legacy 后阅读。

## 11. 核心调用链

### 调用链 A：原始题目到 source pool

- 命令：`python post_train/scripts/data/build_source.py --config post_train/configs/data_build.yaml`
- 配置：`data_build.yaml`。
- 入口：`build_source.py:main`。
- 流程：读取 raw parquet/json -> `get_numbers` -> `solve_countdown` -> `build_solution_prompt` -> `assign_bucket` -> `stratified_sample` -> 写 `processed/*.jsonl` 和 manifest。
- 产物：`source_all.jsonl`, `train_pool.jsonl`, `val_200.jsonl`, `val_eval_50.jsonl`, `test_with_solver_answers.jsonl`, `unsolved_train.jsonl`。
- 常见错误：parquet/pyarrow 缺失；raw 字段是 `nums` 还是 `numbers`；输出目录覆盖；本地未提交 `train_pool.jsonl`。

### 调用链 B：teacher rollout 到 accepted pool

- 命令：`python post_train/scripts/data/build_teacher_pool.py --config post_train/configs/teacher_rollout.yaml`
- 配置：`teacher_rollout.yaml`，8B、vLLM、batch 128、temperature 0.3、max_new_tokens 1024。
- 入口：`run` -> `_execute_locked`。
- 流程：读取 `train_pool.jsonl` -> 初始化 `VLLMGenerator` -> batch chat -> `build_teacher_payload` -> `validate_countdown_response` -> accepted/rejected -> atomic write。
- 产物：`teacher_accepted_20k.jsonl`, `teacher_rejected.jsonl`, `manifest.json`。
- 常见错误：模型路径缺失；vLLM/CUDA/FA2 不兼容；stale lock；`train_pool.jsonl` 未生成；accepted 不足。

### 调用链 C：SFT 配置到模型训练

- 命令：`python post_train/scripts/sft/train_full.py --config post_train/configs/sft_full.yaml`
- 配置：`sft_full.yaml`，0.6B、SFT 8k、max_seq_len 1024、BF16、gradient checkpointing。
- 入口：`run_sft_training`。
- 流程：加载模型/tokenizer -> 读 SFT JSONL -> `encode_prompt_response` 渲染 prompt/full -> prompt labels 置 -100 -> `DataCollatorForCausalSFT` pad -> Transformers `Trainer.train()` -> `final/`。
- 产物：`post_train/outputs/sft/full/final` 完整 HF 模型。
- 常见错误：`sft_train_8k.jsonl` 未生成；prompt tokens 不是 full tokens 前缀；FA2 未安装；显存不足。

### 调用链 D：preference pair 到 DPO

- 命令：`build_dpo_data.py` 后接 `train_dpo.py`。
- 配置：`dpo_data.yaml`, `dpo_train.yaml`。
- 数据流：SFT chosen -> forced-wrong prompt/high-temp prompt -> 8B 生成 rejected candidates -> `classify_rejected` -> `select_dpo_pairs` -> TRL Dataset -> `DPOTrainer`。
- 产物：`data/dpo/dpo_train.jsonl`, `dpo_rejected_candidates.jsonl`, `outputs/dpo/final`。
- 常见错误：unexpected_correct 过多导致 pair shortfall；DPO JSONL 缺 `prompt/chosen/rejected` 字符串；TRL 版本参数不兼容。

### 调用链 E：GRPO 配置到 rollout、reward、更新

- 命令：`python post_train/scripts/grpo/train_grpo.py --config post_train/configs/grpo.yaml`
- 配置：`grpo.yaml`，从 SFT full final 出发，train data 为 GRPO 4k，batch 4，group 5，max_steps 500，KL 0。
- 流程：加载 policy model -> 启动 vLLM rollout model -> `rollout_batch` 每个 prompt 采 group_size 条 -> `compute_rewards` -> `group_relative_advantages` -> `encode_policy_example` -> `sequence_policy_loss` -> optimizer/scheduler -> metrics -> sync/eval/save。
- 产物：`outputs/grpo/metrics.jsonl`, `checkpoint-*`, `eval/step_*`, `final/`。
- 常见错误：单卡 vLLM + policy OOM；stale vLLM 进程；所有 completion 为空；非零 `kl_coeff` 直接报错；sync reload 失败可能继续用旧 generator。

### 调用链 F：checkpoint 到推理评估

- 命令：`python post_train/scripts/eval/evaluate_model.py --config post_train/configs/eval.yaml --model-path ... --output-dir ...`
- 配置：`eval.yaml` 或 `eval_8b.yaml`。
- 流程：加载 full 或 LoRA adapter -> batched left-padding generate -> `score_generation` -> `aggregate_eval_rows`。
- 产物：`eval_samples.jsonl`, `eval_metrics.json`。
- 常见错误：LoRA base path 不可用；batch 太大；模型目录不存在；训练中 eval subset 和独立 test eval 混淆。

## 12. 各训练方法对比

| 方法 | 输入数据 | teacher | preference | online rollout | reward | 优化目标 | 成本/显存 | 输出形式 | 项目作用 | 代码位置 |
|---|---|---|---|---|---|---|---|---|---|---|
| Base 0.6B/8B | test prompts | 否 | 否 | 推理 | eval rule | 无训练 | 低/中 | 原始模型 | baseline | `evaluate_model.py` |
| Full SFT | `sft_train_8k.jsonl` | 数据来自 8B | 否 | 否 | eval rule | response CE loss | 中，完整参数 | full HF model | 格式对齐主基座 | `train_full.py` |
| LoRA SFT | 同 SFT | 数据来自 8B | 否 | 否 | eval rule | adapter CE loss | 低 | PEFT adapter | 低成本 SFT 对照 | `train_lora.py` |
| RFT | `sft_train_2k` 多采样 accepted | 默认 8B rollout | 否 | 离线 rollout | filter rule | accepted response CE loss | 中 | full HF model | 增加正确样本多样性 | `build_rft_data.py`, `train_rft.py` |
| DPO | `prompt/chosen/rejected` | 8B 生成 rejected | 是 | 离线 rollout | 过滤用 rule | DPO preference loss | 中偏高 | full HF model | 让模型偏好正确表达式 | `build_dpo_data.py`, `train_dpo.py` |
| GRPO | `grpo_train_4k.jsonl` | 初始来自 SFT full | 否 | 在线 vLLM rollout | format + answer + length | group-relative policy gradient | 最高，单卡双模型易 OOM | full HF model | 当前最高 accuracy | `train_grpo.py` |

## 13. 最小可运行实验

所有会写正式产物的实验，都先复制 YAML 并把输出改到临时目录。

| 实验 | 目的 | 命令 | 输入/输出 | 观察点 | GPU |
|---|---|---|---|---|---|
| 验证表达式 | 理解 validator | `python -m pytest post_train/tests/test_validation.py -q` | 表达式 -> error/ok | missing tag、wrong value、number mismatch | 否 |
| 小规模 source | 理解 raw->source | `python post_train/scripts/data/build_source.py --config <tmp.yaml> --limit 20` | raw -> processed | bucket、gold_expr、prompt | 否 |
| 查看样本 | 理解 schema | `Get-Content post_train/data/sft/sft_train_2k.jsonl -TotalCount 1` | JSONL row | response、teacher_expr、validation | 否 |
| DPO 选择单测 | 理解 pair selection | `python -m pytest post_train/tests/test_dpo_data_builder.py -q` | synthetic candidates | category priority | 否 |
| 打印 SFT label mask | 理解 loss | `python -m pytest post_train/tests/test_train_full.py::test_encode_masks_prompt_labels_and_keeps_response_trainable -q` | fake tokenizer | -100 mask | 否 |
| RFT 少量 rollout | 看 accepted/rejected | `python post_train/scripts/sft/build_rft_data.py --config <tmp.yaml> --limit 4` | 4 prompts * samples | accepted rate | 是 |
| DPO 少量生成 | 看 candidates | `python post_train/scripts/dpo/build_dpo_data.py --config <tmp.yaml> --limit 8` | chosen rows | unexpected_correct、wrong_value | 是 |
| SFT 2 step | 验证训练入口 | `python post_train/scripts/sft/train_full.py --config <tmp.yaml> --max-steps 2` | small SFT | loss、final dir | 是 |
| GRPO 2 step | 验证 RL loop | `python post_train/scripts/grpo/train_grpo.py --config <tmp.yaml> --max-steps 2` | GRPO prompts | reward std、metrics.jsonl | 是 |
| Eval 10 条 | 验证推理 | `python post_train/scripts/eval/evaluate_model.py --config post_train/configs/eval.yaml --model-path <model> --output-dir <tmp> --limit 10` | model + test | eval_samples, truncation | 是 |

## 14. 阅读过程中的高风险难点

- 大产物未提交：本地 manifest 记录全量，但 `train_pool.jsonl`、`teacher_accepted_20k.jsonl`、`sft_train_8k.jsonl`、`dpo_train.jsonl`、模型权重多数缺失。
- RFT 默认 `base_model_path` 是 Qwen3-8B，不是 SFT 0.6B；如果实验目标是 self-improvement，要复制 config 改路径。
- DPO pair shortfall：当前 manifest 目标 4000，实际 1889；不能为了凑数降低 correctness filter。
- GRPO 是 legacy 自写算法，不等同标准 TRL/verl GRPO；无 reference KL，`clip_eps` 裁剪 advantage。
- GRPO vLLM sync 是 checkpoint reload，失败后可能继续旧 rollout model。
- Eval 的独立 CLI 用 500 条 test；训练 callback 用 50 条 eval subset。
- 单卡共享 policy + vLLM 极易 OOM，`rollout_gpu_memory_utilization` 是关键。
- 测试状态：本机运行 `python -m pytest post_train/tests -q` 得到 127 passed、1 skipped、2 failed、91 errors；大量 errors 是 pytest 临时目录权限问题，2 个真实 failure 来自 `test_grpo_metrics.py` 仍按旧 `compute_rewards(rows, completions, format_reward, answer_reward)` 签名调用，而当前函数已新增 token/length penalty 参数。
- Windows 本地不是目标训练环境；正式训练文档假设 Linux、CUDA、4090/远程 GPU。

## 15. 一天结束后的验收问题

1. 为什么该项目要强制 `<answer>...</answer>`，而不是只看自然语言推理？
2. `solve_countdown` 和 `validate_countdown_expression` 的职责有何不同？
3. source record、teacher accepted record、DPO pair、GRPO prompt 各有哪些字段？
4. `stratified_sample` 如何避免只抽 easy 样本？
5. Full SFT 如何保证 prompt 不参与 loss？
6. LoRA 输出为什么不能像 Full SFT 一样直接评估，什么时候需要 `--base-model-path`？
7. RFT 的 accepted 数据如何产生，为什么它仍然是 SFT 风格训练？
8. DPO 的 `wrong_value` 为什么比 `missing_answer_tag` 更有价值？
9. 当前 DPO 为什么没有达到 `target_pairs: 4000`？
10. GRPO 的 reward 由哪几部分组成，长度惩罚何时生效？
11. `group_relative_advantages` 中 group std 为 0 会怎样？
12. 当前 legacy GRPO 与标准 GRPO/verl 的最大差异是什么？
13. vLLM rollout 与 Transformers policy model 如何在单卡共存？
14. `eval.yaml` 中 `test_data` 和 `eval_subset` 分别被谁使用？
15. 如何从 `grpo.yaml` 追踪到最终 `metrics.jsonl` 的每个字段？
16. 如果要新增 reward，需要改哪些函数和测试？
17. 如果要把流程迁移到双 GPU，legacy 哪些地方必须加 rank guard？
18. 与 `agentflow` 相比，本项目新增了哪些后训练工程能力？

## 16. 面试介绍框架

可以按 5 分钟结构讲：

1. 任务：Countdown 可验证算术表达式生成，目标是让 Qwen3-0.6B 在严格格式中答对。
2. 数据：raw parquet/json 经 solver 构造 source，8B teacher rollout 后用同一 validator 过滤 accepted。
3. 训练：Full SFT/LoRA 先对齐格式；RFT 用多采样正确样本继续 SFT；DPO 用 chosen/rejected pair 做偏好优化；GRPO 用在线 rollout 和 rule reward 提升。
4. 工程：YAML 驱动、vLLM 批量生成、Transformers/TRL/PEFT 训练、统一 AST/Fraction validator、独立 eval 矩阵。
5. 结果：base 0.6B accuracy 0，post-trained 0.6B 最高 GRPO 0.436，说明格式对齐和规则奖励比单纯参数规模更关键。
6. 局限：legacy GRPO 非标准、单卡显存脆弱、DPO pair shortfall、大产物未提交、checkpoint resume 不完整。
7. 后续：用 `post_train_v2` 的 DDP/verl/Manifest V2 重构，保留任务语义，替换分布式和 GRPO 训练栈。

## 17. 后续可深入研究方向

- 把 legacy GRPO 替换为 `post_train_v2` 的 verl GRPO，比较 reward/accuracy/throughput。
- 修复 `test_grpo_metrics.py` 与 `compute_rewards` 新签名不一致，补长度惩罚单测。
- 给 DPO data builder 增加更有效的 hard negative 生成策略，提高 pair 数量。
- 对 RFT 做 8B teacher vs SFT 0.6B self-rollout 对照。
- 缩短 teacher response 或训练只保留 answer span，降低 truncation。
- 在 eval 中增加按 bucket/num_count/错误类型的分层指标。
- 为 trainer callback 和 artifact 写入增加 rank guard，准备 DDP。
- 给 LoRA 增加 merge export，减少评估时 base path 风险。
- 将 data/outputs manifest 与输入 hash 串成完整 lineage。
- 做小模型不同 decoding 参数对 format/accuracy/truncation 的敏感性实验。
