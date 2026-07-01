from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_module():
    try:
        return importlib.import_module("post_train.scripts.grpo.train_grpo_trl")
    except ModuleNotFoundError as exc:
        raise AssertionError("post_train.scripts.grpo.train_grpo_trl should exist") from exc


def test_prepare_grpo_records_preserves_schema_and_uses_chat_prompt():
    module = _load_module()
    rows = [
        {
            "id": "row-1",
            "prompt": "Use 1, 2, 3 to make 6.",
            "numbers": [1, 2, 3],
            "target": 6,
            "gold_expr": "1+2+3",
            "bucket": "n3",
        }
    ]

    records = module.prepare_grpo_records(rows)

    assert records == [
        {
            "id": "row-1",
            "prompt": [{"role": "user", "content": "Use 1, 2, 3 to make 6."}],
            "source_prompt": "Use 1, 2, 3 to make 6.",
            "numbers": [1, 2, 3],
            "target": 6,
            "gold_expr": "1+2+3",
            "bucket": "n3",
        }
    ]
    assert rows[0]["prompt"] == "Use 1, 2, 3 to make 6."


def test_countdown_reward_func_matches_legacy_reward_semantics():
    module = _load_module()
    module.consume_reward_diagnostics()

    rewards = module.countdown_reward_func(
        completions=[
            [{"role": "assistant", "content": "Reasoning\n<answer>(7-3)*(8-2)</answer>"}],
            [{"role": "assistant", "content": "No final answer"}],
            [{"role": "assistant", "content": "<answer>1+2+3</answer>"}],
        ],
        numbers=[[7, 3, 8, 2], [1, 2, 3], [1, 2, 3]],
        target=[24, 7, 7],
        completion_ids=[list(range(100)), list(range(50)), list(range(1024))],
        format_reward=0.2,
        answer_reward=1.0,
        max_completion_length=1024,
        length_penalty_start=800,
        length_penalty_max=-0.5,
    )

    assert rewards == [1.2, 0.0, -0.3]
    diagnostics = module.consume_reward_diagnostics()
    assert [row["format_ok"] for row in diagnostics] == [True, False, True]
    assert [row["correct"] for row in diagnostics] == [True, False, False]
    assert [row["truncated"] for row in diagnostics] == [False, False, True]
    assert module.consume_reward_diagnostics() == []


def test_build_trl_config_maps_legacy_clip_eps_to_standard_policy_clipping():
    module = _load_module()

    class FakeGRPOConfig:
        def __init__(
            self,
            output_dir,
            max_steps,
            per_device_train_batch_size,
            gradient_accumulation_steps,
            learning_rate,
            weight_decay,
            warmup_ratio,
            lr_scheduler_type,
            bf16,
            gradient_checkpointing,
            save_strategy,
            save_steps,
            logging_steps,
            report_to,
            run_name,
            remove_unused_columns,
            max_prompt_length,
            max_completion_length,
            num_generations,
            generation_batch_size,
            temperature,
            top_p,
            use_vllm,
            vllm_mode,
            vllm_gpu_memory_utilization,
            vllm_max_model_length,
            vllm_tensor_parallel_size,
            chat_template_kwargs,
            epsilon,
        ):
            self.kwargs = {
                key: value
                for key, value in locals().items()
                if key not in {"self", "__class__"}
            }

    cfg = {
        "output_dir": "unused",
        "batch_size": 4,
        "gradient_accumulation_steps": 2,
        "learning_rate": 5e-6,
        "weight_decay": 0.0,
        "warmup_ratio": 0.03,
        "scheduler": "cosine",
        "bf16": True,
        "gradient_checkpointing": True,
        "save_every_steps": 500,
        "logging_steps": 10,
        "report_to": None,
        "wandb_run_name": "countdown-grpo-trl",
        "max_prompt_len": 256,
        "max_new_tokens": 1024,
        "group_size": 5,
        "generation_batch_size": 20,
        "temperature": 1.2,
        "top_p": 0.95,
        "use_vllm": True,
        "vllm_mode": "colocate",
        "rollout_gpu_memory_utilization": 0.35,
        "rollout_max_model_len": 1024,
        "tensor_parallel_size": 2,
        "enable_thinking": False,
        "clip_eps": 0.2,
    }

    args = module.build_trl_grpo_config(
        cfg,
        output_dir="post_train/outputs/grpo_trl",
        max_steps=7,
        config_cls=FakeGRPOConfig,
    )

    assert args.kwargs["num_generations"] == 5
    assert args.kwargs["max_completion_length"] == 1024
    assert args.kwargs["vllm_gpu_memory_utilization"] == 0.35
    assert args.kwargs["vllm_tensor_parallel_size"] == 2
    assert args.kwargs["epsilon"] == 0.2
    assert "clip_eps" not in args.kwargs
    assert "advantage_clip_eps" not in args.kwargs


def test_build_legacy_metric_row_uses_reward_diagnostics_without_advantage_clipping():
    module = _load_module()
    diagnostics = [
        {"reward": 1.2, "format_ok": True, "correct": True, "truncated": False, "token_count": 100},
        {"reward": -0.3, "format_ok": True, "correct": False, "truncated": True, "token_count": 1024},
    ]

    row = module.build_legacy_metric_row(
        step=3,
        diagnostics=diagnostics,
        group_size=2,
        logs={"loss": 0.5, "learning_rate": 1e-6},
    )

    assert row["step"] == 3
    assert row["loss"] == 0.5
    assert row["learning_rate"] == 1e-6
    assert row["reward_mean"] == 0.45
    assert row["accuracy"] == 0.5
    assert row["format_rate"] == 1.0
    assert row["truncated_rate"] == 0.5
    assert "advantage_clip_eps" not in row

def test_reward_closure_accepts_trl_positional_prompts_and_completions():
    module = _load_module()
    reward_func = module.build_countdown_reward_func(
        {
            "format_reward": 0.2,
            "answer_reward": 1.0,
            "max_new_tokens": 1024,
            "length_penalty_start": 800,
            "length_penalty_max": -0.5,
        }
    )

    rewards = reward_func(
        ["Use 7, 3, 8, 2 to make 24."],
        [[{"role": "assistant", "content": "<answer>(7-3)*(8-2)</answer>"}]],
        numbers=[[7, 3, 8, 2]],
        target=[24],
        completion_ids=[list(range(20))],
    )

    assert rewards == [1.2]

def test_trl_metrics_callback_writes_only_on_world_process_zero(monkeypatch):
    module = _load_module()
    module.consume_reward_diagnostics()
    module._REWARD_DIAGNOSTICS.append(
        {"reward": 1.0, "format_ok": True, "correct": True, "truncated": False, "token_count": 10}
    )
    written = []
    monkeypatch.setattr(module, "_append_jsonl", lambda path, row: written.append((path, row)))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(TrainerCallback=object))

    callback = module.build_trl_metrics_callback(Path("out"), group_size=1)
    state = SimpleNamespace(global_step=1, is_world_process_zero=False)

    callback.on_log(args=None, state=state, control=SimpleNamespace(), logs={"loss": 0.5})

    assert written == []
