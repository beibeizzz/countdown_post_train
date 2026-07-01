from __future__ import annotations

import importlib


def _load_module():
    try:
        return importlib.import_module("post_train.scripts.opd.train_opd_gkd")
    except ModuleNotFoundError as exc:
        raise AssertionError("post_train.scripts.opd.train_opd_gkd should exist") from exc


def test_prepare_opd_gkd_records_builds_chatml_messages_and_preserves_metadata():
    module = _load_module()
    rows = [
        {
            "id": "row-1",
            "prompt": "Use 1, 2, 3 to make 6.",
            "response": "<answer>1+2+3</answer>",
            "numbers": [1, 2, 3],
            "target": 6,
            "bucket": "n3",
        }
    ]

    records = module.prepare_opd_gkd_records(rows)

    assert records == [
        {
            "id": "row-1",
            "messages": [
                {"role": "user", "content": "Use 1, 2, 3 to make 6."},
                {"role": "assistant", "content": "<answer>1+2+3</answer>"},
            ],
            "prompt": "Use 1, 2, 3 to make 6.",
            "response": "<answer>1+2+3</answer>",
            "numbers": [1, 2, 3],
            "target": 6,
            "bucket": "n3",
        }
    ]


def test_prepare_opd_gkd_records_requires_prompt_and_response():
    module = _load_module()

    for row in (
        {"id": "row-1", "response": "x"},
        {"id": "row-1", "prompt": "x"},
    ):
        try:
            module.prepare_opd_gkd_records([row])
        except ValueError as exc:
            assert "must be a non-empty string" in str(exc)
        else:
            raise AssertionError("Expected invalid OPD GKD row to fail")


def test_build_gkd_config_maps_on_policy_forward_kl_settings():
    module = _load_module()

    class FakeGKDConfig:
        def __init__(
            self,
            output_dir,
            max_steps,
            num_train_epochs,
            per_device_train_batch_size,
            gradient_accumulation_steps,
            learning_rate,
            weight_decay,
            warmup_steps,
            lr_scheduler_type,
            bf16,
            gradient_checkpointing,
            save_strategy,
            save_steps,
            logging_steps,
            report_to,
            run_name,
            remove_unused_columns,
            max_length,
            max_new_tokens,
            temperature,
            lmbda,
            beta,
            seq_kd,
            teacher_model_name_or_path,
            teacher_model_init_kwargs,
        ):
            self.kwargs = {
                key: value
                for key, value in locals().items()
                if key not in {"self", "__class__"}
            }

    cfg = {
        "teacher_model_path": "post_train/model/qwen/qwen3-8b",
        "output_dir": "post_train/outputs/opd/gkd",
        "max_steps": 500,
        "epochs": 1,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "learning_rate": 5e-6,
        "weight_decay": 0.0,
        "warmup_ratio": 0.03,
        "scheduler": "cosine",
        "bf16": True,
        "gradient_checkpointing": True,
        "save_every_steps": 100,
        "logging_steps": 10,
        "report_to": None,
        "run_name": "opd_gkd",
        "run_name_auto_suffix": False,
        "max_seq_len": 1024,
        "max_new_tokens": 256,
        "temperature": 0.9,
        "lmbda": 1.0,
        "beta": 0.0,
        "seq_kd": False,
    }

    args = module.build_gkd_training_args(
        cfg,
        output_dir="post_train/outputs/opd/gkd",
        max_steps=500,
        config_cls=FakeGKDConfig,
    )

    assert args.kwargs["lmbda"] == 1.0
    assert args.kwargs["beta"] == 0.0
    assert args.kwargs["max_new_tokens"] == 256
    assert args.kwargs["warmup_steps"] == 15
    assert args.kwargs["teacher_model_name_or_path"] == "post_train/model/qwen/qwen3-8b"
    assert args.kwargs["teacher_model_init_kwargs"] == {
        "trust_remote_code": True,
        "attn_implementation": "flash_attention_2",
        "dtype": "bfloat16",
    }


def test_build_gkd_trainer_passes_student_model_teacher_path_and_processing_class():
    module = _load_module()

    class FakeGKDTrainer:
        def __init__(self, model, teacher_model, args, train_dataset, processing_class, callbacks):
            self.kwargs = {
                "model": model,
                "teacher_model": teacher_model,
                "args": args,
                "train_dataset": train_dataset,
                "processing_class": processing_class,
                "callbacks": callbacks,
            }

    trainer = module.build_gkd_trainer(
        model="student-object",
        tokenizer="tokenizer-object",
        teacher_model_path="teacher-path",
        training_args="args-object",
        train_dataset="dataset-object",
        callbacks=["callback"],
        trainer_cls=FakeGKDTrainer,
    )

    assert trainer.kwargs == {
        "model": "student-object",
        "teacher_model": "teacher-path",
        "args": "args-object",
        "train_dataset": "dataset-object",
        "processing_class": "tokenizer-object",
        "callbacks": ["callback"],
    }
