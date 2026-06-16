from __future__ import annotations

from types import SimpleNamespace

from post_train_v2.scripts.sft.train_lora import build_parser
from post_train_v2.src.training import lora


class FakeParameter:
    def __init__(self):
        self.requires_grad = True


class FakeModel:
    def __init__(self):
        self.input_grads_enabled = False
        self.params = {
            "model.layers.0.self_attn.q_proj.weight": FakeParameter(),
            "model.layers.0.self_attn.q_proj.lora_A.weight": FakeParameter(),
            "model.layers.0.mlp.down_proj.lora_B.weight": FakeParameter(),
        }

    def named_modules(self):
        for name in (
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.k_proj",
            "model.layers.0.self_attn.v_proj",
            "model.layers.0.self_attn.o_proj",
            "model.layers.0.mlp.gate_proj",
            "model.layers.0.mlp.up_proj",
            "model.layers.0.mlp.down_proj",
        ):
            yield name, object()

    def named_parameters(self):
        return self.params.items()

    def enable_input_require_grads(self):
        self.input_grads_enabled = True


def test_train_lora_cli_accepts_common_options():
    args = build_parser().parse_args(
        [
            "--config",
            "post_train_v2/configs/sft/lora_smoke.yaml",
            "--max-steps",
            "2",
            "--resume-from-checkpoint",
            "checkpoint-100",
        ]
    )

    assert args.config == "post_train_v2/configs/sft/lora_smoke.yaml"
    assert args.max_steps == 2
    assert args.resume_from_checkpoint == "checkpoint-100"


def test_lora_target_modules_and_trainable_params(monkeypatch):
    model = FakeModel()
    peft_calls = []

    class FakeLoraConfig:
        def __init__(self, **kwargs):
            peft_calls.append(("config", kwargs))

    fake_peft = SimpleNamespace(
        LoraConfig=FakeLoraConfig,
        get_peft_model=lambda model, config: peft_calls.append(("wrap", model))
        or model,
        TaskType=SimpleNamespace(CAUSAL_LM="CAUSAL_LM"),
    )
    monkeypatch.setattr(lora, "import_module", lambda name: fake_peft)

    wrapped = lora.apply_lora(
        model,
        {
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "gradient_checkpointing": True,
        },
    )

    assert wrapped is model
    assert peft_calls[0][1]["target_modules"] == [
        "down_proj",
        "gate_proj",
        "k_proj",
        "o_proj",
        "q_proj",
        "up_proj",
        "v_proj",
    ]
    assert peft_calls[0][1]["r"] == 16
    assert peft_calls[0][1]["lora_alpha"] == 32
    assert peft_calls[0][1]["lora_dropout"] == 0.05
    assert model.input_grads_enabled is True
    assert {
        name: parameter.requires_grad
        for name, parameter in model.named_parameters()
    } == {
        "model.layers.0.self_attn.q_proj.weight": False,
        "model.layers.0.self_attn.q_proj.lora_A.weight": True,
        "model.layers.0.mlp.down_proj.lora_B.weight": True,
    }


def test_merge_loads_adapter_over_base_and_merges(monkeypatch, tmp_path):
    events = []
    tokenizer = SimpleNamespace(save_pretrained=lambda path: events.append(("tokenizer", path)))
    base = object()

    class FakePeftModel:
        @classmethod
        def from_pretrained(cls, model, adapter_path):
            events.append(("from_pretrained", model, adapter_path))
            return cls()

        def merge_and_unload(self):
            events.append(("merge",))
            return SimpleNamespace(
                save_pretrained=lambda path: events.append(("model", path))
            )

    monkeypatch.setattr(
        lora,
        "load_causal_lm_and_tokenizer",
        lambda model_path, gradient_checkpointing: (base, tokenizer),
    )
    monkeypatch.setattr(
        lora,
        "import_module",
        lambda name: SimpleNamespace(PeftModel=FakePeftModel),
    )

    lora.merge_lora_adapter(
        base_model_path="base",
        adapter_path="adapter",
        output_dir=tmp_path,
    )

    assert events == [
        ("from_pretrained", base, "adapter"),
        ("merge",),
        ("model", tmp_path),
        ("tokenizer", tmp_path),
    ]


def test_lora_config_marks_adapter_export_kind():
    from post_train_v2.src.config.loading import load_yaml

    assert load_yaml("post_train_v2/configs/sft/lora.yaml")["export_kind"] == "lora_adapter"
