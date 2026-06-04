import pytest

from post_train.scripts.sft.train_lora import resolve_lora_target_modules


class FakeModel:
    def __init__(self, names):
        self.names = names

    def named_modules(self):
        for name in self.names:
            yield name, object()


def test_resolve_lora_target_modules_auto_keeps_configured_suffix_order():
    model = FakeModel(
        [
            "model.layers.0.mlp.down_proj",
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.v_proj",
            "model.layers.0.other.not_a_projection",
        ]
    )

    assert resolve_lora_target_modules("auto", model) == ["q_proj", "v_proj", "down_proj"]


def test_resolve_lora_target_modules_auto_fails_when_no_known_projection_suffixes_exist():
    model = FakeModel(["model.embed_tokens", "model.layers.0.norm"])

    with pytest.raises(ValueError, match="No LoRA target modules found"):
        resolve_lora_target_modules("auto", model)


def test_resolve_lora_target_modules_supports_comma_separated_string():
    assert resolve_lora_target_modules("q_proj, v_proj,o_proj", FakeModel([])) == [
        "q_proj",
        "v_proj",
        "o_proj",
    ]


def test_resolve_lora_target_modules_supports_explicit_list():
    assert resolve_lora_target_modules(["q_proj", "k_proj"], FakeModel([])) == ["q_proj", "k_proj"]
