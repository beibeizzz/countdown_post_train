import builtins
import sys
import types
from pathlib import Path

import pytest

from post_train.scripts.sft.train_full import load_model_and_tokenizer


class FakeTokenizer:
    pad_token_id = None
    eos_token = "<eos>"

    def __init__(self, path):
        self.path = Path(path)
        self.pad_token = None


class FakeAutoTokenizer:
    calls = []

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        cls.calls.append((Path(path), kwargs))
        return FakeTokenizer(path)


class FakeConfig:
    use_cache = True


class FakeModel:
    def __init__(self):
        self.config = FakeConfig()
        self.gradient_checkpointing_enabled = False

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing_enabled = True


class FakeAutoModelForCausalLM:
    calls = []

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        cls.calls.append((Path(path), kwargs))
        return FakeModel()


@pytest.fixture(autouse=True)
def fake_model_modules(monkeypatch):
    FakeAutoTokenizer.calls = []
    FakeAutoModelForCausalLM.calls = []
    fake_bfloat16 = object()
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch":
            return types.SimpleNamespace(bfloat16=fake_bfloat16)
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoModelForCausalLM=FakeAutoModelForCausalLM,
            AutoTokenizer=FakeAutoTokenizer,
        ),
    )
    return fake_bfloat16


def test_load_model_and_tokenizer_forces_flash_attention_2_and_bfloat16(
    tmp_path: Path,
    fake_model_modules,
):
    model_path = tmp_path / "model"

    model, tokenizer = load_model_and_tokenizer(model_path, gradient_checkpointing=False)

    assert FakeAutoTokenizer.calls == [(model_path, {"trust_remote_code": True})]
    assert FakeAutoModelForCausalLM.calls == [
        (
            model_path,
            {
                "trust_remote_code": True,
                "attn_implementation": "flash_attention_2",
                "torch_dtype": fake_model_modules,
            },
        )
    ]
    assert tokenizer.pad_token == "<eos>"
    assert model.gradient_checkpointing_enabled is False


def test_load_model_and_tokenizer_preserves_gradient_checkpointing(tmp_path: Path):
    model, _tokenizer = load_model_and_tokenizer(
        tmp_path / "model",
        gradient_checkpointing=True,
    )

    assert model.config.use_cache is False
    assert model.gradient_checkpointing_enabled is True
