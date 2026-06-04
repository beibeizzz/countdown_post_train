import sys
import types
from pathlib import Path

import pytest

from post_train.scripts.eval.evaluate_model import load_model_and_tokenizer


class FakeTokenizer:
    pad_token_id = None
    eos_token = "<eos>"

    def __init__(self, path):
        self.path = path
        self.pad_token = None


class FakeAutoTokenizer:
    loaded_paths = []

    @classmethod
    def from_pretrained(cls, path, trust_remote_code=True):
        cls.loaded_paths.append(Path(path))
        return FakeTokenizer(path)


class FakeBaseModel:
    def __init__(self, path):
        self.path = path
        self.eval_called = False

    def eval(self):
        self.eval_called = True


class FakeAutoModelForCausalLM:
    loaded_paths = []

    @classmethod
    def from_pretrained(cls, path, device_map="auto", trust_remote_code=True):
        cls.loaded_paths.append(Path(path))
        return FakeBaseModel(path)


class FakePeftModel:
    loaded = []

    @classmethod
    def from_pretrained(cls, model, path):
        cls.loaded.append((model, Path(path)))
        return model


@pytest.fixture(autouse=True)
def fake_model_modules(monkeypatch):
    FakeAutoTokenizer.loaded_paths = []
    FakeAutoModelForCausalLM.loaded_paths = []
    FakePeftModel.loaded = []
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoModelForCausalLM=FakeAutoModelForCausalLM,
            AutoTokenizer=FakeAutoTokenizer,
        ),
    )
    monkeypatch.setitem(sys.modules, "peft", types.SimpleNamespace(PeftModel=FakePeftModel))


def test_load_model_and_tokenizer_loads_full_model_directly(tmp_path: Path):
    model_dir = tmp_path / "full"
    model_dir.mkdir()

    tokenizer, model = load_model_and_tokenizer(model_dir)

    assert FakeAutoTokenizer.loaded_paths == [model_dir]
    assert FakeAutoModelForCausalLM.loaded_paths == [model_dir]
    assert FakePeftModel.loaded == []
    assert tokenizer.pad_token == "<eos>"
    assert model.eval_called is True


def test_load_model_and_tokenizer_loads_lora_adapter_with_base_model(tmp_path: Path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    base_dir = tmp_path / "base"
    base_dir.mkdir()

    _tokenizer, model = load_model_and_tokenizer(adapter_dir, base_model_path=base_dir)

    assert FakeAutoTokenizer.loaded_paths == [adapter_dir]
    assert FakeAutoModelForCausalLM.loaded_paths == [base_dir]
    assert FakePeftModel.loaded == [(model, adapter_dir)]
    assert model.eval_called is True


def test_load_model_and_tokenizer_requires_base_for_lora_adapter(tmp_path: Path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="--base-model-path"):
        load_model_and_tokenizer(adapter_dir)
