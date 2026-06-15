from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from post_train_v2.src.evaluation import model_loading


class FakeTokenizer:
    pad_token_id = None
    eos_token = "<eos>"

    def __init__(self):
        self.pad_token = None


class FakeModel:
    def __init__(self):
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return self


def install_fake_modules(monkeypatch):
    calls: dict[str, list] = {"tokenizer": [], "model": [], "peft": []}
    tokenizer = FakeTokenizer()
    base_model = FakeModel()
    adapter_model = FakeModel()
    bfloat16 = object()

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls["tokenizer"].append((path, kwargs))
            return tokenizer

    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls["model"].append((path, kwargs))
            return base_model

    class PeftModel:
        @staticmethod
        def from_pretrained(model, path):
            calls["peft"].append((model, path))
            return adapter_model

    monkeypatch.setitem(
        __import__("sys").modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=AutoTokenizer,
            AutoModelForCausalLM=AutoModelForCausalLM,
        ),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "peft",
        SimpleNamespace(PeftModel=PeftModel),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "torch",
        SimpleNamespace(bfloat16=bfloat16),
    )
    return calls, tokenizer, base_model, adapter_model, bfloat16


def test_full_model_loads_bf16_flash_attention_2(monkeypatch, tmp_path: Path):
    calls, tokenizer, model, _, bfloat16 = install_fake_modules(monkeypatch)

    loaded_tokenizer, loaded_model = model_loading.load_model_and_tokenizer(
        tmp_path / "full"
    )

    assert loaded_tokenizer is tokenizer
    assert loaded_model is model
    assert calls["model"][0][1] == {
        "device_map": "auto",
        "trust_remote_code": True,
        "attn_implementation": "flash_attention_2",
        "torch_dtype": bfloat16,
    }
    assert model.eval_called is True
    assert tokenizer.pad_token == tokenizer.eos_token


def test_lora_uses_explicit_or_adapter_base_path(monkeypatch, tmp_path: Path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "configured-base"}),
        encoding="utf-8",
    )
    calls, _, base_model, adapter_model, _ = install_fake_modules(monkeypatch)

    _, loaded = model_loading.load_model_and_tokenizer(
        adapter, base_model_path="explicit-base"
    )

    assert loaded is adapter_model
    assert calls["model"][0][0] == "explicit-base"
    assert calls["peft"] == [(base_model, adapter)]

    calls, _, _, _, _ = install_fake_modules(monkeypatch)
    model_loading.load_model_and_tokenizer(adapter)
    assert calls["model"][0][0] == "configured-base"


def test_lora_without_base_path_fails(monkeypatch, tmp_path: Path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    install_fake_modules(monkeypatch)

    try:
        model_loading.load_model_and_tokenizer(adapter)
    except ValueError as error:
        assert "base-model-path" in str(error)
    else:
        raise AssertionError("missing LoRA base path must fail")
