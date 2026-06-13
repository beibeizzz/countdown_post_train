from __future__ import annotations

import ast
import builtins
import types
from pathlib import Path

import pytest

from post_train.scripts.dpo import train_dpo
from post_train.scripts.grpo import train_grpo
from post_train.scripts.sft import train_full, train_lora, train_rft


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "post_train" / "scripts"


def parse_script(relative_path: str) -> ast.Module:
    return ast.parse((SCRIPTS_ROOT / relative_path).read_text(encoding="utf-8"))


def imported_names(tree: ast.Module, module: str) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == module:
            names.update(alias.name for alias in node.names)
    return names


def called_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            names.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            names.add(node.func.attr)
    return names


class LoaderReached(RuntimeError):
    pass


def stop_at_shared_loader(*args, **kwargs):
    raise LoaderReached


@pytest.mark.parametrize(
    "relative_path",
    (
        "sft/train_lora.py",
        "dpo/train_dpo.py",
        "grpo/train_grpo.py",
    ),
)
def test_training_entrypoint_imports_and_calls_shared_model_loader(
    relative_path: str,
) -> None:
    tree = parse_script(relative_path)

    assert "load_model_and_tokenizer" in imported_names(
        tree,
        "post_train.scripts.sft.train_full",
    )
    assert "load_model_and_tokenizer" in called_names(tree)


def test_rft_entrypoint_routes_training_through_full_sft_runner() -> None:
    tree = parse_script("sft/train_rft.py")
    full_tree = parse_script("sft/train_full.py")

    assert "run_sft_training" in imported_names(
        tree,
        "post_train.scripts.sft.train_full",
    )
    assert "run_sft_training" in called_names(tree)
    assert "load_model_and_tokenizer" in called_names(full_tree)


def test_lora_main_reaches_shared_model_loader(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        train_lora,
        "parse_args",
        lambda: types.SimpleNamespace(config="config.yaml", max_steps=None),
    )
    monkeypatch.setattr(train_lora, "resolve_path", lambda value, root: tmp_path / value)
    monkeypatch.setattr(
        train_lora,
        "load_yaml_config",
        lambda path: {
            "model_path": "model",
            "train_data": "train.jsonl",
            "output_dir": "output",
        },
    )
    monkeypatch.setattr(train_lora, "load_model_and_tokenizer", stop_at_shared_loader)

    with pytest.raises(LoaderReached):
        train_lora.main()


def test_rft_main_reaches_shared_model_loader(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        train_rft,
        "parse_args",
        lambda: types.SimpleNamespace(config="config.yaml", max_steps=None),
    )
    monkeypatch.setattr(train_rft, "resolve_path", lambda value, root: tmp_path / value)
    monkeypatch.setattr(
        train_rft,
        "load_yaml_config",
        lambda path: {
            "target_model_path": "model",
            "accepted_output": "train.jsonl",
            "output_dir": "output",
            "train": {},
        },
    )
    monkeypatch.setattr(train_full, "load_model_and_tokenizer", stop_at_shared_loader)

    with pytest.raises(LoaderReached):
        train_rft.main()


def test_dpo_main_reaches_shared_model_loader(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        train_dpo,
        "parse_args",
        lambda: types.SimpleNamespace(config="config.yaml", max_steps=None),
    )
    monkeypatch.setattr(train_dpo, "resolve_path", lambda value, root: tmp_path / value)
    monkeypatch.setattr(
        train_dpo,
        "load_yaml_config",
        lambda path: {
            "model_path": "model",
            "train_data": "train.jsonl",
            "output_dir": "output",
        },
    )
    monkeypatch.setattr(train_dpo, "load_model_and_tokenizer", stop_at_shared_loader)

    with pytest.raises(LoaderReached):
        train_dpo.main()


def test_grpo_training_reaches_shared_model_loader(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(train_grpo, "init_wandb_if_enabled", lambda cfg, default_name: None)
    monkeypatch.setattr(train_grpo, "load_model_and_tokenizer", stop_at_shared_loader)
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch":
            return types.ModuleType("torch")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(LoaderReached):
        train_grpo.train_grpo(
            {
                "kl_coeff": 0.0,
                "gradient_checkpointing": False,
            },
            max_steps=1,
            model_path=tmp_path / "model",
            output_dir=tmp_path / "output",
        )


@pytest.mark.parametrize(
    "relative_path",
    (
        "sft/train_lora.py",
        "sft/train_rft.py",
        "dpo/train_dpo.py",
        "grpo/train_grpo.py",
    ),
)
def test_training_entrypoint_does_not_load_auto_model_directly(
    relative_path: str,
) -> None:
    tree = parse_script(relative_path)

    direct_imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "AutoModelForCausalLM" not in direct_imports
    assert not any(
        (isinstance(node, ast.Name) and node.id == "AutoModelForCausalLM")
        or (isinstance(node, ast.Attribute) and node.attr == "AutoModelForCausalLM")
        for node in ast.walk(tree)
    )

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        assert not (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "from_pretrained"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "AutoModelForCausalLM"
        )
        assert not (
            isinstance(node.func, ast.Name)
            and node.func.id == "AutoModelForCausalLM"
        )
