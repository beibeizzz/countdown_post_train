from __future__ import annotations

import ast
import builtins
import importlib.util
import py_compile
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "post_train_v2" / "scripts" / "env"
SCRIPT_NAMES = (
    "check_runtime.py",
    "smoke_nccl.py",
    "smoke_flash_attention.py",
    "smoke_transformers.py",
    "smoke_vllm.py",
    "smoke_teacher_dual_engine.py",
    "smoke_trl_peft.py",
    "smoke_legacy_loader.py",
    "smoke_eval_loader.py",
)
HEAVY_MODULES = {
    "datasets",
    "flash_attn",
    "peft",
    "ray",
    "torch",
    "transformers",
    "trl",
    "vllm",
}


def load_script(name: str):
    path = SCRIPTS_DIR / name
    spec = importlib.util.spec_from_file_location(f"test_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", SCRIPT_NAMES)
def test_environment_script_compiles(name: str, tmp_path: Path) -> None:
    py_compile.compile(
        str(SCRIPTS_DIR / name),
        cfile=str(tmp_path / f"{name}.pyc"),
        doraise=True,
    )


@pytest.mark.parametrize("name", SCRIPT_NAMES)
def test_gpu_and_ml_dependencies_are_not_imported_at_module_scope(name: str) -> None:
    tree = ast.parse((SCRIPTS_DIR / name).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".", 1)[0])

    assert imported.isdisjoint(HEAVY_MODULES)


@pytest.mark.parametrize(
    ("name", "expected_options"),
    (
        ("check_runtime.py", ("--manifest", "--require-gpus", "--check-ray")),
        ("smoke_nccl.py", ()),
        ("smoke_flash_attention.py", ("--device",)),
        (
            "smoke_transformers.py",
            ("--model-path", "--max-seq-length", "--device"),
        ),
        (
            "smoke_vllm.py",
            ("--model-path", "--tensor-parallel-size", "--gpu-memory-utilization"),
        ),
        (
            "smoke_teacher_dual_engine.py",
            ("--model-path", "--gpu-memory-utilization", "--timeout-seconds"),
        ),
        ("smoke_trl_peft.py", ("--model-path", "--work-dir", "--device")),
        (
            "smoke_legacy_loader.py",
            ("--model-path", "--device", "--max-seq-length"),
        ),
        (
            "smoke_eval_loader.py",
            ("--model-path", "--base-model-path", "--max-new-tokens"),
        ),
    ),
)
def test_help_works_without_gpu_dependencies(
    name: str, expected_options: tuple[str, ...]
) -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / name), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    for option in expected_options:
        assert option in result.stdout


def test_nccl_peer_access_warning_does_not_fail(capsys: pytest.CaptureFixture[str]) -> None:
    module = load_script("smoke_nccl.py")

    exit_code = module.report_peer_access(
        peer_access=False,
        ipc_available=False,
        writer=print,
    )

    assert exit_code == 0
    assert "WARNING" in capsys.readouterr().out


def test_runtime_peer_access_warning_does_not_fail(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = load_script("check_runtime.py")

    exit_code = module.report_peer_access(
        peer_access=False,
        ipc_available=False,
        writer=print,
    )

    assert exit_code == 0
    assert "WARNING" in capsys.readouterr().out


class FakeScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeFinite:
    def all(self):
        return FakeScalar(True)


class FakeTensor:
    def __init__(self, value: float = 0.0):
        self.value = value

    def item(self):
        return self.value

    def untyped_storage(self):
        return SimpleNamespace(_share_cuda_=lambda: ())


class FakeCuda:
    def __init__(self, names: tuple[str, str], capabilities=((8, 0), (8, 0))):
        self.names = names
        self.capabilities = capabilities

    def is_available(self):
        return True

    def device_count(self):
        return 2

    def set_device(self, index):
        return None

    def synchronize(self, index):
        return None

    def can_device_access_peer(self, source, target):
        return False

    def get_device_properties(self, index):
        major, minor = self.capabilities[index]
        return SimpleNamespace(name=self.names[index], major=major, minor=minor)

    def device(self, index):
        return nullcontext()

    def mem_get_info(self):
        return 30 * 2**30, 40 * 2**30


class FakeDist:
    def __init__(self):
        self.destroyed = False

    def init_process_group(self, backend):
        assert backend == "nccl"

    def get_rank(self):
        return 0

    def all_reduce(self, tensor):
        tensor.value = 3.0

    def barrier(self):
        return None

    def is_initialized(self):
        return True

    def destroy_process_group(self):
        self.destroyed = True


def fake_torch(names=("NVIDIA A100-SXM4-80GB", "NVIDIA A100-SXM4-80GB"), capabilities=((8, 0), (8, 0))):
    cuda = FakeCuda(names=names, capabilities=capabilities)
    return SimpleNamespace(
        __version__="2.7.0+cu128",
        version=SimpleNamespace(cuda="12.8"),
        _C=SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=True),
        cuda=cuda,
        tensor=lambda value, device: FakeTensor(value[0]),
        ones=lambda size, device: FakeTensor(),
        isfinite=lambda tensor: FakeFinite(),
    )


def runtime_manifest():
    return {
        "cuda_runtime": "12.8",
        "packages": {"torch": "2.7.0"},
        "platform": {"gpu_compute_capability": "8.0"},
    }


def test_nccl_collective_success_with_no_peer_access_warns_and_returns_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = load_script("smoke_nccl.py")
    dist = FakeDist()

    result = module.run_collective(
        torch_module=fake_torch(),
        dist_module=dist,
        environment={"RANK": "0", "WORLD_SIZE": "2", "LOCAL_RANK": "0"},
        writer=print,
        ipc_probe=lambda tensor: False,
    )

    assert result == 0
    assert dist.destroyed is True
    output = capsys.readouterr().out
    assert "WARNING" in output
    assert "all_reduce=3.0" in output


def test_runtime_accepts_manifest_matching_a100_devices() -> None:
    module = load_script("check_runtime.py")

    module.check_torch(
        runtime_manifest(),
        require_gpus=2,
        torch_module=fake_torch(),
        writer=lambda message: None,
        ipc_probe=lambda tensor: False,
    )


@pytest.mark.parametrize(
    ("torch_module", "match"),
    (
        (fake_torch(names=("NVIDIA H100", "NVIDIA A100")), "A100"),
        (fake_torch(capabilities=((9, 0), (8, 0))), "compute capability"),
    ),
)
def test_runtime_rejects_wrong_gpu_platform(torch_module, match: str) -> None:
    module = load_script("check_runtime.py")

    with pytest.raises(RuntimeError, match=match):
        module.check_torch(
            runtime_manifest(),
            require_gpus=2,
            torch_module=torch_module,
            writer=lambda message: None,
            ipc_probe=lambda tensor: False,
        )


class FakeRay:
    def __init__(self, initialized: bool) -> None:
        self.initialized = initialized
        self.init_calls = 0
        self.shutdown_calls = 0

    def is_initialized(self):
        return self.initialized

    def init(self, **kwargs):
        self.init_calls += 1
        self.initialized = True

    def cluster_resources(self):
        return {"GPU": 2.0}

    def shutdown(self):
        self.shutdown_calls += 1
        self.initialized = False


def test_ray_existing_session_is_not_shutdown() -> None:
    module = load_script("check_runtime.py")
    ray = FakeRay(initialized=True)

    module.check_ray_resources(2, ray_module=ray, writer=lambda message: None)

    assert ray.init_calls == 0
    assert ray.shutdown_calls == 0


def test_ray_session_started_here_is_shutdown() -> None:
    module = load_script("check_runtime.py")
    ray = FakeRay(initialized=False)

    module.check_ray_resources(2, ray_module=ray, writer=lambda message: None)

    assert ray.init_calls == 1
    assert ray.shutdown_calls == 1


def test_teacher_child_command_and_environment_are_isolated() -> None:
    module = load_script("smoke_teacher_dual_engine.py")

    command = module.build_child_command(
        python_executable="/env/bin/python",
        script_path=Path("/repo/smoke_teacher_dual_engine.py"),
        model_path="/models/qwen3-8b",
        gpu_memory_utilization=0.42,
        launch_index=1,
    )
    environment = module.build_child_environment({"PATH": "/bin"}, launch_index=1)

    assert command == [
        "/env/bin/python",
        str(Path("/repo/smoke_teacher_dual_engine.py")),
        "--child",
        "--model-path",
        "/models/qwen3-8b",
        "--gpu-memory-utilization",
        "0.42",
        "--launch-index",
        "1",
    ]
    assert environment["CUDA_VISIBLE_DEVICES"] == "1"
    assert environment["PATH"] == "/bin"


def make_teacher_report(
    index: int,
    uuid: str,
    pci_bus_id: str,
    **overrides,
) -> dict[str, object]:
    memory = {
        "name": "NVIDIA A100-SXM4-80GB",
        "free_bytes": 30 * 2**30,
        "total_bytes": 40 * 2**30,
    }
    report = {
        "launch_index": index,
        "cuda_visible_devices": str(index),
        "device_count": 1,
        "current_device": 0,
        "cuda_uuid": uuid,
        "cuda_pci_bus_id": pci_bus_id,
        "memory_before": memory,
        "memory_after": memory,
        "output": "<answer> 4 </answer>",
    }
    report.update(overrides)
    return report


def test_teacher_parses_and_validates_distinct_physical_gpus() -> None:
    module = load_script("smoke_teacher_dual_engine.py")
    physical = module.parse_nvidia_smi_output(
        "0, GPU-00112233-4455-6677-8899-aabbccddeeff, 00000000:8E:00.0\n"
        "1, GPU-10213243-5465-7687-98a9-bacbdcedfe0f, 00000000:B3:00.0\n"
    )
    reports = [
        make_teacher_report(
            0,
            "GPU-00112233-4455-6677-8899-aabbccddeeff",
            "0000:8e:00.0",
        ),
        make_teacher_report(
            1,
            "GPU-10213243-5465-7687-98a9-bacbdcedfe0f",
            "0000:b3:00.0",
        ),
    ]

    validated = module.validate_teacher_reports(reports, physical)

    assert [item["cuda_uuid"] for item in validated] == [
        "GPU-00112233-4455-6677-8899-aabbccddeeff",
        "GPU-10213243-5465-7687-98a9-bacbdcedfe0f",
    ]
    assert [item["cuda_pci_bus_id"] for item in validated] == [
        "0000:8e:00.0",
        "0000:b3:00.0",
    ]


def test_teacher_queries_required_nvidia_smi_fields() -> None:
    module = load_script("smoke_teacher_dual_engine.py")
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(command: list[str], **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(
            returncode=0,
            stdout="0, GPU-aaaa, 00000000:8E:00.0\n",
            stderr="",
        )

    inventory = module.query_physical_gpus(run_command=fake_run)

    assert inventory[0]["uuid"] == "GPU-aaaa"
    assert calls == [
        (
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,pci.bus_id",
                "--format=csv,noheader,nounits",
            ],
            {"check": False, "capture_output": True, "text": True},
        )
    ]


@pytest.mark.parametrize(
    "inventory",
    (
        "0, GPU-same, 00000000:8E:00.0\n1, GPU-same, 00000000:B3:00.0\n",
        "0, GPU-aaaa, 00000000:8E:00.0\n1, GPU-bbbb, 00000000:8E:00.0\n",
    ),
)
def test_teacher_rejects_duplicate_physical_identity(inventory: str) -> None:
    module = load_script("smoke_teacher_dual_engine.py")
    physical = module.parse_nvidia_smi_output(inventory)
    reports = [
        make_teacher_report(
            index,
            physical[index]["uuid"],
            physical[index]["pci_bus_id"],
            output="ok",
        )
        for index in (0, 1)
    ]

    with pytest.raises(ValueError, match="physical GPU"):
        module.validate_teacher_reports(reports, physical)


def test_teacher_rejects_device_isolation_violation() -> None:
    module = load_script("smoke_teacher_dual_engine.py")
    physical = module.parse_nvidia_smi_output(
        "0, GPU-00112233-4455-6677-8899-aabbccddeeff, 00000000:8E:00.0\n"
        "1, GPU-10213243-5465-7687-98a9-bacbdcedfe0f, 00000000:B3:00.0\n"
    )
    reports = [
        make_teacher_report(
            0,
            physical[0]["uuid"],
            physical[0]["pci_bus_id"],
            cuda_visible_devices="0,1",
            device_count=2,
            output="ok",
        ),
        make_teacher_report(
            1,
            physical[1]["uuid"],
            physical[1]["pci_bus_id"],
            output="ok",
        ),
    ]

    with pytest.raises(ValueError, match="isolation"):
        module.validate_teacher_reports(reports, physical)


def script_tree(name: str) -> ast.Module:
    return ast.parse((SCRIPTS_DIR / name).read_text(encoding="utf-8"))


def call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def call_nodes(name: str, target: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(script_tree(name))
        if isinstance(node, ast.Call) and call_name(node) == target
    ]


def call_names(name: str) -> set[str]:
    tree = script_tree(name)
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        names.add(call_name(node))
    return names


@pytest.mark.parametrize(
    ("name", "required_calls"),
    (
        ("smoke_flash_attention.py", {"flash_attn_func", "backward"}),
        (
            "smoke_transformers.py",
            {"from_pretrained", "to", "backward"},
        ),
        ("smoke_vllm.py", {"LLM", "chat"}),
        ("smoke_teacher_dual_engine.py", {"LLM", "chat", "query_cuda_identity"}),
        (
            "smoke_trl_peft.py",
            {
                "save_pretrained",
                "from_pretrained",
                "merge_and_unload",
                "SFTTrainer",
                "DPOTrainer",
            },
        ),
        (
            "smoke_legacy_loader.py",
            {
                "load_model_and_tokenizer",
                "to",
                "backward",
                "isfinite",
            },
        ),
        (
            "smoke_eval_loader.py",
            {
                "load_model_and_tokenizer",
                "generate",
                "decode",
            },
        ),
    ),
)
def test_scripts_contain_required_call_structure(
    name: str, required_calls: set[str]
) -> None:
    assert required_calls <= call_names(name)


def test_transformers_and_vllm_keyword_contracts_are_ast_enforced() -> None:
    def keyword_values(node: ast.Call) -> dict[str, ast.expr]:
        return {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg}

    transformer_loads = [
        keyword_values(node)
        for node in call_nodes("smoke_transformers.py", "from_pretrained")
    ]
    assert any(
        isinstance(keywords.get("attn_implementation"), ast.Constant)
        and keywords["attn_implementation"].value == "flash_attention_2"
        and isinstance(keywords.get("torch_dtype"), ast.Attribute)
        and keywords["torch_dtype"].attr == "bfloat16"
        for keywords in transformer_loads
    )

    for name in ("smoke_vllm.py", "smoke_teacher_dual_engine.py"):
        chats = [
            keyword_values(node)
            for node in call_nodes(name, "chat")
        ]
        assert chats
        thinking_values = []
        for keywords in chats:
            value = keywords.get("chat_template_kwargs")
            if not isinstance(value, ast.Dict):
                continue
            for key, item in zip(value.keys, value.values):
                if isinstance(key, ast.Constant) and key.value == "enable_thinking":
                    thinking_values.append(item)
        assert any(
            isinstance(value, ast.Constant) and value.value is False
            for value in thinking_values
        )

    teacher_llm = call_nodes("smoke_teacher_dual_engine.py", "LLM")
    assert any(
        isinstance(keyword_values(node).get("tensor_parallel_size"), ast.Constant)
        and keyword_values(node)["tensor_parallel_size"].value == 1
        for node in teacher_llm
    )

    flash_random = call_nodes("smoke_flash_attention.py", "randn")
    assert any(
        isinstance(keyword_values(node).get("dtype"), ast.Attribute)
        and keyword_values(node)["dtype"].attr == "bfloat16"
        for node in flash_random
    )


def test_trl_peft_smoke_saves_tokenizer_with_adapter() -> None:
    tree = script_tree("smoke_trl_peft.py")
    matching_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "tokenizer"
        and node.func.attr == "save_pretrained"
    ]

    assert len(matching_calls) == 1
    assert len(matching_calls[0].args) == 1
    assert isinstance(matching_calls[0].args[0], ast.Name)
    assert matching_calls[0].args[0].id == "adapter_dir"


def test_trl_peft_smoke_runs_one_sft_and_one_dpo_training_step() -> None:
    tree = script_tree("smoke_trl_peft.py")
    train_receivers = [
        node.func.value.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "train"
        and isinstance(node.func.value, ast.Name)
    ]

    assert train_receivers.count("sft_trainer") == 1
    assert train_receivers.count("dpo_trainer") == 1

    configs = call_nodes("smoke_trl_peft.py", "SFTConfig") + call_nodes(
        "smoke_trl_peft.py", "DPOConfig"
    )
    assert len(configs) == 2
    for config in configs:
        keywords = {item.arg: item.value for item in config.keywords if item.arg}
        assert isinstance(keywords.get("max_steps"), ast.Constant)
        assert keywords["max_steps"].value == 1


def test_trl_peft_dpo_uses_fresh_base_model() -> None:
    tree = script_tree("smoke_trl_peft.py")
    assignments = {
        target.id: node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert "dpo_base_model" in assignments
    assert isinstance(assignments["dpo_base_model"], ast.Call)
    assert call_name(assignments["dpo_base_model"]) == "load_model"

    trainer = call_nodes("smoke_trl_peft.py", "DPOTrainer")
    assert len(trainer) == 1
    keywords = {item.arg: item.value for item in trainer[0].keywords if item.arg}
    assert isinstance(keywords["model"], ast.Name)
    assert keywords["model"].id == "dpo_base_model"


def test_legacy_loader_smoke_uses_real_shared_loader_and_checks_fa2() -> None:
    tree = script_tree("smoke_legacy_loader.py")

    assert "load_model_and_tokenizer" in {
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "post_train.scripts.sft.train_full"
        for alias in node.names
    }
    attributes = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
    }
    assert "_attn_implementation" in attributes
    assert "dtype" in attributes
    assert "is_floating_point" in attributes
    assert "grad" in attributes
    assert "bfloat16" in attributes


class CleanupCuda:
    def __init__(self) -> None:
        self.empty_cache_calls = 0

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1


class CleanupTorch:
    def __init__(self) -> None:
        self.cuda = CleanupCuda()

    @staticmethod
    def isfinite(value):
        return value


class CleanupModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(_attn_implementation="flash_attention_2")
        self.zero_grad_calls = 0

    def to(self, device):
        return self

    def train(self):
        return None

    def parameters(self):
        raise RuntimeError("controlled failure")

    def zero_grad(self, set_to_none=True):
        self.zero_grad_calls += 1


def test_legacy_loader_cleanup_runs_and_preserves_exception(monkeypatch) -> None:
    module = load_script("smoke_legacy_loader.py")
    model = CleanupModel()
    torch = CleanupTorch()
    gc_calls: list[bool] = []

    monkeypatch.setattr(
        module,
        "load_model_and_tokenizer",
        lambda model_path, gradient_checkpointing: (model, object()),
    )
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch":
            return torch
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(module.gc, "collect", lambda: gc_calls.append(True))

    with pytest.raises(RuntimeError, match="controlled failure"):
        module.run_smoke(Path("model"), "cuda:0", 16)

    assert model.zero_grad_calls == 1
    assert gc_calls == [True]
    assert torch.cuda.empty_cache_calls == 1


class FakeDeviceTensor:
    def __init__(self, values=None) -> None:
        self.values = values or [[1, 2]]
        self.moved_to = None

    def to(self, device):
        self.moved_to = str(device)
        return self

    def __getitem__(self, item):
        return self.values[item]

    @property
    def shape(self):
        return (len(self.values), len(self.values[0]))


class FakeEvalTokenizer:
    def __init__(self) -> None:
        self.input_ids = FakeDeviceTensor()
        self.attention_mask = FakeDeviceTensor([[1, 1]])

    def __call__(self, text, return_tensors):
        return {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
        }

    def decode(self, token_ids, skip_special_tokens):
        return "generated"


class FakeEvalModel:
    def __init__(self, hf_device_map=None, parameter_device="cuda:1") -> None:
        self.hf_device_map = hf_device_map
        self.parameter_device = parameter_device
        self.generate_kwargs = None

    def parameters(self):
        yield SimpleNamespace(device=self.parameter_device)

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return [[1, 2, 3]]


def test_eval_loader_smoke_prefers_hf_device_map_for_input_device(monkeypatch) -> None:
    module = load_script("smoke_eval_loader.py")
    tokenizer = FakeEvalTokenizer()
    model = FakeEvalModel(hf_device_map={"model.embed_tokens": "cuda:0"})

    monkeypatch.setattr(
        module,
        "load_model_and_tokenizer",
        lambda model_path, base_model_path=None: (tokenizer, model),
    )

    text = module.run_smoke(Path("full"), None, max_new_tokens=4)

    assert text == "generated"
    assert tokenizer.input_ids.moved_to == "cuda:0"
    assert tokenizer.attention_mask.moved_to == "cuda:0"
    assert model.generate_kwargs["max_new_tokens"] == 4
    assert model.generate_kwargs["do_sample"] is False


def test_eval_loader_smoke_falls_back_to_first_parameter_device(monkeypatch) -> None:
    module = load_script("smoke_eval_loader.py")
    tokenizer = FakeEvalTokenizer()
    model = FakeEvalModel(hf_device_map=None, parameter_device="cuda:1")

    monkeypatch.setattr(
        module,
        "load_model_and_tokenizer",
        lambda model_path, base_model_path=None: (tokenizer, model),
    )

    module.run_smoke(Path("full"), None, max_new_tokens=2)

    assert tokenizer.input_ids.moved_to == "cuda:1"


def test_eval_loader_smoke_rejects_empty_generation(monkeypatch) -> None:
    module = load_script("smoke_eval_loader.py")
    tokenizer = FakeEvalTokenizer()
    tokenizer.decode = lambda token_ids, skip_special_tokens: "   "
    model = FakeEvalModel(hf_device_map={"": "cuda:0"})
    monkeypatch.setattr(
        module,
        "load_model_and_tokenizer",
        lambda model_path, base_model_path=None: (tokenizer, model),
    )

    with pytest.raises(RuntimeError, match="empty generation"):
        module.run_smoke(Path("full"), None, max_new_tokens=2)


def test_eval_loader_adapter_mode_requires_base_model_path(tmp_path: Path) -> None:
    module = load_script("smoke_eval_loader.py")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="base-model-path"):
        module.validate_paths(adapter, None)


def test_eval_loader_adapter_mode_forwards_base_model_path(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = load_script("smoke_eval_loader.py")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    base = tmp_path / "base"
    tokenizer = FakeEvalTokenizer()
    model = FakeEvalModel(hf_device_map={"model.embed_tokens": "cuda:0"})
    calls = []

    def fake_loader(model_path, base_model_path=None):
        calls.append((model_path, base_model_path))
        return tokenizer, model

    monkeypatch.setattr(module, "load_model_and_tokenizer", fake_loader)

    module.run_smoke(adapter, base, max_new_tokens=2)

    assert calls == [(adapter, base)]
