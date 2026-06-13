import json
import re
import tomllib
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "configs" / "environment" / "runtime-cu128.json"
PYPROJECT_PATH = ROOT / "pyproject.toml"
CONSTRAINTS_PATH = ROOT / "constraints-verl060-vllm091-cu128.txt"
REQUIREMENTS_PATH = ROOT / "requirements-runtime.txt"
OLD_CONSTRAINTS_PATH = ROOT / "constraints-verl071-vllm017-cu129.txt"

OLD_BASELINE_TOKENS = (
    "constraints-verl071-vllm017-cu129",
    "pytorch-cu129",
    "cu129",
    "verl==0.7.1",
    "vllm==0.17.0",
    "torch==2.10.0",
    "flash-attn==2.8.3",
)

ACTIVE_ENVIRONMENT_DOCS = (
    ROOT / "README.md",
    ROOT / "environment.md",
    ROOT / "migration_plan.md",
    ROOT / "open_questions.md",
    ROOT / "docs" / "environment_setup.md",
    ROOT / "wheels" / "README.md",
    ROOT / "scripts" / "env" / "README.md",
)

OLD_DOCUMENTATION_PHRASES = (
    "verl 0.7.1",
    "vllm 0.17.0",
    "torch 2.10",
    "cu129",
    "flash attention 2.8.3",
    "constraints-verl071-vllm017-cu129.txt",
)


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def load_pyproject() -> dict:
    with PYPROJECT_PATH.open("rb") as handle:
        return tomllib.load(handle)


def normalized_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_exact_requirements(
    lines: list[str],
) -> dict[str, tuple[frozenset[str], str]]:
    parsed: dict[str, tuple[frozenset[str], str]] = {}
    for raw_line in lines:
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        match = re.fullmatch(
            r"(?P<name>[A-Za-z0-9_.-]+)"
            r"(?:\[(?P<extras>[A-Za-z0-9_.,-]+)\])?"
            r"==(?P<version>[^\s]+)",
            line,
        )
        assert match is not None, f"requirement is not exactly pinned: {line}"
        name = normalized_name(match.group("name"))
        assert name not in parsed, f"duplicate normalized requirement: {name}"
        extras_text = match.group("extras")
        extra_items = extras_text.split(",") if extras_text else []
        assert len(extra_items) == len(set(extra_items)), (
            f"duplicate extras for {name}: {extra_items}"
        )
        extras = frozenset(extra_items)
        parsed[name] = (extras, match.group("version"))
    return parsed


def read_requirement_file(
    path: Path,
) -> dict[str, tuple[frozenset[str], str]]:
    return parse_exact_requirements(path.read_text(encoding="utf-8").splitlines())


def expected_pins(packages: dict[str, str]) -> dict[str, tuple[frozenset[str], str]]:
    return {
        normalized_name(name): (
            frozenset({"default", "cgraph"}) if normalized_name(name) == "ray" else frozenset(),
            version,
        )
        for name, version in packages.items()
    }


def assert_allowed_extras(
    requirements: dict[str, tuple[frozenset[str], str]],
) -> None:
    for name, (extras, _) in requirements.items():
        expected = frozenset({"default", "cgraph"}) if name == "ray" else frozenset()
        assert extras == expected, f"invalid extras for {name}: {sorted(extras)}"
    assert requirements["verl"] == (frozenset(), "0.6.0")


def test_requirement_parser_rejects_duplicate_normalized_packages() -> None:
    with pytest.raises(AssertionError, match="duplicate normalized requirement"):
        parse_exact_requirements(["flash-attn==1", "flash_attn==1"])


def test_requirement_parser_preserves_extras() -> None:
    assert parse_exact_requirements(["ray[default,cgraph]==2.48.0"]) == {
        "ray": (frozenset({"default", "cgraph"}), "2.48.0")
    }


def test_requirement_parser_rejects_duplicate_extras() -> None:
    with pytest.raises(AssertionError, match="duplicate extras for ray"):
        parse_exact_requirements(["ray[default,default,cgraph]==2.48.0"])


def test_only_ray_default_cgraph_extras_are_allowed() -> None:
    assert_allowed_extras(
        parse_exact_requirements(
            ["ray[default,cgraph]==2.48.0", "verl==0.6.0", "torch==2.7.0"]
        )
    )
    with pytest.raises(AssertionError, match="invalid extras for torch"):
        assert_allowed_extras(
            parse_exact_requirements(
                ["ray[default,cgraph]==2.48.0", "verl==0.6.0", "torch[cuda]==2.7.0"]
            )
        )


def test_manifest_locks_all_direct_and_dev_packages() -> None:
    data = load_manifest()
    assert data["schema_version"] == 1
    assert data["python"] == "3.11.15"
    assert data["cuda_runtime"] == "12.8"
    assert data["packages"]["torch"] == "2.7.0"
    assert data["packages"]["flash-attn"] == "2.7.4.post1"
    assert data["packages"]["vllm"] == "0.9.1"
    assert data["packages"]["verl"] == "0.6.0"
    assert data["dev_packages"] == {"pytest": "8.3.5"}
    assert all(version for version in data["packages"].values())


def test_manifest_locks_exact_official_artifacts() -> None:
    artifacts = load_manifest()["artifacts"]
    assert artifacts["vllm"] == {
        "filename": "vllm-0.9.1-cp38-abi3-manylinux1_x86_64.whl",
        "url": (
            "https://github.com/vllm-project/vllm/releases/download/v0.9.1/"
            "vllm-0.9.1-cp38-abi3-manylinux1_x86_64.whl"
        ),
        "sha256": (
            "28b99e8df39c7aaeda04f7e5353b18564a1a9d1c579691945523fc4777a1a8c8"
        ),
    }
    assert artifacts["flash-attn"] == {
        "filename": (
            "flash_attn-2.7.4.post1+cu12torch2.7cxx11abiTRUE-"
            "cp311-cp311-linux_x86_64.whl"
        ),
        "url": (
            "https://github.com/Dao-AILab/flash-attention/releases/download/"
            "v2.7.4.post1/flash_attn-2.7.4.post1%2Bcu12torch2.7"
            "cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
        ),
        "sha256": (
            "22013b8c74a63fc70e69be1e10ff02e4ad8fec84a43600bdca67b434ed417113"
        ),
    }


def test_manifest_forbids_conflicting_verl_extras() -> None:
    assert load_manifest()["forbidden_requirements"] == [
        "verl[gpu]",
        "verl[trl]",
        "verl[vllm]",
    ]


def test_pyproject_matches_manifest_and_uses_local_artifacts() -> None:
    manifest = load_manifest()
    pyproject = load_pyproject()
    project = pyproject["project"]
    dependencies = parse_exact_requirements(project["dependencies"])
    expected = expected_pins(manifest["packages"])
    assert dependencies == expected
    assert project["requires-python"] == "==3.11.15"
    assert_allowed_extras(dependencies)

    dev_dependencies = parse_exact_requirements(
        pyproject["dependency-groups"]["dev"]
    )
    assert dev_dependencies == expected_pins(manifest["dev_packages"])
    assert dependencies.keys().isdisjoint(dev_dependencies)
    assert_allowed_extras({**dependencies, **dev_dependencies})

    indexes = {item["name"]: item for item in pyproject["tool"]["uv"]["index"]}
    assert indexes["pytorch-cu128"] == {
        "name": "pytorch-cu128",
        "url": "https://download.pytorch.org/whl/cu128",
        "explicit": True,
    }

    sources = pyproject["tool"]["uv"]["sources"]
    for package in ("torch", "torchvision", "torchaudio"):
        assert sources[package] == {"index": "pytorch-cu128"}
    assert sources["vllm"] == {
        "path": (
            "wheels/vllm-0.9.1-cp38-abi3-manylinux1_x86_64.whl"
        )
    }
    assert sources["flash-attn"] == {
        "path": (
            "wheels/flash_attn-2.7.4.post1+cu12torch2.7"
            "cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
        )
    }

    raw_dependencies = "\n".join(project["dependencies"]).lower()
    assert "verl[" not in raw_dependencies


def test_pyproject_allows_only_base_verl_060() -> None:
    dependencies = parse_exact_requirements(
        load_pyproject()["project"]["dependencies"]
    )
    assert dependencies["verl"] == (frozenset(), "0.6.0")


def test_constraints_match_all_manifest_pins() -> None:
    manifest = load_manifest()
    constraints = read_requirement_file(CONSTRAINTS_PATH)
    expected = expected_pins(
        {**manifest["packages"], **manifest["dev_packages"]}
    )
    assert constraints == expected
    assert_allowed_extras(constraints)


def test_constraints_allow_only_base_verl_060() -> None:
    constraints = read_requirement_file(CONSTRAINTS_PATH)
    assert constraints["verl"] == (frozenset(), "0.6.0")


def test_runtime_requirements_match_runtime_manifest_pins() -> None:
    manifest = load_manifest()
    requirements = read_requirement_file(REQUIREMENTS_PATH)
    expected = expected_pins(manifest["packages"])
    assert requirements == expected
    assert_allowed_extras(requirements)
    assert "pytest" not in requirements


def test_runtime_requirements_allow_only_base_verl_060() -> None:
    requirements = read_requirement_file(REQUIREMENTS_PATH)
    assert requirements["verl"] == (frozenset(), "0.6.0")


def test_dependency_files_contain_no_old_baseline() -> None:
    assert not OLD_CONSTRAINTS_PATH.exists()
    active_files = (
        PYPROJECT_PATH,
        CONSTRAINTS_PATH,
        REQUIREMENTS_PATH,
        MANIFEST_PATH,
    )
    for path in active_files:
        text = path.read_text(encoding="utf-8").lower()
        for token in OLD_BASELINE_TOKENS:
            assert token.lower() not in text, f"{token!r} remains in {path}"


def test_active_environment_docs_contain_no_old_baseline() -> None:
    for path in ACTIVE_ENVIRONMENT_DOCS:
        text = path.read_text(encoding="utf-8").lower()
        for phrase in OLD_DOCUMENTATION_PHRASES:
            assert phrase.lower() not in text, f"{phrase!r} remains in {path}"


def test_environment_runbook_returns_to_repository_root_before_static_tests() -> None:
    runbook = (ROOT / "docs" / "environment_setup.md").read_text(encoding="utf-8")
    static_section = runbook.split("## 6. Run Static Tests", maxsplit=1)[1]
    command_block = static_section.split("## 7.", maxsplit=1)[0]

    assert "cd .." in command_block
    assert command_block.index("cd ..") < command_block.index(
        "python -m pytest -q post_train_v2/tests/env"
    )


def test_design_does_not_claim_remote_level_one_was_executed() -> None:
    design = (
        ROOT
        / "docs"
        / "superpowers"
        / "specs"
        / "2026-06-12-flash-attention-environment-design.md"
    ).read_text(encoding="utf-8")

    assert "This level is implemented and executed" not in design
