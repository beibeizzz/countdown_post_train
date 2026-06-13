import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "env" / "verify_artifacts.py"


def load_module():
    spec = importlib.util.spec_from_file_location("verify_artifacts", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_manifest_artifacts(path: Path, artifacts: dict[str, tuple[str, str]]) -> None:
    path.write_text(
        json.dumps(
            {
                "artifacts": {
                    name: {
                        "filename": filename,
                        "url": "https://example.invalid/test.whl",
                        "sha256": sha256,
                    }
                    for name, (filename, sha256) in artifacts.items()
                }
            }
        ),
        encoding="utf-8",
    )


def write_manifest(path: Path, filename: str, sha256: str) -> None:
    write_manifest_artifacts(path, {"test-wheel": (filename, sha256)})


def write_raw_manifest(path: Path, metadata: object) -> None:
    path.write_text(
        json.dumps({"artifacts": {"test-wheel": metadata}}),
        encoding="utf-8",
    )


def test_verify_artifacts_accepts_exact_filename_and_hash(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = load_module()
    wheels_dir = tmp_path / "wheels"
    wheels_dir.mkdir()
    artifact = wheels_dir / "exact.whl"
    artifact.write_bytes(b"verified wheel")
    expected_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = tmp_path / "runtime.json"
    write_manifest(manifest, artifact.name, expected_hash)

    results = module.verify_artifacts(manifest, wheels_dir)

    assert results == [(artifact.name, expected_hash)]
    assert capsys.readouterr().out.strip() == f"OK {artifact.name} {expected_hash}"


def test_verify_artifacts_prints_one_ordered_ok_line_per_artifact(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = load_module()
    wheels_dir = tmp_path / "wheels"
    wheels_dir.mkdir()
    artifacts = {
        "vllm": ("vllm.whl", b"vllm wheel"),
        "flash-attn": ("flash_attn.whl", b"flash attention wheel"),
    }
    manifest_artifacts: dict[str, tuple[str, str]] = {}
    expected_results: list[tuple[str, str]] = []
    expected_lines: list[str] = []
    for name, (filename, contents) in artifacts.items():
        path = wheels_dir / filename
        path.write_bytes(contents)
        sha256 = hashlib.sha256(contents).hexdigest()
        manifest_artifacts[name] = (filename, sha256)
        expected_results.append((filename, sha256))
        expected_lines.append(f"OK {filename} {sha256}")
    manifest = tmp_path / "runtime.json"
    write_manifest_artifacts(manifest, manifest_artifacts)

    results = module.verify_artifacts(manifest, wheels_dir)

    assert results == expected_results
    assert capsys.readouterr().out.splitlines() == expected_lines


def test_verify_artifacts_rejects_missing_file(tmp_path: Path) -> None:
    module = load_module()
    wheels_dir = tmp_path / "wheels"
    wheels_dir.mkdir()
    manifest = tmp_path / "runtime.json"
    write_manifest(manifest, "missing.whl", "0" * 64)

    with pytest.raises(FileNotFoundError, match="missing.whl"):
        module.verify_artifacts(manifest, wheels_dir)


def test_verify_artifacts_rejects_wrong_hash(tmp_path: Path) -> None:
    module = load_module()
    wheels_dir = tmp_path / "wheels"
    wheels_dir.mkdir()
    artifact = wheels_dir / "wrong.whl"
    artifact.write_bytes(b"wrong contents")
    manifest = tmp_path / "runtime.json"
    write_manifest(manifest, artifact.name, "0" * 64)

    with pytest.raises(ValueError, match="SHA-256 mismatch.*wrong.whl"):
        module.verify_artifacts(manifest, wheels_dir)


@pytest.mark.parametrize("metadata", [None, [], "artifact"])
def test_verify_artifacts_rejects_non_mapping_metadata(
    tmp_path: Path,
    metadata: object,
) -> None:
    module = load_module()
    manifest = tmp_path / "runtime.json"
    write_raw_manifest(manifest, metadata)

    with pytest.raises(ValueError, match="metadata must be an object"):
        module.verify_artifacts(manifest, tmp_path)


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ({"url": "https://example.invalid/a", "sha256": "0" * 64}, "filename"),
        ({"filename": "a.whl", "sha256": "0" * 64}, "URL"),
        ({"filename": "a.whl", "url": "https://example.invalid/a"}, "SHA-256"),
        ({"filename": 1, "url": "https://example.invalid/a", "sha256": "0" * 64}, "filename"),
        ({"filename": "a.whl", "url": 1, "sha256": "0" * 64}, "URL"),
        ({"filename": "a.whl", "url": "https://example.invalid/a", "sha256": 1}, "SHA-256"),
        ({"filename": "a.whl", "url": "https://example.invalid/a", "sha256": "g" * 64}, "SHA-256"),
        ({"filename": "a.whl", "url": "https://example.invalid/a", "sha256": "0" * 63}, "SHA-256"),
    ],
)
def test_verify_artifacts_validates_metadata_fields(
    tmp_path: Path,
    metadata: dict[str, object],
    message: str,
) -> None:
    module = load_module()
    manifest = tmp_path / "runtime.json"
    write_raw_manifest(manifest, metadata)

    with pytest.raises(ValueError, match=message):
        module.verify_artifacts(manifest, tmp_path)


@pytest.mark.parametrize(
    "filename",
    [
        "nested/artifact.whl",
        r"nested\artifact.whl",
        "../artifact.whl",
        "/tmp/artifact.whl",
        r"C:\tmp\artifact.whl",
    ],
)
def test_verify_artifacts_rejects_non_basename_filename(
    tmp_path: Path,
    filename: str,
) -> None:
    module = load_module()
    manifest = tmp_path / "runtime.json"
    write_manifest(manifest, filename, "0" * 64)

    with pytest.raises(ValueError, match="invalid exact filename"):
        module.verify_artifacts(manifest, tmp_path)


def test_verify_artifacts_rejects_symlink(tmp_path: Path) -> None:
    module = load_module()
    wheels_dir = tmp_path / "wheels"
    wheels_dir.mkdir()
    target = tmp_path / "target.whl"
    target.write_bytes(b"target")
    artifact = wheels_dir / "artifact.whl"
    try:
        artifact.symlink_to(target)
    except OSError as error:
        pytest.skip(f"symlinks unavailable: {error}")
    expected_hash = hashlib.sha256(target.read_bytes()).hexdigest()
    manifest = tmp_path / "runtime.json"
    write_manifest(manifest, artifact.name, expected_hash)

    with pytest.raises(ValueError, match="symlink"):
        module.verify_artifacts(manifest, wheels_dir)


@pytest.mark.parametrize("failure", ["missing", "wrong-hash"])
def test_cli_returns_nonzero_for_invalid_artifact(
    tmp_path: Path,
    failure: str,
) -> None:
    wheels_dir = tmp_path / "wheels"
    wheels_dir.mkdir()
    filename = "artifact.whl"
    expected_hash = "0" * 64
    if failure == "wrong-hash":
        (wheels_dir / filename).write_bytes(b"not the expected artifact")
    manifest = tmp_path / "runtime.json"
    write_manifest(manifest, filename, expected_hash)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--manifest",
            str(manifest),
            "--wheels-dir",
            str(wheels_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert filename in result.stderr


def test_cli_returns_zero_for_valid_artifact(tmp_path: Path) -> None:
    wheels_dir = tmp_path / "wheels"
    wheels_dir.mkdir()
    artifact = wheels_dir / "artifact.whl"
    artifact.write_bytes(b"valid artifact")
    expected_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = tmp_path / "runtime.json"
    write_manifest(manifest, artifact.name, expected_hash)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--manifest",
            str(manifest),
            "--wheels-dir",
            str(wheels_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == f"OK {artifact.name} {expected_hash}"
    assert result.stderr == ""


def test_cli_invalid_metadata_returns_one_without_traceback(tmp_path: Path) -> None:
    manifest = tmp_path / "runtime.json"
    write_raw_manifest(manifest, {"filename": "artifact.whl"})

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--manifest",
            str(manifest),
            "--wheels-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert result.stderr.startswith("ERROR: ")
    assert "Traceback" not in result.stderr


def test_cli_help_documents_required_paths() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--manifest" in result.stdout
    assert "--wheels-dir" in result.stdout
