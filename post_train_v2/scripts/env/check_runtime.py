from __future__ import annotations

import argparse
import json
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable


DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "environment"
    / "runtime-cu128.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the pinned training runtime.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--require-gpus", type=int, default=2)
    parser.add_argument("--check-ray", action="store_true")
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError("runtime manifest must be a JSON object")
    return manifest


def report_peer_access(
    peer_access: bool,
    ipc_available: bool,
    writer: Callable[[str], None] = print,
) -> int:
    if peer_access and ipc_available:
        writer("OK: CUDA peer access and IPC are available")
    else:
        writer(
            "WARNING: CUDA P2P/IPC is not fully exposed; "
            "HAMI may still support NCCL and vLLM correctly"
        )
    return 0


def probe_cuda_ipc(tensor: Any) -> bool:
    try:
        storage = tensor.untyped_storage()
        share_cuda = getattr(storage, "_share_cuda_", None)
        return share_cuda is not None and bool(share_cuda())
    except (AttributeError, RuntimeError):
        return False


def check_package_versions(packages: dict[str, str]) -> None:
    from packaging.version import Version

    errors: list[str] = []
    for package_name, expected in sorted(packages.items()):
        try:
            installed = version(package_name)
        except PackageNotFoundError:
            errors.append(f"{package_name}: not installed")
            continue
        if Version(installed).base_version != Version(expected).base_version:
            errors.append(
                f"{package_name}: expected {expected}, installed {installed}"
            )
        else:
            print(f"OK: {package_name}=={installed}")
    if errors:
        raise RuntimeError("\n".join(errors))


def check_torch(
    manifest: dict[str, Any],
    require_gpus: int,
    torch_module: Any | None = None,
    writer: Callable[[str], None] = print,
    ipc_probe: Callable[[Any], bool] = probe_cuda_ipc,
) -> None:
    if torch_module is None:
        import torch as torch_module

    expected_base = manifest["packages"]["torch"]
    expected_full = f"{expected_base}+cu128"
    if torch_module.__version__ != expected_full:
        raise RuntimeError(
            "torch: expected complete version "
            f"{expected_full}, got {torch_module.__version__}"
        )
    if torch_module.version.cuda != manifest["cuda_runtime"]:
        raise RuntimeError(
            "torch CUDA: expected "
            f"{manifest['cuda_runtime']}, got {torch_module.version.cuda}"
        )
    abi = bool(torch_module._C._GLIBCXX_USE_CXX11_ABI)
    if not abi:
        raise RuntimeError("torch must use CXX11 ABI")
    if not torch_module.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    device_count = torch_module.cuda.device_count()
    if device_count != require_gpus:
        raise RuntimeError(
            f"expected exactly {require_gpus} CUDA devices, found {device_count}"
        )

    writer(
        f"OK: torch={torch_module.__version__} cuda={torch_module.version.cuda} "
        f"cxx11_abi={abi}"
    )
    expected_capability = tuple(
        int(part)
        for part in manifest["platform"]["gpu_compute_capability"].split(".")
    )
    for index in range(device_count):
        properties = torch_module.cuda.get_device_properties(index)
        actual_capability = (properties.major, properties.minor)
        if actual_capability != expected_capability:
            raise RuntimeError(
                f"GPU {index} compute capability must be "
                f"{expected_capability[0]}.{expected_capability[1]}, got "
                f"{properties.major}.{properties.minor}"
            )
        if "A100" not in properties.name.upper():
            raise RuntimeError(
                f"GPU {index} name must contain A100, got {properties.name!r}"
            )
        with torch_module.cuda.device(index):
            free_bytes, total_bytes = torch_module.cuda.mem_get_info()
        writer(
            "GPU "
            f"{index}: name={properties.name!r} "
            f"compute_capability={properties.major}.{properties.minor} "
            f"total_gib={total_bytes / 2**30:.2f} "
            f"free_gib={free_bytes / 2**30:.2f}"
        )

    peer_access = device_count >= 2 and all(
        torch_module.cuda.can_device_access_peer(source, target)
        for source in range(device_count)
        for target in range(device_count)
        if source != target
    )
    ipc_tensor = torch_module.ones(1, device="cuda:0")
    ipc_available = ipc_probe(ipc_tensor)
    report_peer_access(peer_access, ipc_available, writer=writer)


def check_ray_resources(
    require_gpus: int,
    ray_module: Any | None = None,
    writer: Callable[[str], None] = print,
) -> None:
    if ray_module is None:
        import ray as ray_module

    started_here = False
    try:
        if not ray_module.is_initialized():
            ray_module.init(include_dashboard=False, ignore_reinit_error=False)
            started_here = True
        gpu_resources = float(ray_module.cluster_resources().get("GPU", 0.0))
        if gpu_resources != float(require_gpus):
            raise RuntimeError(
                f"Ray expected {require_gpus} GPU resources, found {gpu_resources}"
            )
        writer(f"OK: Ray reports {gpu_resources:g} GPU resources")
    finally:
        if started_here and ray_module.is_initialized():
            ray_module.shutdown()


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    expected_python = tuple(int(part) for part in manifest["python"].split("."))
    if sys.version_info[:3] != expected_python:
        raise RuntimeError(
            f"Python must be {manifest['python']}, got "
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        )

    check_package_versions(manifest["packages"])
    check_torch(manifest, args.require_gpus)
    if args.check_ray:
        check_ray_resources(args.require_gpus)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
