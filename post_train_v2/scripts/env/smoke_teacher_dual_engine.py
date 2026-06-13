from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


REPORT_PREFIX = "TEACHER_REPORT_JSON="
OOM_MARKERS = ("out of memory", "cuda oom", "cudaerroroutofmemory")
SIGTERM = getattr(signal, "SIGTERM", 15)
SIGKILL = getattr(signal, "SIGKILL", 9)
REPORT_SCHEMA = {
    "launch_index": int,
    "cuda_visible_devices": str,
    "device_count": int,
    "current_device": int,
    "cuda_uuid": str,
    "cuda_pci_bus_id": str,
    "memory_before": dict,
    "memory_after": dict,
    "output": str,
}
MEMORY_SCHEMA = {
    "name": str,
    "free_bytes": int,
    "total_bytes": int,
}


class CudaUuid(ctypes.Structure):
    _fields_ = [("bytes", ctypes.c_ubyte * 16)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run two concurrent TP1 Qwen3-8B vLLM engines on isolated GPUs."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--launch-index", type=int, choices=(0, 1), help=argparse.SUPPRESS)
    return parser.parse_args()


def parse_nvidia_smi_output(output: str) -> dict[int, dict[str, str]]:
    inventory: dict[int, dict[str, str]] = {}
    for line_number, raw_line in enumerate(output.splitlines(), start=1):
        if not raw_line.strip():
            continue
        parts = [part.strip() for part in raw_line.split(",")]
        if len(parts) != 3 or not all(parts):
            raise ValueError(f"malformed nvidia-smi line {line_number}: {raw_line!r}")
        try:
            index = int(parts[0])
        except ValueError as exc:
            raise ValueError(
                f"invalid nvidia-smi GPU index on line {line_number}: {parts[0]!r}"
            ) from exc
        if index in inventory:
            raise ValueError(f"duplicate nvidia-smi GPU index: {index}")
        inventory[index] = {"uuid": parts[1], "pci_bus_id": parts[2]}
    if not inventory:
        raise ValueError("nvidia-smi returned no GPU inventory")
    return inventory


def query_physical_gpus(
    run_command=subprocess.run,
) -> dict[int, dict[str, str]]:
    result = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,pci.bus_id",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "nvidia-smi inventory query failed: "
            + (result.stderr.strip() or f"exit {result.returncode}")
        )
    return parse_nvidia_smi_output(result.stdout)


def load_libcudart():
    candidates = [
        ctypes.util.find_library("cudart"),
        "libcudart.so",
        "libcudart.so.12",
    ]
    errors: list[str] = []
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return ctypes.CDLL(candidate)
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")
    raise OSError("; ".join(errors) or "libcudart was not found")


def format_cuda_uuid(raw: bytes) -> str:
    hexadecimal = raw.hex()
    return "GPU-" + "-".join(
        (
            hexadecimal[0:8],
            hexadecimal[8:12],
            hexadecimal[12:16],
            hexadecimal[16:20],
            hexadecimal[20:32],
        )
    )


def query_cuda_identity(
    device: int = 0,
    cudart_loader=load_libcudart,
) -> dict[str, str]:
    try:
        cudart = cudart_loader()
    except OSError as exc:
        raise RuntimeError(f"failed to load libcudart: {exc}") from exc

    uuid = CudaUuid()
    pci_buffer = ctypes.create_string_buffer(32)
    try:
        get_uuid = cudart.cudaDeviceGetUuid
        get_pci = cudart.cudaDeviceGetPCIBusId
        if hasattr(get_uuid, "argtypes"):
            get_uuid.argtypes = [ctypes.POINTER(CudaUuid), ctypes.c_int]
            get_uuid.restype = ctypes.c_int
        if hasattr(get_pci, "argtypes"):
            get_pci.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
            get_pci.restype = ctypes.c_int
        uuid_status = get_uuid(ctypes.byref(uuid), device)
        pci_status = get_pci(pci_buffer, len(pci_buffer), device)
    except (AttributeError, TypeError) as exc:
        raise RuntimeError(f"libcudart identity API is unavailable: {exc}") from exc
    if uuid_status != 0 or pci_status != 0:
        raise RuntimeError(
            "libcudart failed to query CUDA identity: "
            f"cudaDeviceGetUuid={uuid_status}, cudaDeviceGetPCIBusId={pci_status}"
        )

    pci_bus_id = pci_buffer.value.decode("ascii", errors="strict").strip()
    if not pci_bus_id:
        raise RuntimeError("libcudart returned an empty CUDA PCI bus ID")
    return {
        "cuda_uuid": format_cuda_uuid(bytes(uuid.bytes)),
        "cuda_pci_bus_id": pci_bus_id,
    }


def build_child_command(
    python_executable: str,
    script_path: Path,
    model_path: str,
    gpu_memory_utilization: float,
    launch_index: int,
) -> list[str]:
    return [
        python_executable,
        str(script_path),
        "--child",
        "--model-path",
        model_path,
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--launch-index",
        str(launch_index),
    ]


def build_child_environment(
    base_environment: Mapping[str, str],
    launch_index: int,
) -> dict[str, str]:
    environment = dict(base_environment)
    environment["CUDA_VISIBLE_DEVICES"] = str(launch_index)
    return environment


def parse_child_report(stdout: str) -> dict[str, Any]:
    report_lines = [
        line[len(REPORT_PREFIX) :]
        for line in stdout.splitlines()
        if line.startswith(REPORT_PREFIX)
    ]
    if len(report_lines) != 1:
        raise ValueError(
            f"expected one structured child report, found {len(report_lines)}"
        )
    try:
        report = json.loads(report_lines[0])
    except json.JSONDecodeError as exc:
        raise ValueError("child report is not valid JSON") from exc
    if not isinstance(report, dict):
        raise ValueError("child report must be a JSON object")
    validate_exact_schema(report, REPORT_SCHEMA, "child report")
    validate_exact_schema(report["memory_before"], MEMORY_SCHEMA, "memory_before")
    validate_exact_schema(report["memory_after"], MEMORY_SCHEMA, "memory_after")
    return report


def validate_exact_schema(
    value: Mapping[str, Any],
    schema: Mapping[str, type],
    label: str,
) -> None:
    if type(value) is not dict:
        raise ValueError(f"{label} must be an object")
    missing = set(schema) - set(value)
    unknown = set(value) - set(schema)
    if missing:
        raise ValueError(f"{label} missing fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{label} has unknown fields: {sorted(unknown)}")
    for field, expected_type in schema.items():
        if type(value[field]) is not expected_type:
            raise ValueError(
                f"{label}.{field} must have exact type "
                f"{expected_type.__name__}, got {type(value[field]).__name__}"
            )


def normalize_uuid(value: str) -> str:
    return value.removeprefix("GPU-").replace("-", "").lower()


def normalize_pci_bus_id(value: str) -> tuple[int, int, int, int]:
    try:
        domain_bus, device_function = value.lower().split(":", maxsplit=1)
        bus, device_function = device_function.split(":", maxsplit=1)
        device, function = device_function.split(".", maxsplit=1)
        return (
            int(domain_bus, 16),
            int(bus, 16),
            int(device, 16),
            int(function, 16),
        )
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"invalid PCI bus ID: {value!r}") from exc


def validate_teacher_reports(
    reports: Sequence[dict[str, Any]],
    physical_gpus: Mapping[int, Mapping[str, str]],
) -> list[dict[str, Any]]:
    if len(reports) != 2:
        raise ValueError(f"expected two teacher reports, got {len(reports)}")

    validated: list[dict[str, Any]] = []
    seen_launch_indices: set[int] = set()
    for report in sorted(reports, key=lambda item: item.get("launch_index", -1)):
        validate_exact_schema(report, REPORT_SCHEMA, "child report")
        validate_exact_schema(report["memory_before"], MEMORY_SCHEMA, "memory_before")
        validate_exact_schema(report["memory_after"], MEMORY_SCHEMA, "memory_after")
        launch_index = report.get("launch_index")
        if launch_index not in (0, 1) or launch_index in seen_launch_indices:
            raise ValueError(f"unmappable teacher launch index: {launch_index!r}")
        seen_launch_indices.add(launch_index)
        if report.get("cuda_visible_devices") != str(launch_index):
            raise ValueError(
                f"device isolation violation for child {launch_index}: "
                f"CUDA_VISIBLE_DEVICES={report.get('cuda_visible_devices')!r}"
            )
        if report.get("device_count") != 1 or report.get("current_device") != 0:
            raise ValueError(
                f"device isolation violation for child {launch_index}: "
                f"count={report.get('device_count')!r}, "
                f"current={report.get('current_device')!r}"
            )
        if not str(report.get("output", "")).strip():
            raise ValueError(f"teacher child {launch_index} returned empty output")

        matches = [
            (index, identity)
            for index, identity in physical_gpus.items()
            if normalize_uuid(identity["uuid"]) == normalize_uuid(report["cuda_uuid"])
            and normalize_pci_bus_id(identity["pci_bus_id"])
            == normalize_pci_bus_id(report["cuda_pci_bus_id"])
        ]
        if len(matches) != 1:
            raise ValueError(
                f"teacher child {launch_index} CUDA identity cannot be mapped "
                "to exactly one nvidia-smi GPU"
            )
        enriched = dict(report)
        enriched["nvidia_smi_index"] = matches[0][0]
        validated.append(enriched)

    uuids = {normalize_uuid(report["cuda_uuid"]) for report in validated}
    pci_ids = {normalize_pci_bus_id(report["cuda_pci_bus_id"]) for report in validated}
    if len(uuids) != 2 or len(pci_ids) != 2:
        raise ValueError("teacher children do not map to distinct physical GPU identities")
    return validated


def terminate_children(
    processes: Sequence[subprocess.Popen[str]],
    os_name: str = os.name,
    killpg=os.killpg if hasattr(os, "killpg") else None,
    wait_timeout: float = 5.0,
) -> None:
    running = [process for process in processes if process.poll() is None]
    process_groups: set[int] = set()
    for process in running:
        if os_name == "posix" and killpg is not None:
            process_groups.add(process.pid)
            try:
                killpg(process.pid, SIGTERM)
            except ProcessLookupError:
                pass
        else:
            process.terminate()

    timed_out: list[subprocess.Popen[str]] = []
    for process in running:
        try:
            process.wait(timeout=wait_timeout)
        except subprocess.TimeoutExpired:
            timed_out.append(process)

    for process in timed_out:
        if process.pid in process_groups and killpg is not None:
            try:
                killpg(process.pid, SIGKILL)
            except ProcessLookupError:
                pass
        else:
            process.kill()
        process.wait()


def run_teacher_pair(
    model_path: str,
    gpu_memory_utilization: float,
    timeout_seconds: float,
    physical_gpus: Mapping[int, Mapping[str, str]],
    popen_factory=subprocess.Popen,
    monotonic=time.monotonic,
    os_name: str = os.name,
    terminate=None,
) -> list[dict[str, Any]]:
    script_path = Path(__file__).resolve()
    processes: list[subprocess.Popen[str]] = []
    try:
        for launch_index in (0, 1):
            popen_kwargs = {
                "env": build_child_environment(os.environ, launch_index),
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
                "text": True,
            }
            if os_name == "posix":
                popen_kwargs["start_new_session"] = True
            processes.append(
                popen_factory(
                    build_child_command(
                        sys.executable,
                        script_path,
                        model_path,
                        gpu_memory_utilization,
                        launch_index,
                    ),
                    **popen_kwargs,
                )
            )

        deadline = monotonic() + timeout_seconds
        reports: list[dict[str, Any]] = []
        for launch_index, process in enumerate(processes):
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise TimeoutError("teacher dual-engine smoke timed out")
            try:
                stdout, stderr = process.communicate(timeout=remaining)
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError(
                    f"teacher child {launch_index} timed out"
                ) from exc

            combined = f"{stdout}\n{stderr}".lower()
            if any(marker in combined for marker in OOM_MARKERS):
                raise RuntimeError(f"teacher child {launch_index} reported CUDA OOM")
            if process.returncode != 0:
                raise RuntimeError(
                    f"teacher child {launch_index} failed with "
                    f"exit {process.returncode}: {stderr.strip()}"
                )
            reports.append(parse_child_report(stdout))

        return validate_teacher_reports(reports, physical_gpus)
    finally:
        original_error_active = sys.exc_info()[0] is not None
        try:
            if terminate is not None:
                terminate(processes)
            else:
                terminate_children(processes, os_name=os_name)
        except Exception:
            if not original_error_active:
                raise


def run_parent(args: argparse.Namespace) -> int:
    physical_gpus = query_physical_gpus()
    validated = run_teacher_pair(
        model_path=args.model_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        timeout_seconds=args.timeout_seconds,
        physical_gpus=physical_gpus,
    )
    print(json.dumps(validated, indent=2, sort_keys=True))
    print("OK: two isolated Teacher vLLM engines used distinct physical GPUs")
    return 0


def memory_snapshot(torch_module: Any, device: int = 0) -> dict[str, Any]:
    free_bytes, total_bytes = torch_module.cuda.mem_get_info(device)
    return {
        "name": torch_module.cuda.get_device_name(device),
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
    }


def run_child(args: argparse.Namespace) -> int:
    if args.launch_index not in (0, 1):
        raise ValueError("--launch-index is required in child mode")

    import torch
    from vllm import LLM, SamplingParams

    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"child expected exactly one visible CUDA device, found "
            f"{torch.cuda.device_count()}"
        )
    torch.cuda.set_device(0)
    identity = query_cuda_identity(device=0)
    before = memory_snapshot(torch)
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=256,
    )
    after = memory_snapshot(torch)
    outputs = llm.chat(
        [[{
            "role": "user",
            "content": (
                "Using the numbers [1, 1, 1, 1], create an equation that "
                "equals 4. Return only <answer> equation </answer>."
            ),
        }]],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=32),
        chat_template_kwargs={"enable_thinking": False},
    )
    text = outputs[0].outputs[0].text.strip()
    if not text:
        raise RuntimeError("teacher child returned empty output")

    report = {
        "launch_index": args.launch_index,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_count": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device(),
        **identity,
        "memory_before": before,
        "memory_after": after,
        "output": text,
    }
    print(REPORT_PREFIX + json.dumps(report, sort_keys=True))
    return 0


def main() -> int:
    args = parse_args()
    return run_child(args) if args.child else run_parent(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        ImportError,
        OSError,
        RuntimeError,
        TimeoutError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
