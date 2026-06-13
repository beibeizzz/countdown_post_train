from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = (
    REPO_ROOT
    / "post_train_v2"
    / "scripts"
    / "env"
    / "smoke_teacher_dual_engine.py"
)


def load_script():
    spec = importlib.util.spec_from_file_location("teacher_dual_engine_under_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


UUIDS = (
    "GPU-00112233-4455-6677-8899-aabbccddeeff",
    "GPU-10213243-5465-7687-98a9-bacbdcedfe0f",
)
PCI_IDS = ("0000:8e:00.0", "0000:b3:00.0")


def report_payload(index: int) -> dict[str, object]:
    memory = {
        "name": "NVIDIA A100-SXM4-80GB",
        "free_bytes": 30 * 2**30,
        "total_bytes": 40 * 2**30,
    }
    return {
        "launch_index": index,
        "cuda_visible_devices": str(index),
        "device_count": 1,
        "current_device": 0,
        "cuda_uuid": UUIDS[index],
        "cuda_pci_bus_id": PCI_IDS[index],
        "memory_before": memory,
        "memory_after": memory,
        "output": "<answer>4</answer>",
    }


def report(index: int) -> str:
    import json

    return "TEACHER_REPORT_JSON=" + json.dumps(report_payload(index)) + "\n"


PHYSICAL_GPUS = {
    0: {"uuid": UUIDS[0], "pci_bus_id": "00000000:8E:00.0"},
    1: {"uuid": UUIDS[1], "pci_bus_id": "00000000:B3:00.0"},
}


class FakeProcess:
    def __init__(
        self,
        launched: list["FakeProcess"],
        stdout: str,
        stderr: str = "",
        returncode: int = 0,
        communicate_error: Exception | None = None,
    ) -> None:
        self.launched = launched
        self.stdout = stdout
        self.stderr = stderr
        self.final_returncode = returncode
        self.returncode: int | None = None
        self.communicate_error = communicate_error
        self.terminated = False
        self.killed = False
        self.waited = False
        self.pid = 1000 + len(launched)

    def communicate(self, timeout: float):
        assert len(self.launched) == 2, "both children must start before any wait"
        if self.communicate_error is not None:
            raise self.communicate_error
        self.returncode = self.final_returncode
        return self.stdout, self.stderr

    def poll(self):
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def wait(self, timeout: float | None = None) -> int:
        self.waited = True
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


def make_factory(outcomes: list[dict[str, object]]):
    launched: list[FakeProcess] = []

    def factory(*args, **kwargs):
        process = FakeProcess(launched=launched, **outcomes[len(launched)])
        launched.append(process)
        return process

    return factory, launched


def run_pair(module, outcomes: list[dict[str, object]]):
    factory, launched = make_factory(outcomes)
    result = module.run_teacher_pair(
        model_path="/models/qwen3-8b",
        gpu_memory_utilization=0.8,
        timeout_seconds=10.0,
        physical_gpus=PHYSICAL_GPUS,
        popen_factory=factory,
        monotonic=lambda: 0.0,
    )
    return result, launched


def assert_no_process_left_running(processes: list[FakeProcess]) -> None:
    assert len(processes) == 2
    assert all(process.poll() is not None for process in processes)


def test_both_children_start_before_wait_and_success_returns_two_reports() -> None:
    module = load_script()

    reports, processes = run_pair(
        module,
        [{"stdout": report(0)}, {"stdout": report(1)}],
    )

    assert [item["launch_index"] for item in reports] == [0, 1]
    assert [item["cuda_uuid"] for item in reports] == list(UUIDS)
    assert_no_process_left_running(processes)


@pytest.mark.parametrize(
    ("mutate", "match"),
    (
        (lambda payload: payload.pop("cuda_uuid"), "missing"),
        (lambda payload: payload.__setitem__("unexpected", 1), "unknown"),
        (lambda payload: payload.__setitem__("launch_index", True), "launch_index"),
        (lambda payload: payload.__setitem__("device_count", True), "device_count"),
    ),
)
def test_child_report_schema_rejects_missing_unknown_and_bool_pseudo_ints(
    mutate,
    match: str,
) -> None:
    import json

    module = load_script()
    payload = report_payload(0)
    mutate(payload)

    with pytest.raises(ValueError, match=match):
        module.parse_child_report(
            "TEACHER_REPORT_JSON=" + json.dumps(payload) + "\n"
        )


def test_cuda_identity_comes_from_logical_device_zero_via_libcudart() -> None:
    import ctypes

    module = load_script()
    calls: list[tuple[str, int]] = []
    uuid_bytes = bytes.fromhex("00112233445566778899aabbccddeeff")

    class FakeCudaRuntime:
        def cudaDeviceGetUuid(self, uuid_pointer, device):
            calls.append(("uuid", device))
            for index, value in enumerate(uuid_bytes):
                uuid_pointer._obj.bytes[index] = value
            return 0

        def cudaDeviceGetPCIBusId(self, buffer, length, device):
            calls.append(("pci", device))
            ctypes.memmove(buffer, b"0000:8e:00.0\0", 13)
            return 0

    identity = module.query_cuda_identity(
        device=0,
        cudart_loader=lambda: FakeCudaRuntime(),
    )

    assert identity == {
        "cuda_uuid": UUIDS[0],
        "cuda_pci_bus_id": PCI_IDS[0],
    }
    assert calls == [("uuid", 0), ("pci", 0)]


def test_cuda_identity_loader_failure_is_hard_error() -> None:
    module = load_script()

    def fail_loader():
        raise OSError("libcudart missing")

    with pytest.raises(RuntimeError, match="libcudart"):
        module.query_cuda_identity(device=0, cudart_loader=fail_loader)


def test_child_identity_must_map_to_same_nvidia_smi_record() -> None:
    module = load_script()
    mismatched = report_payload(0)
    mismatched["cuda_pci_bus_id"] = PCI_IDS[1]

    with pytest.raises(ValueError, match="cannot be mapped"):
        module.validate_teacher_reports(
            [mismatched, report_payload(1)],
            PHYSICAL_GPUS,
        )


class StubbornProcess(FakeProcess):
    def __init__(self, launched: list[FakeProcess]) -> None:
        super().__init__(launched=launched, stdout="")
        self.wait_calls = 0
        self.signals: list[int] = []

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls += 1
        if self.wait_calls == 1:
            raise subprocess.TimeoutExpired("child", timeout)
        self.waited = True
        self.returncode = -9
        return self.returncode


def test_posix_cleanup_escalates_process_groups_and_waits_after_kill() -> None:
    module = load_script()
    launched: list[FakeProcess] = []
    processes = [StubbornProcess(launched), StubbornProcess(launched)]
    launched.extend(processes)
    killpg_calls: list[tuple[int, int]] = []

    module.terminate_children(
        processes,
        os_name="posix",
        killpg=lambda pgid, sig: killpg_calls.append((pgid, sig)),
        wait_timeout=0.01,
    )

    assert killpg_calls == [
        (processes[0].pid, module.SIGTERM),
        (processes[1].pid, module.SIGTERM),
        (processes[0].pid, module.SIGKILL),
        (processes[1].pid, module.SIGKILL),
    ]
    assert all(process.wait_calls == 2 for process in processes)
    assert all(process.waited for process in processes)


def test_posix_cleanup_ignores_process_group_disappearing_during_cleanup() -> None:
    module = load_script()
    launched: list[FakeProcess] = []
    processes = [StubbornProcess(launched), StubbornProcess(launched)]
    launched.extend(processes)
    calls: list[tuple[int, int]] = []

    def disappearing_killpg(pgid: int, sig: int) -> None:
        calls.append((pgid, sig))
        raise ProcessLookupError

    module.terminate_children(
        processes,
        os_name="posix",
        killpg=disappearing_killpg,
        wait_timeout=0.01,
    )

    assert calls == [
        (processes[0].pid, module.SIGTERM),
        (processes[1].pid, module.SIGTERM),
        (processes[0].pid, module.SIGKILL),
        (processes[1].pid, module.SIGKILL),
    ]
    assert all(process.wait_calls == 2 for process in processes)
    assert all(process.waited for process in processes)


def test_cleanup_failure_does_not_replace_original_teacher_error() -> None:
    module = load_script()
    factory, _ = make_factory(
        [
            {"stdout": report(0), "stderr": "CUDA out of memory"},
            {"stdout": report(1)},
        ]
    )

    def broken_cleanup(processes) -> None:
        raise RuntimeError("cleanup failed")

    with pytest.raises(RuntimeError, match="CUDA OOM"):
        module.run_teacher_pair(
            model_path="/models/qwen3-8b",
            gpu_memory_utilization=0.8,
            timeout_seconds=10.0,
            physical_gpus=PHYSICAL_GPUS,
            popen_factory=factory,
            monotonic=lambda: 0.0,
            terminate=broken_cleanup,
        )


def test_posix_popen_starts_each_child_in_new_session() -> None:
    module = load_script()
    factory, launched = make_factory(
        [{"stdout": report(0)}, {"stdout": report(1)}]
    )
    kwargs_seen: list[dict[str, object]] = []

    def recording_factory(*args, **kwargs):
        kwargs_seen.append(kwargs)
        return factory(*args, **kwargs)

    module.run_teacher_pair(
        model_path="/models/qwen3-8b",
        gpu_memory_utilization=0.8,
        timeout_seconds=10.0,
        physical_gpus=PHYSICAL_GPUS,
        popen_factory=recording_factory,
        monotonic=lambda: 0.0,
        os_name="posix",
        terminate=lambda processes: module.terminate_children(
            processes, os_name="nt"
        ),
    )

    assert len(launched) == 2
    assert [kwargs["start_new_session"] for kwargs in kwargs_seen] == [True, True]


@pytest.mark.parametrize(
    ("outcomes", "error_type", "match"),
    (
        (
            [
                {
                    "stdout": "",
                    "communicate_error": subprocess.TimeoutExpired("child", 1),
                },
                {"stdout": report(1)},
            ],
            TimeoutError,
            "timed out",
        ),
        (
            [
                {"stdout": report(0), "stderr": "failure", "returncode": 2},
                {"stdout": report(1)},
            ],
            RuntimeError,
            "exit 2",
        ),
        (
            [
                {"stdout": report(0), "stderr": "CUDA out of memory"},
                {"stdout": report(1)},
            ],
            RuntimeError,
            "OOM",
        ),
        (
            [{"stdout": ""}, {"stdout": report(1)}],
            ValueError,
            "structured child report",
        ),
        (
            [
                {"stdout": "TEACHER_REPORT_JSON={not-json}\n"},
                {"stdout": report(1)},
            ],
            ValueError,
            "valid JSON",
        ),
    ),
)
def test_child_failures_are_hard_errors_and_cleanup_both_processes(
    outcomes: list[dict[str, object]],
    error_type: type[Exception],
    match: str,
) -> None:
    module = load_script()
    factory, processes = make_factory(outcomes)

    with pytest.raises(error_type, match=match):
        module.run_teacher_pair(
            model_path="/models/qwen3-8b",
            gpu_memory_utilization=0.8,
            timeout_seconds=10.0,
            physical_gpus=PHYSICAL_GPUS,
            popen_factory=factory,
            monotonic=lambda: 0.0,
        )

    assert_no_process_left_running(processes)
