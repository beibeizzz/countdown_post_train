import json
import os
import threading
import types
from datetime import datetime
from pathlib import Path

import pytest

from post_train.src.countdown import output_lock
from post_train.src.countdown.output_lock import OutputLock, process_is_alive


def make_lock(tmp_path: Path, **overrides) -> OutputLock:
    values = {
        "path": tmp_path / "output" / ".generation.lock",
        "config_path": tmp_path / "config.yaml",
        "output_dir": tmp_path / "output",
        "topology": "dual_gpu_teacher",
        "hostname": "local-host",
        "pid": 1234,
        "process_alive": lambda pid: False,
        "owner_token": "owner-token",
    }
    values.update(overrides)
    return OutputLock(**values)


def existing_lock_metadata(lock: OutputLock) -> dict:
    return {
        "schema_version": 1,
        "pid": 2222,
        "hostname": "local-host",
        "started_at": "2026-06-13T00:00:00+00:00",
        "config_path": str(lock.config_path.resolve()),
        "output_dir": str(lock.output_dir.resolve()),
        "topology": lock.topology,
        "owner_token": "other",
    }


def write_existing_lock(lock: OutputLock, *, pid: int, hostname: str, owner_token: str = "other"):
    lock.path.parent.mkdir(parents=True, exist_ok=True)
    metadata = existing_lock_metadata(lock)
    metadata.update(pid=pid, hostname=hostname, owner_token=owner_token)
    lock.path.write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )


def test_acquire_writes_metadata_and_requests_exclusive_mode(tmp_path: Path, monkeypatch):
    lock = make_lock(tmp_path)
    calls = []
    real_open = os.open

    def recording_open(path, flags, mode=0o777):
        calls.append((path, flags, mode))
        return real_open(path, flags, mode)

    monkeypatch.setattr(output_lock.os, "open", recording_open)

    lock.acquire()

    metadata = json.loads(lock.path.read_text(encoding="utf-8"))
    assert metadata["schema_version"] == 1
    assert metadata["pid"] == 1234
    assert metadata["hostname"] == "local-host"
    assert datetime.fromisoformat(metadata["started_at"]).utcoffset().total_seconds() == 0
    assert metadata["config_path"] == str(lock.config_path.resolve())
    assert metadata["output_dir"] == str(lock.output_dir.resolve())
    assert metadata["topology"] == "dual_gpu_teacher"
    assert metadata["owner_token"] == "owner-token"
    assert lock.output_dir.is_dir()

    _, flags, mode = calls[0]
    assert flags == os.O_CREAT | os.O_EXCL | os.O_WRONLY
    assert mode == 0o600


def test_defaults_are_resolved_when_output_lock_is_constructed(tmp_path: Path, monkeypatch):
    generated_uuid = types.SimpleNamespace(hex="0123456789abcdef")
    monkeypatch.setattr(output_lock.socket, "gethostname", lambda: "patched-host")
    monkeypatch.setattr(output_lock.os, "getpid", lambda: 9876)
    monkeypatch.setattr(output_lock.uuid, "uuid4", lambda: generated_uuid)

    lock = OutputLock(
        path=tmp_path / "output" / ".generation.lock",
        config_path=tmp_path / "config.yaml",
        output_dir=tmp_path / "output",
        topology="dual_gpu_teacher",
    )

    assert lock.hostname == "patched-host"
    assert lock.pid == 9876
    assert lock.process_alive is process_is_alive
    assert lock.owner_token == generated_uuid.hex


def test_default_process_alive_is_resolved_when_output_lock_is_constructed(
    tmp_path: Path, monkeypatch
):
    replacement = lambda pid: True
    monkeypatch.setattr(output_lock, "process_is_alive", replacement)

    lock = OutputLock(
        path=tmp_path / "output" / ".generation.lock",
        config_path=tmp_path / "config.yaml",
        output_dir=tmp_path / "output",
        topology="dual_gpu_teacher",
    )

    assert lock.process_alive is replacement


def test_acquire_rejects_live_pid_on_same_host_even_with_recovery(tmp_path: Path):
    lock = make_lock(tmp_path, process_alive=lambda pid: pid == 2222)
    write_existing_lock(lock, pid=2222, hostname="local-host")

    with pytest.raises(RuntimeError, match="active"):
        lock.acquire(recover_stale=True)

    assert lock.path.exists()


def test_acquire_recovers_stale_same_host_lock_only_when_explicit(tmp_path: Path):
    lock = make_lock(tmp_path, process_alive=lambda pid: False)
    write_existing_lock(lock, pid=2222, hostname="local-host")

    with pytest.raises(RuntimeError, match="recover_stale"):
        lock.acquire()

    lock.acquire(recover_stale=True)

    assert json.loads(lock.path.read_text(encoding="utf-8"))["owner_token"] == "owner-token"


def test_acquire_refuses_foreign_host_lock_even_with_recovery(tmp_path: Path):
    lock = make_lock(tmp_path)
    write_existing_lock(lock, pid=2222, hostname="foreign-host")

    with pytest.raises(RuntimeError, match="foreign-host"):
        lock.acquire(recover_stale=True)

    assert lock.path.exists()


def test_acquire_refuses_corrupt_lock_even_with_recovery(tmp_path: Path):
    lock = make_lock(tmp_path)
    lock.path.parent.mkdir(parents=True)
    lock.path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(RuntimeError, match="unreadable|corrupt"):
        lock.acquire(recover_stale=True)

    assert lock.path.read_text(encoding="utf-8") == "{not-json"


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("started_at", "2026-06-13T08:00:00+08:00"),
        ("started_at", "2026-06-13T00:00:00"),
        ("config_path", "relative/config.yaml"),
        ("output_dir", "relative/output"),
    ],
)
def test_acquire_refuses_non_utc_timestamp_or_relative_paths(
    tmp_path: Path, field_name: str, invalid_value: str
):
    def unexpected_liveness_check(pid):
        raise AssertionError("process liveness must not be checked for corrupt metadata")

    lock = make_lock(tmp_path, process_alive=unexpected_liveness_check)
    metadata = existing_lock_metadata(lock)
    metadata[field_name] = invalid_value
    lock.path.parent.mkdir(parents=True)
    lock.path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(RuntimeError, match="corrupt"):
        lock.acquire(recover_stale=True)

    assert json.loads(lock.path.read_text(encoding="utf-8")) == metadata


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("schema_version", 2),
        ("schema_version", True),
        ("schema_version", "1"),
        ("pid", 0),
        ("pid", True),
        ("pid", "2222"),
        ("pid", None),
        ("started_at", ""),
        ("started_at", "not-a-date"),
        ("started_at", "2026-06-13T00:00:00"),
        ("config_path", ""),
        ("config_path", 123),
        ("output_dir", ""),
        ("output_dir", None),
        ("topology", ""),
        ("topology", []),
        ("owner_token", ""),
        ("owner_token", False),
    ],
)
def test_acquire_refuses_structurally_corrupt_json_lock(
    tmp_path: Path, field_name: str, invalid_value
):
    def unexpected_liveness_check(pid):
        raise AssertionError("process liveness must not be checked for corrupt metadata")

    lock = make_lock(tmp_path, process_alive=unexpected_liveness_check)
    metadata = existing_lock_metadata(lock)
    metadata[field_name] = invalid_value
    lock.path.parent.mkdir(parents=True)
    lock.path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(RuntimeError, match="corrupt"):
        lock.acquire(recover_stale=True)

    assert json.loads(lock.path.read_text(encoding="utf-8")) == metadata


@pytest.mark.parametrize("hostname", [None, "", 123])
def test_acquire_refuses_missing_or_unverifiable_hostname(tmp_path: Path, hostname):
    def unexpected_liveness_check(pid):
        raise AssertionError("process liveness must not be checked without a valid hostname")

    lock = make_lock(tmp_path, process_alive=unexpected_liveness_check)
    metadata = existing_lock_metadata(lock)
    if hostname is None:
        metadata.pop("hostname")
    else:
        metadata["hostname"] = hostname
    lock.path.parent.mkdir(parents=True)
    lock.path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(RuntimeError, match="corrupt"):
        lock.acquire(recover_stale=True)

    assert lock.path.exists()


@pytest.mark.parametrize(
    "missing_field",
    [
        "schema_version",
        "pid",
        "started_at",
        "config_path",
        "output_dir",
        "topology",
        "owner_token",
    ],
)
def test_acquire_refuses_lock_with_missing_required_metadata(tmp_path: Path, missing_field: str):
    def unexpected_liveness_check(pid):
        raise AssertionError("process liveness must not be checked for incomplete metadata")

    lock = make_lock(tmp_path, process_alive=unexpected_liveness_check)
    metadata = existing_lock_metadata(lock)
    metadata.pop(missing_field)
    lock.path.parent.mkdir(parents=True)
    lock.path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(RuntimeError, match="corrupt"):
        lock.acquire(recover_stale=True)

    assert lock.path.exists()


def test_release_only_removes_lock_owned_by_matching_token(tmp_path: Path):
    lock = make_lock(tmp_path)
    write_existing_lock(lock, pid=1234, hostname="local-host", owner_token="replacement-owner")

    lock.release()

    assert lock.path.exists()

    metadata = json.loads(lock.path.read_text(encoding="utf-8"))
    metadata["owner_token"] = "owner-token"
    lock.path.write_text(json.dumps(metadata), encoding="utf-8")

    lock.release()

    assert not lock.path.exists()


def test_context_manager_releases_lock(tmp_path: Path):
    lock = make_lock(tmp_path)

    with lock as acquired:
        assert acquired is lock
        assert lock.path.exists()

    assert not lock.path.exists()
    assert lock.path.with_name(f"{lock.path.name}.guard").exists()


def test_two_stale_recoverers_cannot_both_acquire(tmp_path: Path):
    first_checked_stale = threading.Event()
    second_checked_stale = threading.Event()
    results = {}

    def first_process_alive(pid):
        if pid == 2222:
            first_checked_stale.set()
            second_checked_stale.wait(timeout=0.3)
            return False
        return True

    def second_process_alive(pid):
        if pid == 2222:
            second_checked_stale.set()
            return False
        return True

    first = make_lock(
        tmp_path,
        pid=3001,
        owner_token="first-owner",
        process_alive=first_process_alive,
    )
    second = make_lock(
        tmp_path,
        pid=3002,
        owner_token="second-owner",
        process_alive=second_process_alive,
    )
    write_existing_lock(first, pid=2222, hostname="local-host")

    def acquire(name, lock):
        try:
            lock.acquire(recover_stale=True)
        except Exception as exc:
            results[name] = exc
        else:
            results[name] = "acquired"

    first_thread = threading.Thread(target=acquire, args=("first", first))
    second_thread = threading.Thread(target=acquire, args=("second", second))
    first_thread.start()
    assert first_checked_stale.wait(timeout=1)
    second_thread.start()
    first_thread.join(timeout=2)
    second_thread.join(timeout=2)

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert list(results.values()).count("acquired") == 1
    rejection = next(result for result in results.values() if result != "acquired")
    assert isinstance(rejection, RuntimeError)
    assert "active" in str(rejection)


def test_release_cannot_delete_replacement_owner_lock(tmp_path: Path):
    releaser = make_lock(tmp_path, pid=3001, owner_token="original-owner")
    replacement = make_lock(tmp_path, pid=3002, owner_token="replacement-owner")
    releaser.acquire()
    release_read = threading.Event()
    replacement_acquired = threading.Event()
    original_read = releaser._read_existing_lock

    def paused_read():
        metadata = original_read()
        release_read.set()
        replacement_acquired.wait(timeout=0.3)
        return metadata

    releaser._read_existing_lock = paused_read
    results = {}

    def release():
        releaser.release()
        results["release"] = "released"

    def acquire_replacement():
        try:
            replacement.acquire(recover_stale=True)
        except Exception as exc:
            results["replacement"] = exc
        else:
            results["replacement"] = "acquired"
            replacement_acquired.set()

    release_thread = threading.Thread(target=release)
    acquire_thread = threading.Thread(target=acquire_replacement)
    release_thread.start()
    assert release_read.wait(timeout=1)
    acquire_thread.start()
    release_thread.join(timeout=2)
    acquire_thread.join(timeout=2)

    assert not release_thread.is_alive()
    assert not acquire_thread.is_alive()
    assert results == {"release": "released", "replacement": "acquired"}
    assert json.loads(replacement.path.read_text(encoding="utf-8"))["owner_token"] == (
        "replacement-owner"
    )


@pytest.mark.parametrize("pid", [0, -1, True, "42", None])
def test_process_is_alive_rejects_invalid_pid(pid, monkeypatch):
    def unexpected_kill(pid, signal):
        raise AssertionError("os.kill must not be called")

    monkeypatch.setattr(output_lock.os, "kill", unexpected_kill)

    assert process_is_alive(pid) is False


def test_process_is_alive_returns_true_when_signal_check_succeeds(monkeypatch):
    monkeypatch.setattr(output_lock, "_is_windows", lambda: False)
    monkeypatch.setattr(output_lock.os, "kill", lambda pid, signal: None)

    assert process_is_alive(42) is True


def test_process_is_alive_treats_permission_error_as_alive(monkeypatch):
    monkeypatch.setattr(output_lock, "_is_windows", lambda: False)

    def permission_denied(pid, signal):
        raise PermissionError

    monkeypatch.setattr(output_lock.os, "kill", permission_denied)

    assert process_is_alive(42) is True


def test_process_is_alive_treats_process_lookup_error_as_dead(monkeypatch):
    monkeypatch.setattr(output_lock, "_is_windows", lambda: False)

    def missing_process(pid, signal):
        raise ProcessLookupError

    monkeypatch.setattr(output_lock.os, "kill", missing_process)

    assert process_is_alive(42) is False


def test_process_is_alive_windows_branch_never_calls_os_kill(monkeypatch):
    monkeypatch.setattr(output_lock, "_is_windows", lambda: True)
    monkeypatch.setattr(output_lock, "_process_is_alive_windows", lambda pid: True)

    def unexpected_kill(pid, signal):
        raise AssertionError("Windows liveness checks must not call os.kill")

    monkeypatch.setattr(output_lock.os, "kill", unexpected_kill)

    assert process_is_alive(42) is True


def test_windows_process_query_success_closes_handle():
    closed_handles = []

    assert output_lock._process_is_alive_windows(
        42,
        open_process=lambda access, inherit, pid: 123,
        close_handle=closed_handles.append,
        get_last_error=lambda: 0,
    )
    assert closed_handles == [123]


@pytest.mark.parametrize(
    ("error_code", "expected"),
    [
        (87, False),
        (5, True),
    ],
)
def test_windows_process_query_maps_known_open_errors(error_code: int, expected: bool):
    assert (
        output_lock._process_is_alive_windows(
            42,
            open_process=lambda access, inherit, pid: 0,
            close_handle=lambda handle: None,
            get_last_error=lambda: error_code,
        )
        is expected
    )


def test_windows_process_query_raises_for_unverifiable_error():
    with pytest.raises(OSError):
        output_lock._process_is_alive_windows(
            42,
            open_process=lambda access, inherit, pid: 0,
            close_handle=lambda handle: None,
            get_last_error=lambda: 1234,
        )
