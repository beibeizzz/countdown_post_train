"""Process-aware output lock used by the Teacher coordinator."""

from __future__ import annotations

import json
import os
import socket
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
ERROR_ACCESS_DENIED = 5
ERROR_INVALID_PARAMETER = 87


def _is_windows() -> bool:
    return os.name == "nt"


def _get_windows_process_api():
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    open_process = kernel32.OpenProcess
    open_process.argtypes = [
        wintypes.DWORD,
        wintypes.BOOL,
        wintypes.DWORD,
    ]
    open_process.restype = wintypes.HANDLE
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [wintypes.HANDLE]
    close_handle.restype = wintypes.BOOL
    return open_process, close_handle, ctypes.get_last_error


def _process_is_alive_windows(
    pid: int,
    *,
    open_process=None,
    close_handle=None,
    get_last_error=None,
) -> bool:
    if open_process is None or close_handle is None or get_last_error is None:
        default_open, default_close, default_get_error = (
            _get_windows_process_api()
        )
        open_process = open_process or default_open
        close_handle = close_handle or default_close
        get_last_error = get_last_error or default_get_error

    handle = open_process(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if handle:
        try:
            return True
        finally:
            close_handle(handle)

    error_code = get_last_error()
    if error_code == ERROR_INVALID_PARAMETER:
        return False
    if error_code == ERROR_ACCESS_DENIED:
        return True
    raise OSError(error_code, f"OpenProcess failed for PID {pid}")


def process_is_alive(pid: int) -> bool:
    if type(pid) is not int or pid <= 0:
        return False
    if _is_windows():
        return _process_is_alive_windows(pid)
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except ProcessLookupError:
        return False
    return True


@dataclass
class OutputLock:
    path: Path
    config_path: Path
    output_dir: Path
    topology: str
    hostname: str = field(default_factory=socket.gethostname)
    pid: int = field(default_factory=os.getpid)
    process_alive: Callable[[int], bool] = process_is_alive
    owner_token: str = field(default_factory=lambda: uuid.uuid4().hex)
    _recovered_stale: bool = field(default=False, init=False, repr=False)

    @property
    def recovered_stale(self) -> bool:
        return self._recovered_stale

    def acquire(self, recover_stale: bool = False) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        with self._operation_guard():
            self._recovered_stale = False
            try:
                descriptor = os.open(
                    self.path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o600,
                )
            except FileExistsError:
                self._handle_existing_lock(recover_stale)
                descriptor = os.open(
                    self.path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o600,
                )
                self._recovered_stale = True

            metadata = {
                "schema_version": 1,
                "pid": self.pid,
                "hostname": self.hostname,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "config_path": str(self.config_path.resolve()),
                "output_dir": str(self.output_dir.resolve()),
                "topology": self.topology,
                "owner_token": self.owner_token,
            }
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    json.dump(metadata, handle)
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            except BaseException:
                self.path.unlink(missing_ok=True)
                raise

    @contextmanager
    def _operation_guard(self):
        guard_path = self.path.with_name(f"{self.path.name}.guard")
        with guard_path.open("a+b") as guard_file:
            self._lock_guard_file(guard_file)
            try:
                yield
            finally:
                self._unlock_guard_file(guard_file)

    @staticmethod
    def _lock_guard_file(guard_file) -> None:
        if _is_windows():
            import msvcrt

            guard_file.seek(0, os.SEEK_END)
            if guard_file.tell() == 0:
                guard_file.write(b"\0")
                guard_file.flush()
                os.fsync(guard_file.fileno())
            guard_file.seek(0)
            msvcrt.locking(guard_file.fileno(), msvcrt.LK_LOCK, 1)
            return

        import fcntl

        fcntl.flock(guard_file.fileno(), fcntl.LOCK_EX)

    @staticmethod
    def _unlock_guard_file(guard_file) -> None:
        if _is_windows():
            import msvcrt

            guard_file.seek(0)
            msvcrt.locking(guard_file.fileno(), msvcrt.LK_UNLCK, 1)
            return

        import fcntl

        fcntl.flock(guard_file.fileno(), fcntl.LOCK_UN)

    def _handle_existing_lock(self, recover_stale: bool) -> None:
        metadata = self._read_existing_lock()
        existing_hostname = metadata["hostname"]
        existing_pid = metadata["pid"]
        if existing_hostname != self.hostname:
            raise RuntimeError(
                "existing output lock belongs to foreign host "
                f"{existing_hostname}: {self.path}"
            )
        try:
            alive = self.process_alive(existing_pid)
        except Exception as error:
            raise RuntimeError(
                "could not verify existing output lock process "
                f"{existing_pid}: {self.path}"
            ) from error
        if alive:
            raise RuntimeError(
                f"existing output lock is active for PID {existing_pid}: "
                f"{self.path}"
            )
        if not recover_stale:
            raise RuntimeError(
                "existing output lock is stale; pass recover_stale=True "
                f"to replace it: {self.path}"
            )
        self.path.unlink()

    def _read_existing_lock(self) -> dict:
        try:
            metadata = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"existing output lock is unreadable or corrupt: {self.path}"
            ) from error
        if not isinstance(metadata, dict):
            raise RuntimeError(f"existing output lock is corrupt: {self.path}")
        self._validate_metadata(metadata)
        return metadata

    def _validate_metadata(self, metadata: dict) -> None:
        if (
            type(metadata.get("schema_version")) is not int
            or metadata["schema_version"] != 1
            or type(metadata.get("pid")) is not int
            or metadata["pid"] <= 0
        ):
            raise RuntimeError(f"existing output lock is corrupt: {self.path}")
        for field_name in (
            "hostname",
            "started_at",
            "config_path",
            "output_dir",
            "topology",
            "owner_token",
        ):
            value = metadata.get(field_name)
            if not isinstance(value, str) or not value.strip():
                raise RuntimeError(
                    f"existing output lock is corrupt: {self.path}"
                )
        try:
            started_at = datetime.fromisoformat(metadata["started_at"])
        except ValueError as error:
            raise RuntimeError(
                f"existing output lock is corrupt: {self.path}"
            ) from error
        offset = started_at.utcoffset()
        if offset is None or offset.total_seconds() != 0:
            raise RuntimeError(f"existing output lock is corrupt: {self.path}")
        if not Path(metadata["config_path"]).is_absolute() or not Path(
            metadata["output_dir"]
        ).is_absolute():
            raise RuntimeError(f"existing output lock is corrupt: {self.path}")

    def release(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._operation_guard():
            if not self.path.exists():
                return
            try:
                metadata = self._read_existing_lock()
            except RuntimeError:
                return
            if metadata.get("owner_token") == self.owner_token:
                self.path.unlink(missing_ok=True)
