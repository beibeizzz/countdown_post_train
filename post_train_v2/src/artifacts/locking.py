"""Exclusive output-directory locks for multi-file artifact publication."""

from __future__ import annotations

import errno
import json
import os
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    unsupported = {
        errno.EINVAL,
        getattr(errno, "ENOTSUP", errno.EINVAL),
        getattr(errno, "EOPNOTSUPP", errno.EINVAL),
    }
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(directory, flags)
    except OSError as error:
        if error.errno in unsupported:
            return
        raise
    try:
        try:
            os.fsync(descriptor)
        except OSError as error:
            if error.errno not in unsupported:
                raise
    finally:
        os.close(descriptor)


@contextmanager
def exclusive_output_lock(
    output_dir: str | Path,
    *,
    lock_name: str,
    metadata: Mapping[str, Any],
) -> Iterator[None]:
    directory = Path(output_dir)
    if (
        not isinstance(lock_name, str)
        or not lock_name
        or Path(lock_name).name != lock_name
    ):
        raise ValueError("lock_name must be a plain filename")
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / lock_name
    owner_token = uuid.uuid4().hex
    try:
        descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError as error:
        raise RuntimeError(f"output lock already exists: {lock_path}") from error

    metadata_written = False
    try:
        payload = {
            "schema_version": 1,
            "pid": os.getpid(),
            "owner_token": owner_token,
            **dict(metadata),
        }
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        metadata_written = True
        _fsync_directory(directory)
        yield
    finally:
        if not metadata_written:
            lock_path.unlink(missing_ok=True)
            _fsync_directory(directory)
        else:
            try:
                current = json.loads(lock_path.read_text(encoding="utf-8"))
            except (FileNotFoundError, UnicodeError, json.JSONDecodeError):
                current = None
            if isinstance(current, Mapping) and current.get(
                "owner_token"
            ) == owner_token:
                lock_path.unlink(missing_ok=True)
                _fsync_directory(directory)

