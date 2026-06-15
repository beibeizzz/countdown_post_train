from __future__ import annotations

import errno
import os
import tempfile
from collections.abc import Callable, Iterable
from io import BufferedWriter
from pathlib import Path
from typing import Any

from post_train_v2.src.artifacts.hashing import canonical_json_bytes

UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS = {
    errno.EINVAL,
    getattr(errno, "ENOTSUP", errno.EINVAL),
    getattr(errno, "EOPNOTSUPP", errno.EINVAL),
}


def _fsync_parent_directory(directory: Path) -> None:
    if os.name == "nt":
        return

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        directory_descriptor = os.open(directory, flags)
    except OSError as error:
        if error.errno in UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS:
            return
        raise

    try:
        try:
            os.fsync(directory_descriptor)
        except OSError as error:
            if error.errno not in UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS:
                raise
    finally:
        os.close(directory_descriptor)


def _publish(
    path: str | Path, write_payload: Callable[[BufferedWriter], None]
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            delete=False,
            suffix=".tmp",
        ) as temporary:
            temporary_path = Path(temporary.name)
            write_payload(temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
        _fsync_parent_directory(destination.parent)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _publish_bytes(path: str | Path, payload: bytes) -> None:
    _publish(path, lambda temporary: temporary.write(payload))


def publish_json(path: str | Path, value: Any) -> None:
    _publish_bytes(path, canonical_json_bytes(value))


def publish_jsonl(path: str | Path, rows: Iterable[Any]) -> None:
    def write_rows(temporary: BufferedWriter) -> None:
        for row in rows:
            temporary.write(canonical_json_bytes(row) + b"\n")

    _publish(path, write_rows)
