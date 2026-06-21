import errno
import json
import os
import tempfile
from pathlib import Path

import pytest

from post_train_v2.src.artifacts import atomic
from post_train_v2.src.artifacts.atomic import publish_json, publish_jsonl
from post_train_v2.src.artifacts.hashing import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_canonical_json,
    sha256_config,
    sha256_file,
)


def test_publish_json_replaces_atomically(tmp_path):
    path = tmp_path / "artifact.json"
    publish_json(path, {"value": 1})
    assert json.loads(path.read_text()) == {"value": 1}
    assert not list(tmp_path.glob("*.tmp"))


def test_publish_json_creates_parent_and_uses_canonical_utf8(tmp_path):
    path = tmp_path / "nested" / "artifact.json"

    publish_json(path, {"z": "雪", "a": 1})

    assert path.read_bytes() == b'{"a":1,"z":"\xe9\x9b\xaa"}'


def test_publish_jsonl_writes_one_canonical_utf8_record_per_line(tmp_path):
    path = tmp_path / "rows.jsonl"

    publish_jsonl(path, [{"z": 2, "a": 1}, {"text": "雪"}])

    assert path.read_bytes() == (
        b'{"a":1,"z":2}\n{"text":"\xe9\x9b\xaa"}\n'
    )
    assert not list(tmp_path.glob("*.tmp"))


def test_publish_jsonl_streams_each_single_pass_row_to_the_temp_file(
    tmp_path, monkeypatch
):
    path = tmp_path / "rows.jsonl"
    events = []
    original_named_temporary_file = tempfile.NamedTemporaryFile

    class SinglePassRows:
        def __init__(self):
            self.iterations = 0

        def __iter__(self):
            self.iterations += 1
            if self.iterations > 1:
                raise AssertionError("rows were consumed more than once")
            for row in ({"id": 1}, {"id": 2}):
                events.append(("yield", row["id"]))
                yield row

    class WritingTemporaryFile:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            self.handle.__enter__()
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

        def __getattr__(self, name):
            return getattr(self.handle, name)

        def write(self, payload):
            events.append(("write", payload))
            return self.handle.write(payload)

    def named_temporary_file(*args, **kwargs):
        events.append(("temp", Path(kwargs["dir"])))
        return WritingTemporaryFile(
            original_named_temporary_file(*args, **kwargs)
        )

    monkeypatch.setattr(
        atomic.tempfile, "NamedTemporaryFile", named_temporary_file
    )
    monkeypatch.setattr(
        atomic,
        "_fsync_parent_directory",
        lambda directory: events.append(("dir_fsync", Path(directory))),
        raising=False,
    )
    rows = SinglePassRows()

    publish_jsonl(path, rows)

    assert rows.iterations == 1
    assert events == [
        ("temp", path.parent),
        ("yield", 1),
        ("write", b'{"id":1}\n'),
        ("yield", 2),
        ("write", b'{"id":2}\n'),
        ("dir_fsync", path.parent),
    ]


def test_atomic_publication_removes_temp_file_when_replace_fails(
    tmp_path, monkeypatch
):
    path = tmp_path / "artifact.json"

    def fail_replace(source: str | Path, destination: str | Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr("post_train_v2.src.artifacts.atomic.os.replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        publish_json(path, {"value": 1})

    assert not path.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_publication_flushes_and_replaces_from_destination_directory(
    tmp_path, monkeypatch
):
    path = tmp_path / "nested" / "artifact.json"
    events = []
    original_named_temporary_file = tempfile.NamedTemporaryFile
    original_fsync = os.fsync
    original_replace = os.replace

    class FlushedTemporaryFile:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            self.handle.__enter__()
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

        def __getattr__(self, name):
            return getattr(self.handle, name)

        def flush(self):
            events.append(("flush", Path(self.handle.name)))
            return self.handle.flush()

    def named_temporary_file(*args, **kwargs):
        events.append(("tempdir", Path(kwargs["dir"])))
        return FlushedTemporaryFile(
            original_named_temporary_file(*args, **kwargs)
        )

    def fsync(file_descriptor):
        events.append(("fsync", file_descriptor))
        return original_fsync(file_descriptor)

    def replace(source, destination):
        events.append(("replace", Path(source), Path(destination)))
        return original_replace(source, destination)

    def fsync_parent_directory(directory):
        events.append(("dir_fsync", Path(directory)))

    monkeypatch.setattr(
        "post_train_v2.src.artifacts.atomic.tempfile.NamedTemporaryFile",
        named_temporary_file,
    )
    monkeypatch.setattr("post_train_v2.src.artifacts.atomic.os.fsync", fsync)
    monkeypatch.setattr("post_train_v2.src.artifacts.atomic.os.replace", replace)
    monkeypatch.setattr(
        atomic,
        "_fsync_parent_directory",
        fsync_parent_directory,
        raising=False,
    )

    publish_json(path, {"value": 1})

    assert events[0] == ("tempdir", path.parent)
    assert [event[0] for event in events] == [
        "tempdir",
        "flush",
        "fsync",
        "replace",
        "dir_fsync",
    ]
    assert events[1][1].parent == path.parent
    assert events[3][1].parent == path.parent
    assert events[3][2] == path
    assert events[4] == ("dir_fsync", path.parent)


def test_parent_directory_fsync_ignores_only_unsupported_fsync(
    tmp_path, monkeypatch
):
    events = []
    monkeypatch.setattr(atomic.os, "name", "posix")
    monkeypatch.setattr(atomic.os, "open", lambda path, flags: 91)
    monkeypatch.setattr(
        atomic.os,
        "fsync",
        lambda descriptor: (_ for _ in ()).throw(
            OSError(errno.EINVAL, "directory fsync unsupported")
        ),
    )
    monkeypatch.setattr(
        atomic.os,
        "close",
        lambda descriptor: events.append(("close", descriptor)),
    )

    atomic._fsync_parent_directory(tmp_path)

    assert events == [("close", 91)]


def test_parent_directory_fsync_does_not_hide_other_io_errors(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(atomic.os, "name", "posix")
    monkeypatch.setattr(
        atomic.os,
        "open",
        lambda path, flags: (_ for _ in ()).throw(
            OSError(errno.EIO, "storage failure")
        ),
    )

    with pytest.raises(OSError, match="storage failure"):
        atomic._fsync_parent_directory(tmp_path)


def test_hashing_uses_canonical_json_and_file_bytes(tmp_path):
    expected_bytes = b'{"a":1,"z":"\xe9\x9b\xaa"}'
    path = tmp_path / "payload.json"
    path.write_bytes(expected_bytes)

    assert canonical_json_bytes({"z": "雪", "a": 1}) == expected_bytes
    assert sha256_bytes(expected_bytes) == sha256_file(path)
    assert sha256_canonical_json({"z": "雪", "a": 1}) == sha256_bytes(
        expected_bytes
    )
    assert sha256_config({"z": "雪", "a": 1}) == sha256_canonical_json(
        {"a": 1, "z": "雪"}
    )


def test_canonical_json_rejects_nan():
    with pytest.raises(ValueError, match="Out of range float values"):
        canonical_json_bytes({"value": float("nan")})
