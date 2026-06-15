import json
from pathlib import Path

import pytest

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
