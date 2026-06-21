from __future__ import annotations

import json
from pathlib import Path

import pytest

from post_train_v2.src.generation.output_lock import OutputLock


def make_lock(tmp_path: Path, **overrides) -> OutputLock:
    values = {
        "path": tmp_path / "output/.teacher_pool.lock",
        "config_path": tmp_path / "config.yaml",
        "output_dir": tmp_path / "output",
        "topology": "dual_tp1",
        "hostname": "host-a",
        "pid": 123,
        "process_alive": lambda pid: False,
        "owner_token": "owner-a",
    }
    values.update(overrides)
    return OutputLock(**values)


def test_lock_acquire_and_owned_release(tmp_path: Path) -> None:
    lock = make_lock(tmp_path)

    lock.acquire()

    metadata = json.loads(lock.path.read_text(encoding="utf-8"))
    assert metadata["owner_token"] == "owner-a"
    assert lock.recovered_stale is False
    lock.release()
    assert not lock.path.exists()


def test_stale_lock_requires_explicit_recovery(tmp_path: Path) -> None:
    stale = make_lock(tmp_path, owner_token="old-owner")
    stale.acquire()
    replacement = make_lock(tmp_path, owner_token="new-owner")

    with pytest.raises(RuntimeError, match="stale"):
        replacement.acquire()

    replacement.acquire(recover_stale=True)
    assert replacement.recovered_stale is True
    replacement.release()


def test_active_lock_is_never_recovered(tmp_path: Path) -> None:
    active = make_lock(tmp_path, owner_token="active-owner")
    active.acquire()
    replacement = make_lock(
        tmp_path,
        owner_token="new-owner",
        process_alive=lambda pid: True,
    )

    with pytest.raises(RuntimeError, match="active"):
        replacement.acquire(recover_stale=True)


def test_non_owner_does_not_remove_lock(tmp_path: Path) -> None:
    owner = make_lock(tmp_path, owner_token="owner")
    owner.acquire()

    make_lock(tmp_path, owner_token="other").release()

    assert owner.path.exists()
