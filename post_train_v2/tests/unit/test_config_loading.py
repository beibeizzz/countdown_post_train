import pytest

from post_train_v2.src.config.loading import (
    REPO_ROOT,
    require_keys,
    resolve_repo_path,
)


def test_resolve_repo_path_is_independent_of_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert resolve_repo_path("post_train_v2/data/x.jsonl") == (
        REPO_ROOT / "post_train_v2/data/x.jsonl"
    )


def test_require_keys_reports_all_missing_keys():
    with pytest.raises(ValueError, match=r"missing keys: a, c"):
        require_keys({"b": 1}, "a", "b", "c")
