import pytest

from post_train_v2.src.config import (
    REPO_ROOT,
    load_yaml,
    require_keys,
    resolve_repo_path,
)


def test_load_yaml_returns_mapping(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("name: countdown\ncount: 3\n", encoding="utf-8")

    assert load_yaml(path) == {"name": "countdown", "count": 3}


@pytest.mark.parametrize(
    "content",
    (
        "",
        "countdown\n",
        "- countdown\n- training\n",
    ),
)
def test_load_yaml_rejects_non_mapping_values(tmp_path, content):
    path = tmp_path / "config.yaml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=r"YAML config must be a mapping"):
        load_yaml(path)


def test_resolve_repo_path_is_independent_of_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert resolve_repo_path("post_train_v2/data/x.jsonl") == (
        REPO_ROOT / "post_train_v2/data/x.jsonl"
    )


def test_require_keys_reports_all_missing_keys():
    with pytest.raises(ValueError, match=r"missing keys: a, c"):
        require_keys({"b": 1}, "a", "b", "c")
