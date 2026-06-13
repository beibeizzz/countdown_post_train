from __future__ import annotations

import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
GENERATED_PATTERN = re.compile(
    r"(^|/)(?:__pycache__/.*\.pyc|\.pytest_cache(?:/|$)|\.pytest_tmp(?:/|$))"
)
REQUIRED_IGNORE_RULES = {
    "__pycache__/",
    "*.py[cod]",
    ".pytest_cache/",
    ".pytest_tmp/",
}


def test_generated_python_and_pytest_artifacts_are_not_tracked() -> None:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    tracked_generated = [
        path
        for path in result.stdout.splitlines()
        if GENERATED_PATTERN.search(path)
    ]

    assert tracked_generated == []


def test_repository_gitignore_blocks_generated_test_artifacts() -> None:
    rules: set[str] = set()
    for relative_path in (".gitignore", "post_train_v2/.gitignore"):
        path = REPO_ROOT / relative_path
        if not path.exists():
            continue
        rules.update(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )

    assert REQUIRED_IGNORE_RULES <= rules
