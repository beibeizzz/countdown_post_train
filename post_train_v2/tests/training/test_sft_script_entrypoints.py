from __future__ import annotations

import subprocess
import sys


def test_sft_scripts_are_directly_invokable_from_repo_root():
    scripts = [
        "post_train_v2/scripts/sft/train_full.py",
        "post_train_v2/scripts/sft/train_lora.py",
        "post_train_v2/scripts/sft/build_rft_data.py",
        "post_train_v2/scripts/sft/train_rft.py",
        "post_train_v2/scripts/sft/merge_lora.py",
    ]

    for script in scripts:
        result = subprocess.run(
            [sys.executable, script, "--help"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, result.stderr
