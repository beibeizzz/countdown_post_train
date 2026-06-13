# Environment Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the actionable findings from the Flash Attention environment review and leave a clean, executable remote acceptance runbook.

**Architecture:** LoRA evaluation will use defense in depth: the PEFT smoke producer saves tokenizer artifacts, while the evaluator falls back to the base model tokenizer when an adapter directory has no tokenizer configuration. Documentation tests will enforce correct working-directory transitions and prevent unexecuted GPU gates from being described as complete. Repository hygiene tests will reject tracked Python and pytest artifacts.

**Tech Stack:** Python 3.11, pytest, Transformers, PEFT, uv, Git.

---

### Task 1: Make LoRA Smoke Artifacts Evaluatable

**Files:**
- Modify: `post_train/tests/test_evaluate_model_loader.py`
- Modify: `post_train_v2/tests/env/test_env_scripts.py`
- Modify: `post_train/scripts/eval/evaluate_model.py`
- Modify: `post_train_v2/scripts/env/smoke_trl_peft.py`

- [ ] Add a failing evaluator test proving an adapter without
  `tokenizer_config.json` loads the tokenizer from `base_model_path`.
- [ ] Add a failing AST test requiring
  `tokenizer.save_pretrained(adapter_dir)` in the TRL/PEFT smoke script.
- [ ] Run the focused tests and confirm both fail for the intended reasons.
- [ ] Implement tokenizer fallback and save tokenizer artifacts beside the
  smoke adapter.
- [ ] Run focused tests and confirm they pass.

### Task 2: Correct Remote Acceptance Documentation

**Files:**
- Modify: `post_train_v2/tests/env/test_runtime_manifest.py`
- Modify: `post_train_v2/docs/environment_setup.md`
- Modify: `post_train_v2/docs/superpowers/specs/2026-06-12-flash-attention-environment-design.md`

- [ ] Add failing tests requiring an explicit return to the repository root
  before repository-root pytest commands.
- [ ] Add a failing test rejecting claims that remote Level 1 was already
  executed.
- [ ] Correct the runbook and design status language.
- [ ] Run documentation consistency tests.

### Task 3: Remove Generated Repository Artifacts

**Files:**
- Modify: `.gitignore`
- Modify: `post_train_v2/.gitignore`
- Create: `post_train_v2/tests/env/test_repository_hygiene.py`
- Delete: tracked `__pycache__`, `*.pyc`, `.pytest_cache`, and `.pytest_tmp`
  artifacts.

- [ ] Add a failing Git hygiene test rejecting tracked generated artifacts.
- [ ] Add repository-wide ignore patterns.
- [ ] Remove only generated artifacts already tracked by Git.
- [ ] Run the hygiene test and inspect `git status`.

### Task 4: Final Verification

- [ ] Run `python -m pytest -q post_train_v2/tests/env post_train/tests`.
- [ ] Run stale-version, direct-loader, wheel, and generated-artifact scans.
- [ ] Run `git diff --check`.
- [ ] Record remote-only work as pending: artifact upload, `uv lock`,
  Level 1 GPU gates, and deferred Level 2 GRPO integration.
