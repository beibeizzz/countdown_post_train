# V2 Foundations, Data, and Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make V2 own the Countdown domain, configuration, Manifest V2, deterministic data warehouse, common evaluation, and tracking contracts required by every later training phase.

**Architecture:** Small framework-neutral modules implement exact arithmetic, schemas, hashing, atomic publication, and deterministic sampling. CLI scripts remain thin adapters. The existing V2 Teacher engine is preserved but migrated from V1 imports and its stage-specific manifest is embedded into the unified Manifest V2 envelope.

**Tech Stack:** Python 3.11, dataclasses, `fractions.Fraction`, AST, PyYAML, pandas, PyArrow, Transformers 4.53.2, PEFT 0.15.2, W&B 0.21.4, pytest.

---

## File Map

Create:

- `post_train_v2/src/config/loading.py`: YAML loading, repository-relative path resolution, typed key checks.
- `post_train_v2/src/artifacts/atomic.py`: same-directory atomic JSON/JSONL writes.
- `post_train_v2/src/artifacts/hashing.py`: SHA-256 for bytes, files, canonical JSON, and configuration snapshots.
- `post_train_v2/src/artifacts/manifest.py`: Manifest V2 dataclass, validation, load, and publish.
- `post_train_v2/src/countdown/prompts.py`: solution and forced-wrong prompts plus chat messages.
- `post_train_v2/src/countdown/solver.py`: exact rational solver and expression complexity.
- `post_train_v2/src/countdown/validation.py`: answer extraction and exact validation.
- `post_train_v2/src/countdown/bucketing.py`: number-count and expression-complexity buckets.
- `post_train_v2/src/countdown/sampling.py`: deterministic balanced stratified sampling.
- `post_train_v2/src/data/schema.py`: normalized, SFT/RFT, DPO, and verl record validators.
- `post_train_v2/src/data/source.py`: raw parquet/JSON normalization.
- `post_train_v2/src/data/splits.py`: validation, fixed evaluation, SFT, and GRPO split builders.
- `post_train_v2/src/evaluation/scoring.py`: per-response and aggregate metrics.
- `post_train_v2/src/evaluation/model_loading.py`: full-model and LoRA loading.
- `post_train_v2/src/evaluation/generation.py`: deterministic generation capped at 256 new tokens.
- `post_train_v2/src/tracking/wandb.py`: rank-aware W&B naming and logging.
- `post_train_v2/src/generation/seeding.py`: stable per-request seed derivation.
- `post_train_v2/src/generation/vllm_client.py`: chat-based vLLM client with
  one `SamplingParams` object per prompt.
- `post_train_v2/scripts/data/build_source.py`: source warehouse CLI.
- `post_train_v2/scripts/data/build_splits.py`: accepted-pool split CLI.
- `post_train_v2/scripts/eval/evaluate_model.py`: common evaluation CLI.
- `post_train_v2/configs/common/paths.yaml`: default local model/data paths.
- `post_train_v2/configs/common/eval.yaml`: fixed decoding configuration.
- `post_train_v2/configs/common/tracking.yaml`: optional W&B defaults.
- `post_train_v2/configs/data/build_source.yaml`: source build configuration.
- `post_train_v2/configs/data/build_splits.yaml`: 8k/4k split configuration.
- package markers and README files for new functional directories.
- unit, data, and evaluation tests named in the tasks below.

Modify:

- `post_train_v2/src/generation/parallel_vllm.py`: import V2 generation types.
- `post_train_v2/src/generation/teacher_state.py`: use V2 Countdown, config, artifact, and Manifest V2 modules.
- `post_train_v2/scripts/generation/build_teacher_pool.py`: publish partial-failure Manifest V2 on exhausted source.
- existing generation tests: assert the new manifest envelope without changing the worker protocol.

## Task 1: Package Boundaries and Config Loading

**Files:**

- Create: `post_train_v2/src/config/__init__.py`
- Create: `post_train_v2/src/config/loading.py`
- Create: `post_train_v2/tests/unit/test_config_loading.py`

- [ ] **Step 1: Write failing configuration tests**

```python
def test_resolve_repo_path_is_independent_of_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert resolve_repo_path("post_train_v2/data/x.jsonl") == (
        REPO_ROOT / "post_train_v2/data/x.jsonl"
    )


def test_require_keys_reports_all_missing_keys():
    with pytest.raises(ValueError, match=r"missing keys: a, c"):
        require_keys({"b": 1}, "a", "b", "c")
```

- [ ] **Step 2: Verify RED**

Run:

```bash
python -m pytest -q post_train_v2/tests/unit/test_config_loading.py
```

Expected: import failure for `post_train_v2.src.config.loading`.

- [ ] **Step 3: Implement the config API**

```python
REPO_ROOT = Path(__file__).resolve().parents[3]


def load_yaml(path: str | Path) -> dict[str, Any]:
    resolved = resolve_repo_path(path)
    value = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML config must be a mapping: {resolved}")
    return value


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (REPO_ROOT / candidate).resolve()


def require_keys(mapping: Mapping[str, Any], *keys: str) -> None:
    missing = sorted(key for key in keys if key not in mapping)
    if missing:
        raise ValueError(f"missing keys: {', '.join(missing)}")
```

- [ ] **Step 4: Verify GREEN**

Run the command from Step 2. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add post_train_v2/src/config post_train_v2/tests/unit/test_config_loading.py
git commit -m "feat: add v2 configuration foundation"
```

## Task 2: Atomic Artifacts and Manifest V2

**Files:**

- Create: `post_train_v2/src/artifacts/{__init__.py,atomic.py,hashing.py,manifest.py,README.md}`
- Create: `post_train_v2/tests/unit/test_atomic_artifacts.py`
- Create: `post_train_v2/tests/unit/test_manifest_v2.py`

- [ ] **Step 1: Write failing atomic and manifest tests**

```python
def test_publish_json_replaces_atomically(tmp_path):
    path = tmp_path / "artifact.json"
    publish_json(path, {"value": 1})
    assert json.loads(path.read_text()) == {"value": 1}
    assert not list(tmp_path.glob("*.tmp"))


def test_manifest_round_trip_rejects_changed_parent_hash(tmp_path):
    manifest = ManifestV2.build(
        artifact_type="dataset",
        stage="build_source",
        files=[ArtifactFile("source.jsonl", "abc", 4, 1, {"id": "string"})],
        parents=[ParentArtifact("raw-train", "def")],
        config={"seed": 42},
        stage_metadata={"num_source": 1},
    )
    path = tmp_path / "manifest.json"
    publish_manifest(path, manifest)
    loaded = load_manifest(path)
    loaded.require_parent("raw-train", "bad")
```

Expected: the second test raises `ValueError` containing `parent hash`.

- [ ] **Step 2: Verify RED**

```bash
python -m pytest -q \
  post_train_v2/tests/unit/test_atomic_artifacts.py \
  post_train_v2/tests/unit/test_manifest_v2.py
```

Expected: import failures.

- [ ] **Step 3: Implement canonical hashing and atomic publication**

Use canonical JSON bytes:

```python
def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
```

Write temporary files beside the destination with
`tempfile.NamedTemporaryFile(dir=path.parent, delete=False)`, call
`os.fsync()`, then `os.replace()`. Manifest fields must match Section 5.2 of
the approved design, including `schema_version=2`, stable `artifact_id`,
configuration hash, parents, file metadata, Git revision, runtime versions,
seed, and `stage_metadata`.

- [ ] **Step 4: Verify GREEN**

Run Step 2. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add post_train_v2/src/artifacts post_train_v2/tests/unit/test_atomic_artifacts.py post_train_v2/tests/unit/test_manifest_v2.py
git commit -m "feat: add manifest v2 artifact contract"
```

## Task 3: V2 Countdown Core

**Files:**

- Create: `post_train_v2/src/countdown/{__init__.py,prompts.py,solver.py,validation.py,bucketing.py,sampling.py,README.md}`
- Create: `post_train_v2/tests/unit/test_countdown_core.py`
- Create: `post_train_v2/tests/unit/test_countdown_sampling.py`

- [ ] **Step 1: Write exact arithmetic and prompt tests**

```python
def test_fractional_intermediate_is_valid():
    result = validate_countdown_expression("(85-(45/(69-74)))", [85, 45, 69, 74], 94)
    assert result.ok is True
    assert result.value == Fraction(94, 1)


def test_prompt_contract():
    prompt = build_solution_prompt([1, 1, 1, 1], 4)
    assert "Use each number exactly once" in prompt
    assert "Only use +, -, *, / and parentheses" in prompt
    assert "<answer> equation </answer>" in prompt
    assert "Division must be exact" not in prompt
```

- [ ] **Step 2: Write sampling tests**

Assert duplicate IDs fail, the same seed produces byte-identical selected ID
lists, validation IDs are removed from training candidates, and the fixed 50
is a subset of `val_200`.

- [ ] **Step 3: Verify RED**

```bash
python -m pytest -q \
  post_train_v2/tests/unit/test_countdown_core.py \
  post_train_v2/tests/unit/test_countdown_sampling.py
```

Expected: import failures.

- [ ] **Step 4: Implement the V2 modules**

Port the validated V1 behavior but make `ValidationResult.value` a
`Fraction | None` internally. Serialize values only at artifact boundaries:

```python
def serialize_fraction(value: Fraction | None) -> str | None:
    return None if value is None else f"{value.numerator}/{value.denominator}"
```

Reject booleans, floats, unary operators, exponentiation, function calls, and
unused or repeated input numbers. Preserve fully parenthesized solver output
as diagnostic `gold_expr`.

- [ ] **Step 5: Verify GREEN**

Run Step 3. Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add post_train_v2/src/countdown post_train_v2/tests/unit/test_countdown_core.py post_train_v2/tests/unit/test_countdown_sampling.py
git commit -m "feat: add v2 countdown domain core"
```

## Task 4: Canonical Record Schemas

**Files:**

- Create: `post_train_v2/src/data/{__init__.py,schema.py,README.md}`
- Create: `post_train_v2/tests/data/test_record_schemas.py`

- [ ] **Step 1: Write failing schema tests**

```python
def test_normalized_source_requires_stable_fields():
    row = normalized_row()
    assert validate_normalized_source(row)["target"] == 24
    del row["source_index"]
    with pytest.raises(ValueError, match="source_index"):
        validate_normalized_source(row)


def test_verl_ground_truth_is_structured():
    row = validate_verl_record(verl_row())
    assert row["reward_model"]["ground_truth"] == {
        "numbers": [1, 2, 3, 4],
        "target": 24,
    }
```

- [ ] **Step 2: Verify RED**

```bash
python -m pytest -q post_train_v2/tests/data/test_record_schemas.py
```

- [ ] **Step 3: Implement strict validators**

Provide:

```python
validate_normalized_source(row)
validate_sft_record(row)
validate_dpo_record(row)
validate_verl_record(row)
validate_unique_ids(rows, label)
```

Return normalized copies; never mutate caller dictionaries. DPO
`rejected_category` must be one of `wrong_value`, `number_mismatch`,
`invalid_expression`, `missing_answer_tag`, or `truncated`.

- [ ] **Step 4: Verify GREEN and commit**

```bash
python -m pytest -q post_train_v2/tests/data/test_record_schemas.py
git add post_train_v2/src/data post_train_v2/tests/data/test_record_schemas.py
git commit -m "feat: define v2 dataset schemas"
```

## Task 5: Build the Normalized Source Warehouse

**Files:**

- Create: `post_train_v2/src/data/source.py`
- Create: `post_train_v2/scripts/data/build_source.py`
- Create: `post_train_v2/configs/data/build_source.yaml`
- Create: `post_train_v2/tests/data/test_build_source.py`
- Create: `post_train_v2/scripts/data/README.md`

- [ ] **Step 1: Write failing source-build tests**

Use a small parquet fixture and raw test JSON. Assert:

```python
assert [row["id"] for row in source] == ["train-000001", "train-000002"]
assert source[0]["source_index"] == 1
assert source[0]["prompt"] == build_solution_prompt(source[0]["numbers"], source[0]["target"])
assert test_rows[0]["id"] == "test-000007"
assert manifest.stage == "build_source"
```

Also assert `--limit 1` preserves original source indexing and an unsolved
test row fails rather than being silently omitted.

- [ ] **Step 2: Verify RED**

```bash
python -m pytest -q post_train_v2/tests/data/test_build_source.py
```

- [ ] **Step 3: Implement source normalization**

Expose:

```python
def build_train_source(frame: pd.DataFrame) -> tuple[list[dict], list[dict]]
def build_test_source(rows: list[dict]) -> list[dict]
def run_build_source(config_path: Path, limit: int | None) -> ManifestV2
```

The production configuration reads:

```yaml
train_input: datasets/raw_train.parquet
test_input: datasets/raw_test.json
output_dir: post_train_v2/data/processed
seed: 42
```

Write:

```text
post_train_v2/data/processed/source_all.jsonl
post_train_v2/data/processed/solvable_train.jsonl
post_train_v2/data/processed/unsolved_train.jsonl
post_train_v2/data/processed/test_solved.jsonl
post_train_v2/data/processed/manifest.json
```

- [ ] **Step 4: Verify CLI independence**

```bash
python post_train_v2/scripts/data/build_source.py --help
```

Expected: exit 0 from any current working directory.

- [ ] **Step 5: Verify GREEN and commit**

```bash
python -m pytest -q post_train_v2/tests/data/test_build_source.py
git add post_train_v2/src/data/source.py post_train_v2/scripts/data post_train_v2/configs/data post_train_v2/tests/data/test_build_source.py
git commit -m "feat: build v2 normalized source data"
```

## Task 6: Freeze Validation and Training Splits

**Files:**

- Create: `post_train_v2/src/data/splits.py`
- Create: `post_train_v2/scripts/data/build_splits.py`
- Create: `post_train_v2/configs/data/build_splits.yaml`
- Create: `post_train_v2/tests/data/test_build_splits.py`

- [ ] **Step 1: Write failing deterministic split tests**

Assert `val_200` is drawn before Teacher data, all validation IDs are excluded
from candidates, fixed 50 is a subset, and accepted-pool 8k/4k samples use
separate derived seeds:

```python
assert set(ids(val_200)).isdisjoint(ids(train_candidates))
assert set(ids(eval_50)) <= set(ids(val_200))
assert len(sft_rows) == 8000
assert len(grpo_rows) == 4000
assert sha256_rows(run_one) == sha256_rows(run_two)
```

Insufficient accepted rows must raise an error naming the requested split.

- [ ] **Step 2: Verify RED**

```bash
python -m pytest -q post_train_v2/tests/data/test_build_splits.py
```

- [ ] **Step 3: Implement two CLI modes**

The split configuration contains:

```yaml
seed: 42
source_data: post_train_v2/data/processed/solvable_train.jsonl
teacher_accepted: post_train_v2/data/teacher_rollouts/teacher_accepted_20k.jsonl
val_size: 200
eval_size: 50
sft_size: 8000
grpo_size: 4000
```

```bash
python post_train_v2/scripts/data/build_splits.py \
  --config post_train_v2/configs/data/build_splits.yaml \
  validation

python post_train_v2/scripts/data/build_splits.py \
  --config post_train_v2/configs/data/build_splits.yaml \
  accepted
```

`validation` writes `val_200.jsonl`, `eval_50.jsonl`, and
`train_candidates.jsonl` after removing all validation IDs. `accepted` reads
only a complete Teacher artifact and writes
`sft_train_8k.jsonl` and `grpo_train_4k.jsonl`.

- [ ] **Step 4: Verify GREEN and commit**

```bash
python -m pytest -q post_train_v2/tests/data/test_build_splits.py
git add post_train_v2/src/data/splits.py post_train_v2/scripts/data/build_splits.py post_train_v2/configs/data/build_splits.yaml post_train_v2/tests/data/test_build_splits.py
git commit -m "feat: add deterministic v2 data splits"
```

## Task 7: Common Evaluation and Model Loading

**Files:**

- Create: `post_train_v2/src/evaluation/{__init__.py,scoring.py,model_loading.py,generation.py,README.md}`
- Create: `post_train_v2/scripts/eval/{evaluate_model.py,README.md}`
- Create: `post_train_v2/configs/common/eval.yaml`
- Create: `post_train_v2/tests/evaluation/test_scoring.py`
- Create: `post_train_v2/tests/evaluation/test_model_loading.py`
- Create: `post_train_v2/tests/evaluation/test_evaluate_cli.py`

- [ ] **Step 1: Write failing scoring tests**

```python
def test_metrics_include_truncation_rate():
    metrics = aggregate_rows([
        scored(correct=True, format_ok=True, tokens=10, truncated=False),
        scored(correct=False, format_ok=False, tokens=256, truncated=True),
    ])
    assert metrics["accuracy"] == 0.5
    assert metrics["format_rate"] == 0.5
    assert metrics["truncated_count"] == 1
    assert metrics["truncated_rate"] == 0.5
```

- [ ] **Step 2: Write loader tests**

Patch Transformers/PEFT classes and assert full models and LoRA adapters load
with BF16 and `attn_implementation="flash_attention_2"`. Adapter loading must
accept explicit `--base-model-path` and otherwise read
`base_model_name_or_path`.

- [ ] **Step 3: Verify RED**

```bash
python -m pytest -q post_train_v2/tests/evaluation
```

- [ ] **Step 4: Implement deterministic evaluation**

The generation path must call the Qwen chat template with
`enable_thinking=False`, use `do_sample=False`, and cap
`max_new_tokens=256`. The CLI writes `samples.jsonl`, `metrics.json`, and
Manifest V2.

- [ ] **Step 5: Verify GREEN and commit**

```bash
python -m pytest -q post_train_v2/tests/evaluation
git add post_train_v2/src/evaluation post_train_v2/scripts/eval post_train_v2/configs/common/eval.yaml post_train_v2/tests/evaluation
git commit -m "feat: add common v2 evaluation"
```

## Task 8: Rank-Aware W&B Utilities

**Files:**

- Create: `post_train_v2/src/tracking/{__init__.py,wandb.py,README.md}`
- Create: `post_train_v2/configs/common/tracking.yaml`
- Create: `post_train_v2/tests/unit/test_tracking.py`

- [ ] **Step 1: Write failing tracking tests**

Assert rank 0 alone initializes W&B, suffixes include timestamp and short Git
revision, and disabled tracking imports no W&B module.

- [ ] **Step 2: Implement**

```python
def make_run_name(base: str, now: datetime, git_revision: str) -> str:
    return f"{base}-{now:%Y%m%d-%H%M%S}-{git_revision[:7]}"


def init_run(config: Mapping[str, Any], *, rank: int, stage: str):
    if rank != 0 or not config.get("enabled", False):
        return None
    return wandb.init(
        project=config.get("project", "countdown-post-train-v2"),
        entity=config.get("entity"),
        group=config.get("group"),
        name=make_run_name(config.get("run_name", stage), utcnow(), git_revision()),
        config=dict(config),
    )
```

- [ ] **Step 3: Verify and commit**

```bash
python -m pytest -q post_train_v2/tests/unit/test_tracking.py
git add post_train_v2/src/tracking post_train_v2/configs/common/tracking.yaml post_train_v2/tests/unit/test_tracking.py
git commit -m "feat: add rank-aware v2 tracking"
```

## Task 9: Migrate Existing Teacher to V2 Foundations

**Files:**

- Create: `post_train_v2/src/generation/seeding.py`
- Create: `post_train_v2/src/generation/vllm_client.py`
- Modify: `post_train_v2/src/generation/parallel_vllm.py`
- Modify: `post_train_v2/src/generation/teacher_state.py`
- Modify: `post_train_v2/scripts/generation/build_teacher_pool.py`
- Modify: `post_train_v2/configs/generation/teacher_rollout_2gpu.yaml`
- Modify: `post_train_v2/configs/generation/teacher_rollout_2gpu_smoke.yaml`
- Modify: `post_train_v2/tests/generation/test_parallel_vllm.py`
- Create: `post_train_v2/tests/generation/test_vllm_client.py`
- Modify: `post_train_v2/tests/generation/test_teacher_state.py`
- Modify: `post_train_v2/tests/generation/test_build_teacher_pool.py`

- [ ] **Step 1: Add failing vLLM client and seed tests**

```python
seed = derive_request_seed(
    global_seed=42,
    stage="teacher",
    sample_id="train-000123",
    rollout_index=0,
)
assert seed == derive_request_seed(42, "teacher", "train-000123", 0)
assert seed != derive_request_seed(42, "teacher", "train-000123", 1)
```

Patch vLLM and assert `LLM.chat()` receives conversations in source order,
`chat_template_kwargs={"enable_thinking": False}`, and a list of
`SamplingParams` objects whose seeds correspond one-to-one with prompts.

- [ ] **Step 2: Add failing import-isolation and Manifest V2 tests**

Assert no `post_train.` import remains in V2 generation modules. When source
exhausts before 20k, assert exit code 2, `completed=false`, actual accepted
count in `stage_metadata`, and no complete accepted-pool artifact ID.

- [ ] **Step 3: Verify RED**

```bash
python -m pytest -q post_train_v2/tests/generation
```

- [ ] **Step 4: Implement the V2 generation contract**

```python
@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    seed: int


def derive_request_seed(
    global_seed: int,
    stage: str,
    sample_id: str,
    rollout_index: int,
) -> int:
    payload = f"{global_seed}|{stage}|{sample_id}|{rollout_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")
```

`VLLMGenerator.generate_with_metadata()` accepts ordered
`GenerationRequest` records and builds one
`SamplingParams(seed=request.seed)` per request before calling
`self.llm.chat(...)`.

- [ ] **Step 5: Switch imports and wrap stage metadata**

Keep worker protocol, transaction journal, source-order stopping, and output
filenames unchanged. Replace the Teacher manifest root with Manifest V2 and
place existing generation-contract, worker, resume, and shard details under
`stage_metadata`. Change the production Teacher input to
`post_train_v2/data/processed/train_candidates.jsonl` and its output to
`post_train_v2/data/teacher_rollouts`; the smoke config uses only V2 fixture
paths. Teacher requests derive seeds with stage `teacher` and rollout index
zero.

- [ ] **Step 6: Verify GREEN and commit**

```bash
python -m pytest -q post_train_v2/tests/generation
git add post_train_v2/src/generation post_train_v2/scripts/generation post_train_v2/configs/generation post_train_v2/tests/generation
git commit -m "refactor: integrate teacher generation with v2 foundations"
```

## Task 10: Phase Documentation and Gate

**Files:**

- Modify: `post_train_v2/README.md`
- Create or modify README files in every directory created in this plan.
- Create: `post_train_v2/docs/runbooks/data_and_evaluation.md`

- [ ] **Step 1: Document exact commands and artifacts**

Include source build, validation split, Teacher resume, accepted split,
full-model evaluation, LoRA evaluation, and manifest inspection commands.

- [ ] **Step 2: Run the phase gate**

```bash
python -m pytest -q \
  post_train_v2/tests/unit \
  post_train_v2/tests/data \
  post_train_v2/tests/evaluation \
  post_train_v2/tests/generation
python -m pytest -q post_train_v2/tests/env/test_repository_hygiene.py
git diff --check
```

Expected: PASS and no whitespace errors.

- [ ] **Step 3: Commit**

```bash
git add post_train_v2/README.md post_train_v2/docs/runbooks post_train_v2
git commit -m "docs: add v2 data and evaluation runbook"
```
