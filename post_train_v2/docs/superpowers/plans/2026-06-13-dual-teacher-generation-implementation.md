# Dual-GPU Teacher Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a production-safe V2 Teacher accepted-pool entrypoint using two persistent Qwen3-8B vLLM TP1 workers while preserving source order, exact validation, resumability, and the existing output warehouse.

**Architecture:** A coordinator owns the shared lock, state recovery, ordered validation, transaction journal, and output files. Two spawned worker processes each bind one CUDA device, use a distinct vLLM cache, load one persistent model, and return position-tagged responses. Existing task prompts, vLLM chat behavior, and Countdown validation remain shared under `post_train/src/countdown`.

**Tech Stack:** Python 3.11, multiprocessing `spawn`, vLLM 0.9.1, PyYAML, JSONL, SHA-256, pytest.

---

## File Map

Create:

- `post_train/src/countdown/output_lock.py`: shared lock implementation for legacy and V2 Teacher writers.
- `post_train/tests/test_output_lock.py`: lock ownership and recovery tests.
- `post_train_v2/src/__init__.py`: V2 source package marker.
- `post_train_v2/src/generation/__init__.py`: generation package marker.
- `post_train_v2/src/generation/README.md`: generation module boundaries and extension rules.
- `post_train_v2/src/generation/parallel_vllm.py`: worker protocol and lifecycle.
- `post_train_v2/src/generation/teacher_state.py`: config contract, resume validation, journal, and commit logic.
- `post_train_v2/scripts/generation/build_teacher_pool.py`: V2 coordinator CLI.
- `post_train_v2/configs/generation/teacher_rollout_2gpu.yaml`: production config.
- `post_train_v2/configs/generation/teacher_rollout_2gpu_smoke.yaml`: remote smoke config.
- `post_train_v2/tests/generation/__init__.py`: test package marker.
- `post_train_v2/tests/generation/README.md`: local and remote test boundary.
- `post_train_v2/tests/generation/test_parallel_vllm.py`: worker and ordering tests.
- `post_train_v2/tests/generation/test_teacher_state.py`: lock-independent state and transaction tests.
- `post_train_v2/tests/generation/test_build_teacher_pool.py`: coordinator tests.

Modify:

- `post_train/src/countdown/generation.py`: optional vLLM memory, length, and seed arguments.
- `post_train/tests/test_generation.py`: constructor forwarding tests.
- `post_train/scripts/data/build_teacher_pool.py`: shared output lock and V2-state refusal.
- `post_train/tests/test_build_teacher_pool.py`: legacy lock and ownership tests.
- `post_train_v2/scripts/generation/README.md`: production and smoke commands.
- `post_train_v2/docs/environment_setup.md`: add the V2 Teacher smoke gate after environment acceptance.
- `post_train_v2/docs/next_full_workflow.md`: replace the legacy production Teacher command with the V2 command.

## Task 1: Shared Exclusive Output Lock

**Files:**

- Create: `post_train/src/countdown/output_lock.py`
- Create: `post_train/tests/test_output_lock.py`

- [ ] **Step 1: Write failing lock tests**

Add tests for:

```python
def test_lock_acquire_writes_owner_metadata(tmp_path):
    lock = OutputLock(
        tmp_path / ".teacher_pool.lock",
        config_path=tmp_path / "config.yaml",
        output_dir=tmp_path,
        topology="dual_tp1",
        hostname="host-a",
        pid=123,
        process_alive=lambda pid: False,
    )
    lock.acquire()
    payload = json.loads(lock.path.read_text(encoding="utf-8"))
    assert payload["pid"] == 123
    assert payload["hostname"] == "host-a"
    assert payload["owner_token"] == lock.owner_token


def test_live_local_lock_is_rejected(tmp_path):
    write_lock(tmp_path, hostname="host-a", pid=123, owner_token="old")
    lock = make_lock(tmp_path, process_alive=lambda pid: True)
    with pytest.raises(RuntimeError, match="active Teacher writer"):
        lock.acquire()


def test_stale_local_lock_requires_explicit_recovery(tmp_path):
    write_lock(tmp_path, hostname="host-a", pid=123, owner_token="old")
    lock = make_lock(tmp_path, process_alive=lambda pid: False)
    with pytest.raises(RuntimeError, match="recover-stale-lock"):
        lock.acquire(recover_stale=False)
    lock.acquire(recover_stale=True)


def test_foreign_host_lock_is_never_auto_recovered(tmp_path):
    write_lock(tmp_path, hostname="host-b", pid=123, owner_token="old")
    lock = make_lock(tmp_path, hostname="host-a")
    with pytest.raises(RuntimeError, match="different host"):
        lock.acquire(recover_stale=True)


def test_release_does_not_remove_another_owners_lock(tmp_path):
    lock = make_lock(tmp_path)
    lock.acquire()
    replace_owner_token(lock.path, "replacement")
    lock.release()
    assert lock.path.exists()
```

- [ ] **Step 2: Run the lock tests and verify RED**

Run:

```bash
python -m pytest -q post_train/tests/test_output_lock.py
```

Expected: collection or import failure because `output_lock.py` does not yet
exist.

- [ ] **Step 3: Implement `OutputLock`**

Implement:

```python
@dataclass
class OutputLock:
    path: Path
    config_path: Path
    output_dir: Path
    topology: str
    hostname: str = field(default_factory=socket.gethostname)
    pid: int = field(default_factory=os.getpid)
    process_alive: Callable[[int], bool] = process_is_alive
    owner_token: str = field(default_factory=lambda: uuid.uuid4().hex)
```

Use `os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)` for exclusive
creation. Serialize schema version, PID, hostname, UTC start time, config path,
output directory, topology, and owner token. Treat `os.kill(pid, 0)` success
or `PermissionError` as alive. On release, reread the file and unlink only
when its owner token matches. Add `acquire(recover_stale=False)`, `release()`,
and context-manager methods with the behavior covered by Step 1.

- [ ] **Step 4: Run lock tests and shared tests**

Run:

```bash
python -m pytest -q \
  post_train/tests/test_output_lock.py \
  post_train/tests/test_config_io.py
```

Expected: PASS.

- [ ] **Step 5: Commit the shared lock**

```bash
git add post_train/src/countdown/output_lock.py post_train/tests/test_output_lock.py
git commit -m "feat: add shared teacher output lock"
```

## Task 2: Extend the Shared vLLM Generator Contract

**Files:**

- Modify: `post_train/src/countdown/generation.py`
- Modify: `post_train/tests/test_generation.py`

- [ ] **Step 1: Write a failing constructor-forwarding test**

Inject a fake `vllm.LLM` and assert:

```python
generator = VLLMGenerator(
    "/models/qwen3-8b",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.8,
    max_model_len=512,
    seed=7,
)

assert captured_kwargs == {
    "model": "/models/qwen3-8b",
    "tensor_parallel_size": 1,
    "trust_remote_code": True,
    "gpu_memory_utilization": 0.8,
    "max_model_len": 512,
    "seed": 7,
}
```

Add a second test proving the existing one-argument constructor does not pass
the three optional keywords.

- [ ] **Step 2: Run the tests and verify RED**

```bash
python -m pytest -q post_train/tests/test_generation.py
```

Expected: constructor rejects the new keyword arguments.

- [ ] **Step 3: Implement optional constructor arguments**

Change the signature to:

```python
def __init__(
    self,
    model_path: str,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float | None = None,
    max_model_len: int | None = None,
    seed: int | None = None,
):
```

Build `llm_kwargs` with existing required fields and add each optional field
only when it is not `None`. Do not change `generate_with_metadata()` or chat
template behavior.

- [ ] **Step 4: Run generation and Teacher regression tests**

```bash
python -m pytest -q \
  post_train/tests/test_generation.py \
  post_train/tests/test_build_teacher_pool.py
```

Expected: PASS.

- [ ] **Step 5: Commit the generator extension**

```bash
git add post_train/src/countdown/generation.py post_train/tests/test_generation.py
git commit -m "feat: configure vllm generator runtime"
```

## Task 3: Deterministic Two-Worker vLLM Orchestration

**Files:**

- Create: `post_train_v2/src/__init__.py`
- Create: `post_train_v2/src/generation/__init__.py`
- Create: `post_train_v2/src/generation/README.md`
- Create: `post_train_v2/src/generation/parallel_vllm.py`
- Create: `post_train_v2/tests/generation/__init__.py`
- Create: `post_train_v2/tests/generation/README.md`
- Create: `post_train_v2/tests/generation/test_parallel_vllm.py`

- [ ] **Step 1: Write failing deterministic split and merge tests**

Define the intended public API in tests:

```python
items = [
    PositionedPrompt(position=index, prompt=f"prompt-{index}")
    for index in range(5)
]
left, right = split_contiguous(items)
assert [item.position for item in left] == [0, 1, 2]
assert [item.position for item in right] == [3, 4]

merged = merge_worker_results(
    batch_id=9,
    expected_positions=[0, 1, 2, 3],
    messages=[
        WorkerResult(1, 9, [(2, "r2"), (3, "r3")]),
        WorkerResult(0, 9, [(0, "r0"), (1, "r1")]),
    ],
)
assert merged == [(0, "r0"), (1, "r1"), (2, "r2"), (3, "r3")]
```

Add rejection tests for:

- wrong batch ID;
- duplicate worker result;
- duplicate position;
- missing position;
- unknown position;
- result-count mismatch.

- [ ] **Step 2: Write failing worker-environment tests**

Call `worker_main()` with injected `generator_factory`, request queue, and
result queue. Assert the factory observes:

```python
os.environ["CUDA_VISIBLE_DEVICES"] == "1"
os.environ["VLLM_CACHE_ROOT"] == "/tmp/cache/gpu1"
```

Assert the factory receives TP1, memory utilization, model length, and seed.
Assert an empty shard returns an empty `WorkerResult` without calling
`generate()`.

- [ ] **Step 3: Write failing lifecycle tests**

Use fake process and queue adapters to verify:

- both workers must report ready before `start()` returns;
- startup timeout closes both workers;
- runtime worker error aborts the batch;
- dead worker detection does not wait for the full timeout;
- `close()` sends stop, joins, then terminates stubborn workers;
- no heavy `torch` or `vllm` import exists at module scope.

- [ ] **Step 4: Run the orchestration tests and verify RED**

```bash
python -m pytest -q post_train_v2/tests/generation/test_parallel_vllm.py
```

Expected: import failure because the orchestration module does not exist.

- [ ] **Step 5: Implement protocol dataclasses and pure helpers**

Create:

```python
@dataclass(frozen=True)
class PositionedPrompt:
    position: int
    prompt: str


@dataclass(frozen=True)
class WorkerSpec:
    worker_index: int
    device: int
    cache_root: str


@dataclass(frozen=True)
class WorkerRequest:
    batch_id: int
    items: tuple[PositionedPrompt, ...]


@dataclass(frozen=True)
class WorkerReady:
    worker_index: int


@dataclass(frozen=True)
class WorkerResult:
    worker_index: int
    batch_id: int
    items: tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class WorkerError:
    worker_index: int
    batch_id: int | None
    message: str
    traceback: str
```

Implement `split_contiguous()` and `merge_worker_results()` as pure,
fully validated functions.

- [ ] **Step 6: Implement `worker_main()`**

Set environment variables and create the cache directory before importing
`VLLMGenerator`. Initialize one generator, emit `WorkerReady`, then process
requests until a stop sentinel is received. Convert `GenerationConfig` plus
prompt text into ordered position-response tuples. Catch all exceptions,
emit `WorkerError` with `traceback.format_exc()`, then raise `SystemExit(1)`
so multiprocessing records a nonzero child exit code.

- [ ] **Step 7: Implement `ParallelVLLMEngine`**

Use `multiprocessing.get_context("spawn")`, one request queue per worker, and
one shared response queue. Expose `ParallelVLLMEngine.start()`,
`ParallelVLLMEngine.generate(batch_id, items)`,
`ParallelVLLMEngine.close()`, and context-manager entry/exit methods.
`generate()` returns `list[tuple[int, str]]` in source-position order.

Poll the response queue in short intervals while checking process exit codes.
Use the configured timeout as a single deadline for startup or one batch.

- [ ] **Step 8: Add generation module READMEs**

Document in `post_train_v2/src/generation/README.md` that
`parallel_vllm.py` owns process orchestration and `teacher_state.py` owns
persistence, with no solver or output writes in workers. Document in
`post_train_v2/tests/generation/README.md` that fake-worker tests run locally
and real dual-GPU vLLM acceptance is performed only by the remote smoke gate.

- [ ] **Step 9: Run orchestration tests**

```bash
python -m pytest -q post_train_v2/tests/generation/test_parallel_vllm.py
```

Expected: PASS.

- [ ] **Step 10: Commit orchestration**

```bash
git add \
  post_train_v2/src/__init__.py \
  post_train_v2/src/generation/__init__.py \
  post_train_v2/src/generation/README.md \
  post_train_v2/src/generation/parallel_vllm.py \
  post_train_v2/tests/generation
git commit -m "feat: add dual vllm worker orchestration"
```

## Task 4: Transactional Teacher State and Resume Contract

**Files:**

- Create: `post_train_v2/src/generation/teacher_state.py`
- Create: `post_train_v2/tests/generation/test_teacher_state.py`

- [ ] **Step 1: Write failing contract and prefix tests**

Test:

```python
contract = build_generation_contract(config, source_sha256="abc")
assert contract == {
    "schema_version": 1,
    "source_sha256": "abc",
    "model_path": "/models/qwen3-8b",
    "topology": "dual_tp1",
    "batch_size": 64,
    "max_model_len": 512,
    "max_new_tokens": 256,
    "temperature": 0.2,
    "top_p": 0.95,
    "seed": 0,
    "enable_thinking": False,
}
assert fingerprint_contract(contract) == hashlib.sha256(
    json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
```

Create source IDs `[a, b, c, d]`, accepted `[a, c]`, and rejected `[b]`.
Assert prefix validation returns processed count 3. Assert `[a, d]` plus
`[b]` is rejected because it is not a prefix.

- [ ] **Step 2: Write failing legacy adoption tests**

Assert:

- existing rows without a V2 fingerprint require `adopt_legacy_state=True`;
- accepted rows are revalidated as correct;
- rejected rows are revalidated as incorrect;
- duplicate or unknown IDs fail;
- adopted rows must form a contiguous source prefix;
- accepted count cannot exceed the target.

- [ ] **Step 3: Write failing transaction tests**

Initialize old accepted/rejected/manifest files, then inject a replacement
function that fails:

1. after accepted replacement;
2. after rejected replacement;
3. after manifest replacement but before journal removal.

For each case, call `recover_transaction()` and assert exact restoration of
the old files, hashes, row counts, manifest payload, and journal removal.
Also test a first transaction where all three pre-state files were absent.

- [ ] **Step 4: Run state tests and verify RED**

```bash
python -m pytest -q post_train_v2/tests/generation/test_teacher_state.py
```

Expected: import failure because `teacher_state.py` does not exist.

- [ ] **Step 5: Implement configuration and hashing types**

Create:

```python
@dataclass(frozen=True)
class TeacherGenerationConfig:
    model_path: Path
    input_path: Path
    output_dir: Path
    devices: tuple[int, int]
    topology: str
    batch_size: int
    worker_timeout_seconds: float
    gpu_memory_utilization: float
    max_model_len: int
    max_new_tokens: int
    temperature: float
    top_p: float
    seed: int
    enable_thinking: bool
    stop_after_accepted: int
    cache_root: Path
    schema_version: int
```

Implement strict validation and `sha256_file()` with chunked binary reads.
Implement canonical contract JSON and fingerprinting exactly as specified.

- [ ] **Step 6: Implement prefix and resume validation**

Build a unique source-ID-to-position map. Combine accepted and rejected IDs,
sort by source position, and compare with the exact source prefix. Return:

```python
@dataclass(frozen=True)
class ResumeState:
    accepted: tuple[dict[str, Any], ...]
    rejected: tuple[dict[str, Any], ...]
    processed_count: int
    last_committed_position: int | None
    created_at: str
```

Validate manifest contract, source hash, output hashes, counts, target bounds,
and completion state. Implement explicit legacy adoption using the current
Countdown validator.

- [ ] **Step 7: Implement transaction journal and commit**

Create `TeacherStateStore` with `recover_transaction()`,
`load_resume_state(source_rows, config, adopt_legacy_state) -> ResumeState`,
and keyword-only
`commit(batch_id, submitted_start, submitted_stop, accepted, rejected,
manifest)`. The submitted range is half-open: `[submitted_start,
submitted_stop)`. The initial empty commit uses batch ID `0` and range
`[0, 0)`.

Write transaction-specific temporary files in the output directory. Compute
new output hashes before writing the manifest temp file. Atomically write the
journal before the first output replacement. Keep old existence flags, row
counts, hashes, and manifest payload in the journal. On recovery, truncate to
old counts, verify old hashes, restore old manifest or absence, then remove
the journal.

Build every manifest with these exact keys:

```python
{
    "schema_version": config.schema_version,
    "stage": "teacher_accepted_pool",
    "model_path": str(config.model_path.resolve()),
    "source_path": str(config.input_path.resolve()),
    "topology": config.topology,
    "devices": list(config.devices),
    "batch_size": config.batch_size,
    "max_worker_batch_size": (config.batch_size + 1) // 2,
    "worker_timeout_seconds": config.worker_timeout_seconds,
    "gpu_memory_utilization": config.gpu_memory_utilization,
    "max_model_len": config.max_model_len,
    "max_new_tokens": config.max_new_tokens,
    "temperature": config.temperature,
    "top_p": config.top_p,
    "seed": config.seed,
    "enable_thinking": config.enable_thinking,
    "cache_roots": [
        str(config.cache_root / "gpu0"),
        str(config.cache_root / "gpu1"),
    ],
    "processed_count": processed_count,
    "accepted_count": len(accepted),
    "rejected_count": len(rejected),
    "last_committed_position": last_committed_position,
    "target_accepted_count": config.stop_after_accepted,
    "completed": completed,
    "generation_contract": contract,
    "generation_contract_fingerprint": fingerprint_contract(contract),
    "source_sha256": source_sha256,
    "accepted_sha256": accepted_sha256,
    "rejected_sha256": rejected_sha256,
    "created_at": created_at,
    "updated_at": updated_at,
}
```

Materialize empty accepted and rejected JSONL files in the initial transaction
so both output hashes are always present. Preserve `created_at` on resume and
refresh `updated_at` only after a successful commit.

- [ ] **Step 8: Run state tests**

```bash
python -m pytest -q post_train_v2/tests/generation/test_teacher_state.py
```

Expected: PASS.

- [ ] **Step 9: Commit state handling**

```bash
git add \
  post_train_v2/src/generation/teacher_state.py \
  post_train_v2/tests/generation/test_teacher_state.py
git commit -m "feat: add transactional teacher state"
```

## Task 5: V2 Coordinator CLI

**Files:**

- Create: `post_train_v2/scripts/generation/build_teacher_pool.py`
- Create: `post_train_v2/configs/generation/teacher_rollout_2gpu.yaml`
- Create: `post_train_v2/configs/generation/teacher_rollout_2gpu_smoke.yaml`
- Create: `post_train_v2/tests/generation/test_build_teacher_pool.py`

- [ ] **Step 1: Write failing config and visibility tests**

Assert the parser resolves paths from the repository root and rejects:

- missing model/input paths;
- duplicate devices;
- non-`dual_tp1` topology;
- invalid numeric ranges;
- thinking enabled;
- inherited `CUDA_VISIBLE_DEVICES` that does not contain both configured
  device strings.

Assert unset visibility and `CUDA_VISIBLE_DEVICES=0,1` both pass for devices
`[0, 1]`.

Add source-schema cases proving that every row must contain a non-empty unique
`id`, a non-empty `prompt`, a `numbers` list, and an integer-compatible
`target`. These failures must occur before engine construction.

- [ ] **Step 2: Write failing ordered-acceptance tests**

Use fake source rows and responses to assert:

- responses are validated in source order;
- full responses are retained;
- invalid and empty responses become rejected rows;
- reaching the accepted target midway through a batch discards all later
  responses;
- processed rows remain an exact source prefix.

- [ ] **Step 3: Write failing coordinator failure tests**

Inject fake engine and state store objects. Assert:

- engine failure causes no commit for the current batch;
- a successful batch commits once;
- already-complete state does not start the engine;
- source exhaustion below target writes `completed: false` and returns nonzero;
- `KeyboardInterrupt` closes the engine and releases the lock;
- batch IDs increase monotonically;
- global batch size 64 creates two shards of at most 32.

- [ ] **Step 4: Run coordinator tests and verify RED**

```bash
python -m pytest -q post_train_v2/tests/generation/test_build_teacher_pool.py
```

Expected: import failure because the V2 script does not exist.

- [ ] **Step 5: Add production and smoke configs**

Production:

```yaml
model_path: post_train/model/qwen/qwen3-8b
input_path: post_train/data/processed/train_pool.jsonl
output_dir: post_train/data/teacher_rollouts
devices: [0, 1]
topology: dual_tp1
batch_size: 64
worker_timeout_seconds: 600
gpu_memory_utilization: 0.8
max_model_len: 512
max_new_tokens: 256
temperature: 0.2
top_p: 0.95
seed: 0
enable_thinking: false
stop_after_accepted: 20000
cache_root: /tmp/countdown_teacher_vllm
schema_version: 1
```

Smoke config uses the same fields but:

```yaml
output_dir: /tmp/post_train_v2_teacher_smoke
stop_after_accepted: 8
cache_root: /tmp/countdown_teacher_vllm_smoke
```

- [ ] **Step 6: Implement CLI and coordinator**

Expose:

```text
--config
--recover-stale-lock
--adopt-legacy-state
```

Execution order:

1. Resolve and validate config.
2. Create output directory.
3. Acquire `OutputLock`.
4. Recover transaction.
5. Read source and output state.
6. Validate source schema and validate/adopt resume state.
7. Create the initial empty accepted/rejected files and manifest with batch ID
   `0` and submitted range `[0, 0)` when no state exists.
8. Skip worker startup if the target is already met.
9. Start `ParallelVLLMEngine`.
10. Process source slices beginning at `processed_count`.
11. Split, generate, merge, validate, truncate at target, and commit.
12. Mark completed only when target is reached.
13. Exit nonzero if source is exhausted below target.
14. Always close workers and release the lock.

Build payload rows with the same fields as the legacy builder:

```python
{
    **row,
    "response": response.strip(),
    "teacher_expr": extract_answer_text(response.strip()),
    "validation": {
        "ok": result.ok,
        "error": result.error,
        "value": result.value,
    },
}
```

- [ ] **Step 7: Run coordinator tests**

```bash
python -m pytest -q \
  post_train_v2/tests/generation/test_build_teacher_pool.py \
  post_train_v2/tests/generation/test_teacher_state.py \
  post_train_v2/tests/generation/test_parallel_vllm.py
```

Expected: PASS.

- [ ] **Step 8: Commit the V2 entrypoint**

```bash
git add \
  post_train_v2/configs/generation \
  post_train_v2/scripts/generation/build_teacher_pool.py \
  post_train_v2/tests/generation/test_build_teacher_pool.py
git commit -m "feat: add dual gpu teacher pool builder"
```

## Task 6: Protect the Legacy Single-GPU Entry

**Files:**

- Modify: `post_train/scripts/data/build_teacher_pool.py`
- Modify: `post_train/tests/test_build_teacher_pool.py`

- [ ] **Step 1: Write failing legacy ownership tests**

Test:

- `--recover-stale-lock` appears in CLI help;
- the lock is acquired before accepted/rejected files are read;
- a V2 manifest containing `generation_contract_fingerprint` causes a hard
  failure before generator construction;
- the legacy writer releases only its own lock;
- an empty or legacy-owned output directory still follows existing behavior.

- [ ] **Step 2: Run legacy tests and verify RED**

```bash
python -m pytest -q post_train/tests/test_build_teacher_pool.py
```

Expected: failures because the legacy entrypoint does not use the shared lock
or detect V2 state.

- [ ] **Step 3: Add lock and V2-state checks**

Add the CLI flag and acquire:

```python
lock = OutputLock(
    output_dir / ".teacher_pool.lock",
    config_path=cfg_path,
    output_dir=output_dir,
    topology="legacy_single_tp1",
)
lock.acquire(recover_stale=args.recover_stale_lock)
```

Place all output reads, generator construction, and writes inside `try/finally`
that releases the lock. Before reading resume rows, inspect `manifest.json`;
if it contains `generation_contract_fingerprint`, raise an error explaining
that V2 state must be archived or removed before legacy generation starts.

- [ ] **Step 4: Run legacy and V2 tests**

```bash
python -m pytest -q \
  post_train/tests/test_build_teacher_pool.py \
  post_train/tests/test_output_lock.py \
  post_train_v2/tests/generation
```

Expected: PASS.

- [ ] **Step 5: Commit legacy protection**

```bash
git add \
  post_train/scripts/data/build_teacher_pool.py \
  post_train/tests/test_build_teacher_pool.py
git commit -m "fix: protect teacher pool writer ownership"
```

## Task 7: Documentation and Remote Smoke Procedure

**Files:**

- Create: `post_train_v2/scripts/generation/README.md`
- Modify: `post_train_v2/docs/environment_setup.md`
- Modify: `post_train_v2/docs/next_full_workflow.md`

- [ ] **Step 1: Document the production command**

Add:

```bash
python post_train_v2/scripts/generation/build_teacher_pool.py \
  --config post_train_v2/configs/generation/teacher_rollout_2gpu.yaml
```

Document that the command runs from repository root, writes the canonical
warehouse, and requires the Level 1 dual-Teacher smoke to pass.

- [ ] **Step 2: Document the isolated smoke command**

Add:

```bash
rm -rf /tmp/post_train_v2_teacher_smoke

python post_train_v2/scripts/generation/build_teacher_pool.py \
  --config post_train_v2/configs/generation/teacher_rollout_2gpu_smoke.yaml
```

Verification:

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("/tmp/post_train_v2_teacher_smoke")
accepted = [
    json.loads(line)
    for line in (root / "teacher_accepted_20k.jsonl").read_text().splitlines()
    if line.strip()
]
manifest = json.loads((root / "manifest.json").read_text())
assert len(accepted) == 8
assert manifest["completed"] is True
assert manifest["topology"] == "dual_tp1"
assert manifest["accepted_count"] == 8
print("OK: V2 dual Teacher smoke output validated")
PY
```

- [ ] **Step 3: Document stale lock and legacy adoption**

Show:

```bash
python post_train_v2/scripts/generation/build_teacher_pool.py \
  --config post_train_v2/configs/generation/teacher_rollout_2gpu.yaml \
  --recover-stale-lock
```

and:

```bash
python post_train_v2/scripts/generation/build_teacher_pool.py \
  --config post_train_v2/configs/generation/teacher_rollout_2gpu.yaml \
  --adopt-legacy-state
```

State that neither flag should be used routinely and that V2-to-legacy mixing
is prohibited.

- [ ] **Step 4: Run documentation consistency checks**

```bash
rg -n "build_teacher_pool.py|teacher_rollout_2gpu|dual_tp1" \
  post_train_v2/README.md \
  post_train_v2/docs \
  post_train_v2/scripts/generation/README.md
```

Confirm the production workflow points to the V2 entrypoint while the legacy
command is labeled single-GPU fallback.

- [ ] **Step 5: Commit documentation**

```bash
git add \
  post_train_v2/scripts/generation/README.md \
  post_train_v2/docs/environment_setup.md \
  post_train_v2/docs/next_full_workflow.md
git commit -m "docs: add dual teacher generation runbook"
```

## Task 8: Full Verification and Remote Acceptance

**Files:**

- Verify all files changed in Tasks 1-7.

- [ ] **Step 1: Run focused tests**

```bash
python -m pytest -q \
  post_train/tests/test_output_lock.py \
  post_train/tests/test_generation.py \
  post_train/tests/test_build_teacher_pool.py \
  post_train_v2/tests/generation
```

Expected: all pass.

- [ ] **Step 2: Run the complete local suite**

```bash
python -m pytest -q post_train/tests post_train_v2/tests
```

Expected: all tests pass, with only documented platform-dependent skips.

- [ ] **Step 3: Run static repository checks**

```bash
git diff --check
git status --short
git ls-files | rg '(^|/)(__pycache__/|\.pytest_tmp/|\.pytest_cache/)|\.pyc$'
```

Expected:

- no whitespace errors;
- no generated cache files tracked;
- only intended source, test, config, and documentation changes.

- [ ] **Step 4: Run the remote dual-worker smoke**

After Level 1 and HAMI repair:

```bash
cd countdown_post_train
source post_train_v2/.venv/bin/activate

python post_train_v2/scripts/generation/build_teacher_pool.py \
  --config post_train_v2/configs/generation/teacher_rollout_2gpu_smoke.yaml
```

Expected:

- both workers report ready;
- worker 0 uses GPU 0 and its cache root;
- worker 1 uses GPU 1 and its cache root;
- eight accepted examples are committed;
- no lock or transaction journal remains;
- no worker process remains.

- [ ] **Step 5: Run interruption and resume smoke**

Delete the smoke output, start the command, interrupt only after at least one
commit log, then rerun with `--recover-stale-lock` if the lock remains:

```bash
python post_train_v2/scripts/generation/build_teacher_pool.py \
  --config post_train_v2/configs/generation/teacher_rollout_2gpu_smoke.yaml \
  --recover-stale-lock
```

Expected:

- committed IDs are not duplicated;
- the combined accepted/rejected IDs remain a source prefix;
- the run finishes with eight accepted rows;
- output hashes match the manifest.

- [ ] **Step 6: Verify downstream compatibility**

Back up any existing canonical Teacher output before production. After a
small canonical test or the final 20k run:

```bash
python post_train/scripts/data/build_sft_splits.py \
  --config post_train/configs/data_build.yaml
```

Expected: the existing split builder reads the V2 accepted file without
schema changes.

- [ ] **Step 7: Start the production accepted-pool run**

Only after all previous gates:

```bash
python post_train_v2/scripts/generation/build_teacher_pool.py \
  --config post_train_v2/configs/generation/teacher_rollout_2gpu.yaml
```

Monitor accepted/rejected counts, per-worker latency, GPU memory, HAMI logs,
lock ownership, and transaction-journal absence after each successful commit.

Do not weaken lock, ordering, hash, transaction, or HAMI acceptance checks to
make the smoke pass. Any code change discovered during remote acceptance must
be reviewed and committed with an explicit file list rather than a broad
`git add` command.
