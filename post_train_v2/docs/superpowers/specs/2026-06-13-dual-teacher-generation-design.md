# Dual-GPU Teacher Generation Design

## Goal

Add a V2 production entrypoint that runs two persistent Qwen3-8B vLLM
tensor-parallel-size-1 workers, one per GPU, while preserving the existing
teacher acceptance, validation, ordering, resume, and output contracts.

The existing single-GPU
`post_train/scripts/data/build_teacher_pool.py` remains available as a
fallback. Both entrypoints use the same canonical output warehouse, but a
file lock prevents concurrent writers.

## Scope

This design covers only offline Teacher accepted-pool generation.

It does not implement:

- two-GPU DDP training;
- RFT or DPO generation;
- verl GRPO;
- a persistent vLLM HTTP service;
- tensor-parallel-size-2 Teacher generation.

The reusable worker orchestration may later be adopted by RFT and DPO after
the Teacher path is verified.

## Output Contract

The V2 entrypoint writes the existing canonical files:

```text
post_train/data/teacher_rollouts/
  teacher_accepted_20k.jsonl
  teacher_rejected.jsonl
  manifest.json
```

Existing downstream split builders consume these files without copying or
schema conversion.

Only one single-GPU or dual-GPU Teacher builder may write this directory at a
time.

## Process Architecture

The implementation uses one coordinator process and two persistent worker
processes created with the multiprocessing `spawn` context.

### Coordinator

The coordinator:

- acquires the exclusive output lock;
- recovers any transaction journal while holding the lock;
- loads and validates the normalized source rows;
- loads existing accepted and rejected rows for resume;
- validates duplicate IDs, source-prefix ordering, and accepted-count bounds;
- starts and monitors two workers;
- forms global batches of at most the configured global batch size;
- splits each batch into two contiguous shards;
- sends one shard to each worker;
- waits for both results;
- restores original source order;
- validates every response with the existing Countdown validator;
- commits accepted, rejected, and manifest files through the transaction
  journal;
- stops at exactly 20,000 accepted rows;
- shuts workers down on success or failure.

The coordinator is the only process allowed to run solver validation or write
output files.

### Workers

Each worker:

- receives one configured physical-device index;
- sets `CUDA_VISIBLE_DEVICES` before importing torch or vLLM;
- sets its own `VLLM_CACHE_ROOT`;
- initializes one persistent `VLLMGenerator` with
  `tensor_parallel_size=1`, the configured GPU memory utilization, and the
  configured maximum model length and seed;
- processes assigned prompts in order;
- returns responses paired with original source positions;
- never reads or writes the output warehouse;
- never performs solver validation.

Worker 0 uses GPU 0 and:

```text
/tmp/countdown_teacher_vllm/gpu0
```

Worker 1 uses GPU 1 and:

```text
/tmp/countdown_teacher_vllm/gpu1
```

The worker cache roots are created before model initialization. They must not
share vLLM Torch compilation artifacts.

## Ordered Batch Semantics

`batch_size: 64` remains the global batch size.

For each global batch:

- the first contiguous half is assigned to worker 0;
- the second contiguous half is assigned to worker 1;
- odd-sized batches give the extra item to worker 0;
- neither worker receives more than 32 rows for a full batch.

Both worker results must be available before the coordinator commits the
batch. The coordinator merges results by original source position, then
validates and commits them in source order.

Worker completion order must never affect dataset order.

Each source row is rolled out once unless its containing batch fails before
commit. An uncommitted failed batch is rerun in full on resume.

Every worker request and response includes a monotonically increasing batch
ID. The coordinator rejects stale, duplicate, or mismatched worker messages.
An empty tail shard returns an empty successful result without invoking
generation.

The configured device values are host-visible CUDA indices. The production
launcher must expose the same indices or leave `CUDA_VISIBLE_DEVICES` unset;
the coordinator rejects an inherited visibility mask that does not contain
both configured device strings.

## Acceptance and Truncation

The V2 entrypoint reuses the current behavior:

1. Strip the complete Teacher response.
2. Extract the `<answer>...</answer>` expression.
3. Validate syntax, exact number use, and exact target equality with the
   existing Fraction-based validator.
4. Append the row to accepted or rejected output.

When the 20,000th accepted row is reached, processing stops at that source
position. Later responses from the same completed worker batch are not
committed. This preserves the existing source-order stop semantics.

The complete response remains the supervised target. Validation uses only the
answer expression.

## Locking

The output directory contains:

```text
.teacher_pool.lock
```

The lock is created exclusively and records:

- schema version;
- PID;
- hostname;
- start timestamp;
- resolved config path;
- output directory;
- topology.

If the lock exists:

- a live PID on the same host causes an immediate hard failure;
- an unknown host or unverifiable PID causes an immediate hard failure;
- a stale local PID is not removed automatically;
- `--recover-stale-lock` is required to replace a stale lock.

The coordinator removes only the lock it owns during orderly shutdown.
Abnormal process termination may leave the lock for explicit recovery.

The legacy single-GPU entrypoint must also honor this lock before both
entrypoints can be considered safe concurrent alternatives.

The legacy entrypoint must refuse to mutate an output directory whose
manifest contains a V2 immutable generation-contract fingerprint. Dual-TP1
and legacy single-engine rows are not mixed in one pool. Switching from a
partially committed V2 run to the legacy topology requires archiving or
removing the existing output state and starting that pool again. V2 may adopt
legacy state only through the explicit `--adopt-legacy-state` validation
path.

## Transactional Persistence

Accepted, rejected, and manifest outputs use same-directory temporary files
and atomic replacement.

Because three filenames cannot be replaced as one filesystem transaction, the
coordinator also uses:

```text
.teacher_pool.transaction.json
```

Before replacing any output for a batch, the coordinator atomically writes a
transaction journal containing:

- transaction schema version;
- batch ID and submitted source-position range;
- pre-commit accepted and rejected file-existence flags;
- pre-commit accepted and rejected row counts;
- pre-commit accepted and rejected SHA-256 values when the files existed;
- the complete pre-commit manifest payload, or an explicit marker that no
  manifest existed.

The coordinator then replaces accepted, rejected, and manifest files, in that
order, and removes the transaction journal only after all three replacements
succeed.

If a journal exists at startup, the coordinator performs recovery before
starting workers:

1. Read and validate the journal.
2. Truncate the current accepted and rejected snapshots to the recorded
   pre-commit row counts.
3. Restore absence for files that did not exist before the transaction.
4. Verify repaired accepted and rejected hashes against the journal.
5. Restore the recorded pre-commit manifest state.
6. Atomically replace the repaired files.
7. Remove the journal.

This intentionally rolls back a batch even if all output replacements
completed but the process died before journal removal. The batch is then
generated again, preserving at-most-one committed result per source row.

A global batch is committed only after:

- both workers returned successfully;
- response counts match shard sizes;
- every result position is known and unique;
- the merged positions match the submitted batch exactly.

If validation or persistence fails, the process exits nonzero. Previously
committed batches remain valid. A transaction journal left by the current
batch makes the partial state detectable and rollback-safe.

The implementation must never treat output as resumable while an unrecovered
transaction journal exists.

## Failure Handling

Hard failures include:

- worker initialization failure;
- worker timeout;
- worker process exit;
- malformed worker message;
- missing, duplicate, or unknown source positions;
- result-count mismatch;
- output duplicate IDs;
- model or generation configuration mismatch on resume;
- file-lock acquisition failure;
- invalid or unrecoverable transaction journal;
- atomic write failure.

On any worker or batch failure:

1. Do not commit the current batch.
2. Signal both workers to stop.
3. Terminate workers that do not exit within the grace period.
4. Preserve previously committed output.
5. Return a nonzero process status.

No automatic retry occurs inside the same run. The operator resolves the
cause and resumes from committed IDs.

A mathematically invalid, malformed, truncated, or empty model response is a
normal rejected example, not a worker failure. Only transport, process,
protocol, or persistence failures abort a batch.

If the source is exhausted before the accepted target is reached, the
coordinator commits the final processed prefix, writes `completed: false`,
shuts workers down, and exits nonzero with the accepted and target counts.
If resume starts with the accepted target already satisfied, the coordinator
validates and finalizes state without starting workers.

## Configuration

The canonical V2 config is:

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

Configuration validation requires:

- exactly two distinct non-negative device indices;
- topology equal to `dual_tp1`;
- positive batch size and timeout;
- GPU memory utilization greater than zero and at most one;
- positive model and response lengths;
- non-negative temperature;
- top-p greater than zero and at most one;
- non-negative integer seed;
- thinking disabled for the production config;
- positive accepted target;
- existing model and input paths.

Source validation requires every JSONL row to contain a non-empty unique
`id`, a non-empty `prompt`, a `numbers` list, and an integer-compatible
`target`. Missing or malformed source fields are hard failures before workers
start.

The entrypoint accepts:

```text
--recover-stale-lock
--adopt-legacy-state
```

`--recover-stale-lock` has the locking semantics defined above.

`--adopt-legacy-state` is required when accepted or rejected rows already
exist but the manifest lacks the V2 immutable generation contract. Adoption
is permitted only after validating that:

- every existing output ID occurs exactly once in the configured source;
- accepted and rejected rows combined form an exact contiguous prefix of the
  source order;
- every accepted row still passes the current validator;
- every rejected row still fails the current validator;
- existing rows do not exceed the configured accepted target.

Without this explicit flag, legacy state is rejected rather than silently
mixed with V2 generation.

## Manifest

The manifest records these exact keys:

- `schema_version`;
- `stage`;
- `model_path`;
- `source_path`;
- `topology`;
- `devices`;
- `batch_size`;
- `max_worker_batch_size`;
- `worker_timeout_seconds`;
- `gpu_memory_utilization`;
- `max_model_len`;
- `max_new_tokens`;
- `temperature`;
- `top_p`;
- `seed`;
- `enable_thinking`;
- `cache_roots`;
- `processed_count`;
- `accepted_count`;
- `rejected_count`;
- `last_committed_position`;
- `target_accepted_count`;
- `completed`;
- `generation_contract`;
- `generation_contract_fingerprint`;
- `source_sha256`;
- `accepted_sha256`;
- `rejected_sha256`;
- `created_at`;
- `updated_at`.

`stage` is always `teacher_accepted_pool`. Counts are integers.
`last_committed_position` is `null` for an empty committed state. Empty
accepted and rejected files are materialized so their SHA-256 values are
always defined. `created_at` is retained across resume and `updated_at`
changes after each successful commit.

Resume rejects immutable configuration changes that can mix incompatible
generation semantics. Operational timeout and cache-root paths may change;
model, topology, prompt/generation parameters, schema version, and source
identity may not.

The immutable generation-contract fingerprint is SHA-256 over canonical
sorted-key JSON containing exactly:

- schema version;
- source SHA-256;
- resolved model path;
- topology;
- global batch size;
- maximum model length;
- maximum new tokens;
- temperature;
- top-p;
- seed;
- thinking flag.

Device indices, worker timeout, GPU memory utilization, cache root, config
path, and output path are operational fields and are not part of the
fingerprint.

On every resume, accepted and rejected rows combined must form an exact
contiguous prefix of source IDs. Because accepted and rejected rows live in
separate files, the coordinator maps every output ID to its unique source
position, combines both sets, sorts by source position, and compares the
result with `source_ids[:processed_count]`. The last committed source position
is derived from that prefix and cross-checked against the manifest. Arbitrary
processed-ID subsets are rejected.

SHA-256 is mandatory:

- `source_sha256` is the hash of the source JSONL file bytes;
- `accepted_sha256` is the hash of the committed accepted JSONL file bytes;
- `rejected_sha256` is the hash of the committed rejected JSONL file bytes.

The manifest records all three hashes after every committed batch. Resume
verifies the source hash and, when no transaction journal is present, both
output hashes before starting workers.

## Logging

The coordinator logs:

- lock acquisition and recovery;
- worker startup and readiness;
- device and cache assignment;
- source/resume counts;
- global batch range;
- per-worker shard sizes and latency;
- accepted and rejected totals after each commit;
- worker shutdown and exit status.

Workers report initialization success only after vLLM is ready. Worker logs
must identify the logical worker index without writing generated responses to
the canonical output files.

## File Responsibilities

Create:

```text
post_train_v2/configs/generation/teacher_rollout_2gpu.yaml
post_train_v2/configs/generation/teacher_rollout_2gpu_smoke.yaml
post_train_v2/scripts/generation/build_teacher_pool.py
post_train_v2/src/generation/parallel_vllm.py
post_train_v2/src/generation/teacher_state.py
post_train_v2/src/generation/README.md
post_train/src/countdown/output_lock.py
post_train/tests/test_output_lock.py
post_train_v2/tests/generation/test_parallel_vllm.py
post_train_v2/tests/generation/test_build_teacher_pool.py
post_train_v2/tests/generation/test_teacher_state.py
post_train_v2/tests/generation/README.md
```

Responsibilities:

- `teacher_rollout_2gpu_smoke.yaml`: same topology and generation semantics
  as production, with a temporary output directory and accepted target of 8.
- `parallel_vllm.py`: worker protocol, environment isolation, process
  lifecycle, deterministic sharding, result validation, and ordered merge.
- `teacher_state.py`: transaction journal, legacy-state integration,
  legacy-state adoption, contiguous-prefix validation, rollback recovery, and
  manifest contract fingerprinting.
- generation source and test READMEs: module boundaries, test commands, and
  the rule that GPU-dependent acceptance remains a remote gate.
- `post_train/src/countdown/output_lock.py`: shared exclusive lock ownership,
  live-process detection, stale-lock recovery, and owner-token-safe release
  used by both the legacy and V2 entrypoints.
- `build_teacher_pool.py`: config loading, lock ownership, resume validation,
  Countdown validation, atomic persistence, and manifest updates.
- Existing `post_train/src/countdown/generation.py`: shared vLLM chat and
  thinking-disabled generation behavior, extended conservatively to accept
  optional GPU memory utilization and maximum model length arguments without
  changing existing call defaults, plus an optional seed forwarded to vLLM.
- Existing `post_train/src/countdown/validation.py`: canonical mathematical
  correctness.

The legacy builder requires a focused follow-up change to honor the common
lock, expose `--recover-stale-lock`, and reject V2-owned output state. Its
generation behavior otherwise remains unchanged.

## Verification

### Unit tests

Tests cover:

- contiguous deterministic split for even, odd, and tail batches;
- ordered merge independent of worker completion order;
- environment assignment before heavy imports;
- distinct worker cache roots;
- stale or mismatched batch IDs;
- worker readiness, timeout, exit, malformed result, and shutdown;
- exact response-count and source-position validation;
- live lock rejection;
- explicit stale-lock recovery;
- transaction-journal rollback after each possible partial replacement;
- whole-batch non-commit on worker failure;
- exact stop at the accepted target;
- resume without duplicate IDs;
- resume rejection when output IDs do not form a source prefix;
- explicit legacy-state adoption and default legacy-state rejection;
- legacy single-GPU refusal to overwrite V2-owned state;
- immutable configuration mismatch rejection;
- atomic accepted/rejected/manifest persistence.

### Remote smoke

Use a separate temporary output directory and a small accepted target.

Success requires:

- both workers initialize;
- logs show GPU 0 and GPU 1 isolation;
- logs show distinct vLLM cache roots;
- both workers generate non-empty responses;
- committed output follows source order;
- accepted and rejected rows contain no duplicate IDs;
- interruption and resume do not duplicate committed rows;
- no orphan worker process remains.

### Production gate

The production 20,000 accepted-pool run begins only after:

- Level 1 dual-Teacher smoke succeeds;
- the V2 small-target smoke succeeds;
- resume smoke succeeds;
- the legacy single-GPU entrypoint honors the shared lock;
- generated output is accepted by the existing downstream split builder.
