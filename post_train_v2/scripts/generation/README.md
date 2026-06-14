# V2 Teacher Pool Generation

Run every command in this document from the repository root. The V2 builder
starts two independent tensor-parallel-size-1 vLLM engines, one on each GPU,
and commits accepted rows, rejected rows, and the manifest transactionally.
It does not provide distributed SFT, DPO, or verl training.

## Prerequisites

Before either the smoke or production run:

- activate `post_train_v2/.venv`;
- pass every Level 1 gate in
  `post_train_v2/docs/environment_setup.md`, including the dual-engine Teacher
  gate;
- provide `post_train/model/qwen/qwen3-8b`;
- build `post_train/data/processed/train_pool.jsonl`;
- expose physical GPUs 0 and 1 with `CUDA_VISIBLE_DEVICES` unset/blank or
  exactly `0,1`;
- ensure the configured output and cache roots are writable and have enough
  space.

The production config writes the canonical warehouse to
`post_train/data/teacher_rollouts` and uses
`/tmp/countdown_teacher_vllm/gpu0` and
`/tmp/countdown_teacher_vllm/gpu1` as isolated worker caches. The smoke config
writes only to `/tmp/post_train_v2_teacher_smoke` and uses
`/tmp/countdown_teacher_vllm_smoke/gpu0` and
`/tmp/countdown_teacher_vllm_smoke/gpu1`.

Activate and verify the V2 environment:

```bash
source post_train_v2/.venv/bin/activate
test "$VIRTUAL_ENV" = "$PWD/post_train_v2/.venv"
python -V
test -f post_train/model/qwen/qwen3-8b/config.json
test -f post_train/data/processed/train_pool.jsonl
unset CUDA_VISIBLE_DEVICES
```

## Local CPU Tests

These tests use fake workers and do not establish GPU readiness:

```bash
python -m pytest -q \
  post_train_v2/tests/generation \
  post_train/tests/test_build_teacher_pool.py \
  post_train/tests/test_output_lock.py
```

Real vLLM startup, dual-GPU isolation, and output acceptance require the
remote GPU smoke procedure below.

## Remote GPU Smoke

The only destructive smoke cleanup is the exact isolated output path below.
Do not use a wildcard and do not remove the production warehouse.

```bash
rm -rf -- /tmp/post_train_v2_teacher_smoke

set -o pipefail
python post_train_v2/scripts/generation/build_teacher_pool.py \
  --config post_train_v2/configs/generation/teacher_rollout_2gpu_smoke.yaml \
  2>&1 | tee /tmp/post_train_v2_teacher_smoke.log
```

Validate the committed smoke state:

```bash
python - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path("/tmp/post_train_v2_teacher_smoke")
accepted_path = root / "teacher_accepted_20k.jsonl"
rejected_path = root / "teacher_rejected.jsonl"
manifest_path = root / "manifest.json"
lock_path = root / ".teacher_pool.lock"
journal_path = root / ".teacher_pool.transaction.json"

def read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

accepted = read_jsonl(accepted_path)
rejected = read_jsonl(rejected_path)
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
source_path = Path(manifest["source_path"])
source = read_jsonl(source_path)

assert len(accepted) == 8
assert manifest["completed"] is True
assert manifest["topology"] == "dual_tp1"
assert manifest["devices"] == [0, 1]
assert manifest["target_accepted_count"] == 8
assert manifest["accepted_count"] == len(accepted) == 8
assert manifest["rejected_count"] == len(rejected)
assert manifest["processed_count"] == len(accepted) + len(rejected)
assert manifest["last_committed_position"] == manifest["processed_count"] - 1

assert manifest["source_sha256"] == sha256(source_path)
assert manifest["accepted_sha256"] == sha256(accepted_path)
assert manifest["rejected_sha256"] == sha256(rejected_path)
contract = manifest["generation_contract"]
assert contract["source_sha256"] == manifest["source_sha256"]
fingerprint = hashlib.sha256(
    json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()
assert manifest["generation_contract_fingerprint"] == fingerprint

source_ids = [row["id"] for row in source]
output_ids = [row["id"] for row in accepted + rejected]
assert len(source_ids) == len(set(source_ids))
assert len(output_ids) == len(set(output_ids))
source_positions = {row_id: position for position, row_id in enumerate(source_ids)}
assert all(row_id in source_positions for row_id in output_ids)
assert sorted(source_positions[row_id] for row_id in output_ids) == list(
    range(manifest["processed_count"])
)
assert all(row["validation"]["ok"] is True for row in accepted)
assert all(row["validation"]["ok"] is False for row in rejected)
assert not lock_path.exists()
assert not journal_path.exists()

print("OK: V2 dual-Teacher smoke output validated")
PY
```

The log must show both worker cache/device assignments, `engine ready`,
per-worker shard sizes and latencies, committed batches, worker shutdown, and
worker exit codes. The validator is the acceptance gate; a successful process
exit alone is insufficient.

## Production

After the remote smoke validator passes:

```bash
set -o pipefail
python post_train_v2/scripts/generation/build_teacher_pool.py \
  --config post_train_v2/configs/generation/teacher_rollout_2gpu.yaml \
  2>&1 | tee -a /tmp/post_train_v2_teacher_production.log
```

This writes:

- `post_train/data/teacher_rollouts/teacher_accepted_20k.jsonl`;
- `post_train/data/teacher_rollouts/teacher_rejected.jsonl`;
- `post_train/data/teacher_rollouts/manifest.json`.

The production run is complete only when the manifest reports
`accepted_count: 20000`, `target_accepted_count: 20000`,
`completed: true`, and coherent processed/rejected counts and hashes.

## Interruption and Resume

`Ctrl-C` stops workers, releases the owned lock, and returns exit code 130.
Already committed batches remain valid. Resume by rerunning the exact same
config command. The builder validates the source prefix, IDs, hashes, counts,
and immutable generation contract before starting workers.

If an interruption leaves `.teacher_pool.transaction.json`, rerun the same V2
command. Recovery is automatic while the output lock is held: the builder
restores the last committed snapshots, verifies them, removes the journal,
and then resumes. Never blindly delete or edit the transaction journal; an
unrecoverable journal is a hard failure that must be investigated.

An abnormal process death can leave `.teacher_pool.lock`. Inspect it and its
recorded PID before taking action:

```bash
OUTPUT=post_train/data/teacher_rollouts
cat "$OUTPUT/.teacher_pool.lock"
LOCK_PID="$(
  python -c 'import json,sys; print(json.load(open(sys.argv[1]))["pid"])' \
    "$OUTPUT/.teacher_pool.lock"
)"
ps -fp "$LOCK_PID"
```

Use stale-lock recovery only when the lock hostname is the current local host
and `ps` confirms that the recorded PID is dead:

```bash
python post_train_v2/scripts/generation/build_teacher_pool.py \
  --config post_train_v2/configs/generation/teacher_rollout_2gpu.yaml \
  --recover-stale-lock
```

Do not use `--recover-stale-lock` for a live PID, a foreign hostname, or an
unreadable/corrupt lock. Do not manually remove a lock unless the state has
been investigated and the normal recovery path cannot be used.

## Legacy Adoption Boundary

V2 may adopt a legacy-owned canonical pool once:

```bash
python post_train_v2/scripts/generation/build_teacher_pool.py \
  --config post_train_v2/configs/generation/teacher_rollout_2gpu.yaml \
  --adopt-legacy-state
```

Adoption is allowed only when existing accepted and rejected rows are valid,
contain no duplicate or unknown IDs, form an exact contiguous prefix of the
configured source, remain below the configured accepted target, and have no
V2 generation-contract fingerprint. The builder revalidates accepted and
rejected classifications before materializing V2 state.

Adoption is one-way. After a V2 manifest or V2 transaction journal exists,
the legacy builder must never write that directory. V2-to-legacy mixing is
prohibited. The legacy builder is a fallback only for a fresh empty output
directory or a directory that remains exclusively legacy-owned.
