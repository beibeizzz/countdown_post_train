# V2 Teacher Pool Generation

Run every command in this document from the repository root. The V2 builder
starts two independent tensor-parallel-size-1 vLLM engines, one on each GPU,
and commits accepted rows, rejected rows, and the manifest transactionally.
It does not provide distributed SFT, DPO, or verl training.

## Prerequisites

Before either the smoke or production run:

- activate `post_train_v2/.venv`;
- pass the earlier Level 1 gates in
  `post_train_v2/docs/environment_setup.md` through the dual-engine Teacher
  smoke; the coordinator smoke and resume checks in this document are the
  next Level 1 gates, not prerequisites to themselves;
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
```

Run the smoke separately from cleanup:

```bash
set -o pipefail
python post_train_v2/scripts/generation/build_teacher_pool.py \
  --config post_train_v2/configs/generation/teacher_rollout_2gpu_smoke.yaml \
  2>&1 | tee /tmp/post_train_v2_teacher_smoke.log
```

The validator takes its output root and expected target from environment
variables:

```bash
TEACHER_SMOKE_ROOT=/tmp/post_train_v2_teacher_smoke \
TEACHER_SMOKE_TARGET=8 \
python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

from post_train.src.countdown.validation import validate_countdown_response

root = Path(os.environ["TEACHER_SMOKE_ROOT"])
target = int(os.environ["TEACHER_SMOKE_TARGET"])
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

assert len(accepted) == target
assert manifest["completed"] is True
assert manifest["topology"] == "dual_tp1"
assert manifest["devices"] == [0, 1]
assert manifest["target_accepted_count"] == target
assert manifest["accepted_count"] == len(accepted) == target
assert manifest["rejected_count"] == len(rejected)
assert manifest["processed_count"] == len(accepted) + len(rejected)
assert manifest["last_committed_position"] == manifest["processed_count"] - 1

assert manifest["source_sha256"] == sha256(source_path)
assert manifest["accepted_sha256"] == sha256(accepted_path)
assert manifest["rejected_sha256"] == sha256(rejected_path)
contract = manifest["generation_contract"]
for field in (
    "schema_version",
    "source_sha256",
    "model_path",
    "topology",
    "batch_size",
    "max_model_len",
    "max_new_tokens",
    "temperature",
    "top_p",
    "seed",
    "enable_thinking",
):
    assert contract[field] == manifest[field], field
fingerprint = hashlib.sha256(
    json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()
assert manifest["generation_contract_fingerprint"] == fingerprint

source_ids = [row["id"] for row in source]
assert len(source_ids) == len(set(source_ids))
source_positions = {row_id: position for position, row_id in enumerate(source_ids)}
outputs = accepted + rejected
output_ids = [row["id"] for row in outputs]
assert len(output_ids) == len(set(output_ids))
assert all(row_id in source_positions for row_id in output_ids)
ordered = sorted(outputs, key=lambda row: source_positions[row["id"]])
assert [row["id"] for row in ordered] == source_ids[: manifest["processed_count"]]

accepted_ids = {row["id"] for row in accepted}
for output, source_row in zip(
    ordered, source[: manifest["processed_count"]], strict=True
):
    for field in ("prompt", "numbers", "target"):
        assert output[field] == source_row[field], (output["id"], field)
    result = validate_countdown_response(
        output["response"], output["numbers"], output["target"]
    )
    assert result.ok is (output["id"] in accepted_ids), output["id"]
    assert output["validation"] == {
        "ok": result.ok,
        "error": result.error,
        "value": result.value,
    }
assert not lock_path.exists()
assert not journal_path.exists()

print(f"OK: V2 dual-Teacher output validated: {root}")
PY
```

The output validator cannot determine which child generated a response.
Worker attribution and process cleanup are accepted from the coordinator log,
using the runtime-metadata check below. Empty responses remain valid rejected
rows in production. The smoke gate is stricter: every worker must produce at
least one nonempty response in every smoke batch so both generation paths are
proven.

## Deterministic Resume Smoke

Create a temporary resume config from the tracked smoke config. This does not
modify the repository config:

```bash
python - <<'PY'
from pathlib import Path

import yaml

source = Path(
    "post_train_v2/configs/generation/teacher_rollout_2gpu_smoke.yaml"
)
target = Path("/tmp/post_train_v2_teacher_resume_smoke.yaml")
config = yaml.safe_load(source.read_text(encoding="utf-8"))
config.update(
    output_dir="/tmp/post_train_v2_teacher_resume_smoke",
    cache_root="/tmp/countdown_teacher_vllm_resume_smoke",
    batch_size=2,
    stop_after_accepted=8,
)
target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
print(target)
PY
```

Clean the exact resume output path once, before the first run:

```bash
rm -rf -- /tmp/post_train_v2_teacher_resume_smoke
```

Launch the first run through this monitor. It streams the log and sends
`SIGINT` immediately after the first committed batch, which guarantees a
nonempty incomplete checkpoint because the batch size is two and the target
is eight:

```bash
unset CUDA_VISIBLE_DEVICES
python - <<'PY'
import signal
import subprocess
import sys
from pathlib import Path

command = [
    sys.executable,
    "post_train_v2/scripts/generation/build_teacher_pool.py",
    "--config",
    "/tmp/post_train_v2_teacher_resume_smoke.yaml",
]
log_path = Path("/tmp/post_train_v2_teacher_resume_interrupted.log")
interrupted = False
with log_path.open("w", encoding="utf-8") as log:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        log.write(line)
        log.flush()
        if not interrupted and "committed batch=" in line:
            process.send_signal(signal.SIGINT)
            interrupted = True
    returncode = process.wait()

assert interrupted, "run ended before a committed batch was observed"
assert returncode == 130, returncode
PY
```

Validate the interrupted checkpoint, shutdown record, child cleanup, and lock
owner before resuming. The child exit codes must be present and numeric, but
are not compared with a cross-platform whitelist:

```bash
python - <<'PY'
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

root = Path("/tmp/post_train_v2_teacher_resume_smoke")
log_path = Path("/tmp/post_train_v2_teacher_resume_interrupted.log")
accepted_path = root / "teacher_accepted_20k.jsonl"
rejected_path = root / "teacher_rejected.jsonl"
manifest_path = root / "manifest.json"
journal_path = root / ".teacher_pool.transaction.json"
lock_path = root / ".teacher_pool.lock"

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
source = read_jsonl(Path(manifest["source_path"]))
if journal_path.exists():
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    assert set(journal) == {
        "schema_version",
        "batch_id",
        "submitted_start",
        "submitted_stop",
        "accepted",
        "rejected",
        "manifest",
    }
    assert journal["schema_version"] == 1
    assert 0 < journal["submitted_start"] <= journal["submitted_stop"]
    assert journal["submitted_start"] == (
        journal["accepted"]["row_count"] + journal["rejected"]["row_count"]
    )
    assert set(journal["accepted"]) == {"existed", "row_count", "sha256"}
    assert set(journal["rejected"]) == {"existed", "row_count", "sha256"}
    assert set(journal["manifest"]) == {"existed", "payload"}
else:
    assert 0 < manifest["processed_count"] < len(source)
    assert manifest["processed_count"] == len(accepted) + len(rejected)
    assert manifest["accepted_count"] == len(accepted)
    assert manifest["rejected_count"] == len(rejected)
    assert manifest["last_committed_position"] == manifest["processed_count"] - 1
    assert manifest["completed"] is False
    assert manifest["accepted_sha256"] == sha256(accepted_path)
    assert manifest["rejected_sha256"] == sha256(rejected_path)
    source_positions = {row["id"]: position for position, row in enumerate(source)}
    outputs = accepted + rejected
    assert len(source_positions) == len(source)
    assert len({row["id"] for row in outputs}) == len(outputs)
    ordered = sorted(outputs, key=lambda row: source_positions[row["id"]])
    assert [row["id"] for row in ordered] == [
        row["id"] for row in source[: manifest["processed_count"]]
    ]

text = log_path.read_text(encoding="utf-8")
runtime = re.search(
    r"worker0 runtime pid=(?P<pid0>[1-9]\d*) visible_device=(?P<device0>\S+) "
    r"cache_root=(?P<cache0>\S+).*?"
    r"worker1 runtime pid=(?P<pid1>[1-9]\d*) visible_device=(?P<device1>\S+) "
    r"cache_root=(?P<cache1>\S+)",
    text,
    re.DOTALL,
)
assert runtime
pids = (int(runtime["pid0"]), int(runtime["pid1"]))
assert pids[0] != pids[1]
assert (runtime["device0"], runtime["device1"]) == ("0", "1")
cache_root = Path("/tmp/countdown_teacher_vllm_resume_smoke")
assert Path(runtime["cache0"]).resolve() == (cache_root / "gpu0").resolve()
assert Path(runtime["cache1"]).resolve() == (cache_root / "gpu1").resolve()
batch = re.search(
    r"worker0_shard=(?P<shard0>\d+) worker0_results=(?P<results0>\d+) "
    r"worker0_nonempty=(?P<nonempty0>\d+) "
    r"worker0_latency_seconds=(?P<latency0>\S+) "
    r"worker1_shard=(?P<shard1>\d+) worker1_results=(?P<results1>\d+) "
    r"worker1_nonempty=(?P<nonempty1>\d+) "
    r"worker1_latency_seconds=(?P<latency1>\S+)",
    text,
)
assert batch
for worker in ("0", "1"):
    shard = int(batch[f"shard{worker}"])
    results = int(batch[f"results{worker}"])
    nonempty = int(batch[f"nonempty{worker}"])
    latency = float(batch[f"latency{worker}"])
    assert results == shard > 0
    assert 0 < nonempty <= results
    assert math.isfinite(latency) and latency >= 0
shutdown = re.search(
    r"worker shutdown exitcodes=\((?P<exit0>-?\d+), (?P<exit1>-?\d+)\) "
    r"runtime_pids=\((?P<pid0>\d+), (?P<pid1>\d+)\)",
    text,
)
assert shutdown, "shutdown/final exitcodes were not logged"
assert (int(shutdown["pid0"]), int(shutdown["pid1"])) == pids

for pid in pids:
    probe = subprocess.run(
        ["ps", "-p", str(pid), "-o", "pid="],
        check=False,
        capture_output=True,
        text=True,
    )
    assert probe.returncode != 0 and not probe.stdout.strip(), pid

if lock_path.exists():
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    owner = subprocess.run(
        ["ps", "-p", str(lock["pid"]), "-o", "pid="],
        check=False,
        capture_output=True,
        text=True,
    )
    assert owner.returncode != 0 and not owner.stdout.strip(), (
        f"live lock owner remains: {lock['pid']}"
    )

print(f"OK: interrupted checkpoint and shutdown validated for PIDs {pids}")
PY
```

For this deliberately interrupted run, worker exit codes may be graceful
zeros or platform-specific termination codes. POSIX may report negative
signal exit codes; Windows may report a nonzero terminate code. Do not
whitelist exact numeric values. Acceptance requires coordinator exit code
130; shutdown/final exit codes logged; both child PIDs gone; no live lock
owner; and either a valid committed incomplete checkpoint without a journal
or a structurally valid recovery journal. Never delete a journal; the resumed
builder performs the authoritative recovery validation.

Resume with the same temporary config and without cleanup:

```bash
set -o pipefail
python post_train_v2/scripts/generation/build_teacher_pool.py \
  --config /tmp/post_train_v2_teacher_resume_smoke.yaml \
  2>&1 | tee /tmp/post_train_v2_teacher_resume_resumed.log
```

Require evidence that restart loaded committed progress:

```bash
grep -E 'resume processed=[1-9][0-9]* accepted=' \
  /tmp/post_train_v2_teacher_resume_resumed.log
```

Validate the completed resumed output by rerunning the validator block from
"Remote GPU Smoke" with `TEACHER_SMOKE_ROOT` set to
`/tmp/post_train_v2_teacher_resume_smoke` and `TEACHER_SMOKE_TARGET` set to
`8`.

Validate runtime metadata, batch attribution, final exit codes, and orphan
cleanup for both logs:

```bash
python - <<'PY'
import math
import re
import subprocess
from pathlib import Path

checks = (
    (
        Path("/tmp/post_train_v2_teacher_smoke.log"),
        Path("/tmp/countdown_teacher_vllm_smoke"),
    ),
    (
        Path("/tmp/post_train_v2_teacher_resume_resumed.log"),
        Path("/tmp/countdown_teacher_vllm_resume_smoke"),
    ),
)
run_re = re.compile(
    r"worker0 runtime pid=(?P<pid0>[1-9]\d*) visible_device=(?P<device0>\S+) "
    r"cache_root=(?P<cache0>\S+).*?"
    r"worker1 runtime pid=(?P<pid1>[1-9]\d*) visible_device=(?P<device1>\S+) "
    r"cache_root=(?P<cache1>\S+)(?P<body>.*?)"
    r"worker shutdown exitcodes=\((?P<exit0>-?\d+), (?P<exit1>-?\d+)\) "
    r"runtime_pids=\((?P<shutdown_pid0>\d+), (?P<shutdown_pid1>\d+)\)",
    re.DOTALL,
)
batch_re = re.compile(
    r"worker0_shard=(?P<shard0>\d+) worker0_results=(?P<results0>\d+) "
    r"worker0_nonempty=(?P<nonempty0>\d+) "
    r"worker0_latency_seconds=(?P<latency0>\S+) "
    r"worker1_shard=(?P<shard1>\d+) worker1_results=(?P<results1>\d+) "
    r"worker1_nonempty=(?P<nonempty1>\d+) "
    r"worker1_latency_seconds=(?P<latency1>\S+)"
)

recorded_pids = set()
for log_path, cache_root in checks:
    text = log_path.read_text(encoding="utf-8")
    runs = list(run_re.finditer(text))
    assert len(runs) == 1, (log_path, len(runs))
    data = runs[0].groupdict()
    pid0, pid1 = int(data["pid0"]), int(data["pid1"])
    assert pid0 != pid1
    assert (data["device0"], data["device1"]) == ("0", "1")
    assert Path(data["cache0"]).resolve() == (cache_root / "gpu0").resolve()
    assert Path(data["cache1"]).resolve() == (cache_root / "gpu1").resolve()
    assert data["cache0"] != data["cache1"]
    assert (int(data["shutdown_pid0"]), int(data["shutdown_pid1"])) == (
        pid0,
        pid1,
    )
    assert (int(data["exit0"]), int(data["exit1"])) == (0, 0)
    batches = list(batch_re.finditer(data["body"]))
    assert batches, (log_path, pid0, pid1)
    for batch in batches:
        values = batch.groupdict()
        for worker in ("0", "1"):
            shard = int(values[f"shard{worker}"])
            results = int(values[f"results{worker}"])
            nonempty = int(values[f"nonempty{worker}"])
            latency = float(values[f"latency{worker}"])
            assert results == shard > 0
            assert 0 < nonempty <= results
            assert math.isfinite(latency) and latency >= 0
    recorded_pids.update((pid0, pid1))

for pid in sorted(recorded_pids):
    probe = subprocess.run(
        ["ps", "-p", str(pid), "-o", "pid="],
        check=False,
        capture_output=True,
        text=True,
    )
    assert probe.returncode != 0 and not probe.stdout.strip(), (
        f"orphan worker PID still alive: {pid}"
    )

print(f"OK: runtime metadata and orphan checks passed for {len(recorded_pids)} PIDs")
PY
```

For the normal uninterrupted smoke and resumed completion run, each
child-reported runtime pair must contain distinct positive PIDs,
`visible_device=0` and `visible_device=1`, and the expected distinct
`gpu0`/`gpu1` cache roots. Every batch must report each worker's result count
equal to its shard size, a positive nonempty count, and a finite nonnegative
latency. Each run must finish with `worker shutdown exitcodes=(0, 0)` and the
same PID pair. The final `ps -p PID` probes must find none of the recorded
children.

## Production

After both the coordinator output smoke and deterministic resume smoke gates
pass:

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
`completed: true`, and coherent processed/rejected counts and hashes. Its
uninterrupted coordinator log must also end with
`worker shutdown exitcodes=(0, 0)` for the recorded runtime PID pair, and
neither child PID may remain alive.

## Interruption and Resume

`Ctrl-C` stops workers, releases the owned lock, and returns exit code 130.
Already committed batches remain valid. Resume by running the builder again
with the same config, without running any cleanup command. The builder
validates the source prefix, IDs, hashes, counts, and immutable generation
contract before starting workers.

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
configured source, have an accepted count less than or equal to the configured
target (including an already complete pool), and have no V2
generation-contract fingerprint. The builder revalidates accepted and
rejected classifications before materializing V2 state. Source-owned fields
`prompt`, `numbers`, `target`, `source_index`, `gold_expr`, and `bucket` must
exactly match the source whenever present; a legacy row may not add one of
these fields when the source omits it. Accepted-only and rejected-only legacy
prefixes are supported: either side may contain the entire contiguous source
prefix while the other side is absent or empty.

Adoption is one-way. After a V2 manifest or V2 transaction journal exists,
the legacy builder must never write that directory. V2-to-legacy mixing is
prohibited. The legacy builder is a fallback only for a fresh empty output
directory or a directory that remains exclusively legacy-owned.
