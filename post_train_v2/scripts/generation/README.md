# V2 Teacher Generation

The production entrypoint starts two independent TP=1 vLLM engines, preserves
source order, derives a stable seed for every request, validates each full
response with exact Countdown arithmetic, and transactionally publishes
accepted/rejected JSONL plus Manifest V2.

Prerequisites:

- activate the pinned `post_train_v2/.venv`;
- ensure `post_train/model/qwen/qwen3-8b` exists;
- build `post_train_v2/data/processed/train_candidates.jsonl`;
- expose two healthy GPUs as physical devices 0 and 1;
- pass the environment dual-engine smoke gate.

Run the tracked fixture smoke:

```bash
python post_train_v2/scripts/generation/build_teacher_pool.py \
  --config post_train_v2/configs/generation/teacher_rollout_2gpu_smoke.yaml
```

Run production:

```bash
python post_train_v2/scripts/generation/build_teacher_pool.py \
  --config post_train_v2/configs/generation/teacher_rollout_2gpu.yaml
```

The command returns `0` only when the accepted target is reached. It returns
`2` when the source is exhausted first and publishes a
`partial_teacher_pool` manifest with `completed=false`.

Resume by rerunning the same command without deleting outputs. If a dead
process left `.teacher_pool.lock`, inspect its hostname and PID, then use
`--recover-stale-lock`. Never remove `.teacher_pool.transaction.json`;
recovery validates and rolls it back automatically.

Inspect the published state:

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("post_train_v2/data/teacher_rollouts/manifest.json")
manifest = json.loads(path.read_text(encoding="utf-8"))
print(manifest["artifact_id"])
print(manifest["stage_metadata"]["completed"])
print(manifest["stage_metadata"]["accepted_count"])
print(manifest["stage_metadata"]["teacher_state"])
PY
```

The accepted file contains full responses, not only answer expressions. Each
row conforms to the canonical SFT schema and records exact validation and
Teacher provenance.
