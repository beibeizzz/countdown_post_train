#!/usr/bin/env bash
# Run held-out TEST (500 rows) evaluation for all six post-training stages,
# one model at a time, on the single visible GPU.
#
# Each model is evaluated by post_train/scripts/eval/evaluate_model.py, which
# (as of this version) runs on the test split and uses batched inference
# (cfg batch_size, default 32). Output lands in post_train/data/eval/<name>/.
#
# Usage:
#   bash post_train/scripts/eval/run_all_evals.sh            # run all six
#   bash post_train/scripts/eval/run_all_evals.sh --no-batch # force serial (batch=1) everywhere
#   bash post_train/scripts/eval/run_all_evals.sh --smoke    # only first 50 rows per model (fast sanity check)
#   bash post_train/scripts/eval/run_all_evals.sh --models sft_full,dpo   # run a subset
#
# Models already evaluated (eval_metrics.json present) are skipped unless
# REEVAL=1 is set, so the script is safely re-runnable.
set -euo pipefail

# Resolve repo root from this script's location regardless of CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

CONFIG="post_train/configs/eval.yaml"
EVAL_SCRIPT="post_train/scripts/eval/evaluate_model.py"
EVAL_ROOT="post_train/data/eval"
BASE_MODEL="post_train/model/qwen/qwen3-0.6b"

EXTRA_ARGS=()
SMOKE=0
ONLY_MODELS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-batch) EXTRA_ARGS+=("--no-batch"); shift ;;
    --batch-size) EXTRA_ARGS+=("--batch-size" "$2"); shift 2 ;;
    --smoke) SMOKE=1; shift ;;
    --models) ONLY_MODELS="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,18p' "$0"; exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ $SMOKE -eq 1 ]]; then
  EXTRA_ARGS+=("--limit" "50")
  echo "[run_all_evals] SMOKE mode: --limit 50 (per model)"
fi

# name|model-path|[base-model-path flag]
declare -a TARGETS=(
  "base_0_6b|${BASE_MODEL}|"
  "sft_full|post_train/outputs/sft/full/final|"
  "sft_lora|post_train/outputs/sft/lora/final|--base-model-path ${BASE_MODEL}"
  "rft|post_train/outputs/sft/rft/final|"
  "dpo|post_train/outputs/dpo/final|"
  "grpo|post_train/outputs/grpo/final|"
  "grpo_trl|post_train/outputs/grpo_trl/final|"
  "opd_gkd|post_train/outputs/opd/gkd/final|"
)

if [[ -n "$ONLY_MODELS" ]]; then
  FILTERED=()
  IFS=',' read -ra WANTED <<< "$ONLY_MODELS"
  for entry in "${TARGETS[@]}"; do
    name="${entry%%|*}"
    for w in "${WANTED[@]}"; do [[ "$name" == "$w" ]] && FILTERED+=("$entry"); done
  done
  TARGETS=("${FILTERED[@]}")
  [[ ${#TARGETS[@]} -eq 0 ]] && { echo "[run_all_evals] no matching models in --models '$ONLY_MODELS'" >&2; exit 2; }
fi

echo "[run_all_evals] repo_root=$REPO_ROOT"
echo "[run_all_evals] config=$CONFIG  extra_args=[${EXTRA_ARGS[*]}]"
echo "[run_all_evals] models: $(printf '%s ' "${TARGETS[@]%%|*}")"
echo

FAILED=()
SKIPPED=()
DONE=()

for entry in "${TARGETS[@]}"; do
  IFS='|' read -r name model_path base_flag <<< "$entry"
  out_dir="${EVAL_ROOT}/${name}"
  metrics="${out_dir}/eval_metrics.json"

  if [[ -f "$metrics" && "${REEVAL:-0}" != "1" ]]; then
    echo "[run_all_evals] SKIP $name (already evaluated: $metrics); set REEVAL=1 to re-run"
    SKIPPED+=("$name")
    continue
  fi

  if [[ ! -e "$model_path" ]]; then
    echo "[run_all_evals] SKIP $name (model path missing: $model_path)"
    FAILED+=("$name(missing)")
    continue
  fi

  echo "=============================="
  echo "[run_all_evals] EVAL $name"
  echo "  model: $model_path"
  [[ -n "$base_flag" ]] && echo "  $base_flag"
  echo "  out:   $out_dir"
  echo "=============================="
  t0=$(date +%s)

  # shellcheck disable=SC2086
  if python "$EVAL_SCRIPT" \
      --config "$CONFIG" \
      --model-path "$model_path" \
      $base_flag \
      --output-dir "$out_dir" \
      "${EXTRA_ARGS[@]}"; then
    t1=$(date +%s)
    echo "[run_all_evals] DONE $name in $((t1 - t0))s"
    DONE+=("$name")
  else
    rc=$?
    t1=$(date +%s)
    echo "[run_all_evals] FAIL $name (exit $rc) after $((t1 - t0))s" >&2
    FAILED+=("$name")
  fi
  echo
done

echo "=============================="
echo "[run_all_evals] SUMMARY"
echo "  done:   ${DONE[*]:-<none>}"
echo "  skipped:${SKIPPED[*]:- <none>}"
echo "  failed: ${FAILED[*]:-<none>}"
echo "=============================="
echo

echo "[run_all_evals] metrics:"
for entry in "${TARGETS[@]}"; do
  name="${entry%%|*}"
  m="${EVAL_ROOT}/${name}/eval_metrics.json"
  if [[ -f "$m" ]]; then
    printf '  %-12s ' "$name"
    python -c "import json,sys; d=json.load(open('$m')); print({k:(round(v,4) if isinstance(v,float) else v) for k,v in d.items()})" 2>/dev/null || cat "$m"
    echo
  else
    printf '  %-12s (no metrics)\n' "$name"
  fi
done

[[ ${#FAILED[@]} -gt 0 ]] && exit 1 || exit 0
