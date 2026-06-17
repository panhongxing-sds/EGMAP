#!/usr/bin/env bash
# m4b text-only: MASPO (remaining) → EGMAP (all text). VQA deferred until RUN_VQA=1.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG="logs/campaign_m4b_text.log"
exec >>"$LOG" 2>&1

echo "=== m4b TEXT $(date -Is) RUN_VQA=${RUN_VQA:-0} ==="

if ! curl -sf http://127.0.0.1:8005/v1/models >/dev/null 2>&1; then
  echo "ERROR: text vLLM :8005 not ready"
  exit 1
fi

export MODEL_PROFILE="${MODEL_PROFILE:-single_4b}"
export SEED="${SEED:-123}"
export RUN_VQA=0
export MAX_CONCURRENT="${MAX_CONCURRENT:-16}"
export EGMAP_MAX_CONCURRENT="${EGMAP_MAX_CONCURRENT:-8}"
export SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-1}"
PY=".venv/bin/python"

TEXT_ALL="math500 aqua gpqa agieval humaneval"

echo "[phase] MASPO text all: ${TEXT_ALL}"
export DATASETS="${TEXT_ALL}"
bash scripts/run_maspo_official_phase1.sh || true

echo "[phase] EGMAP text all: ${TEXT_ALL}"
export DATASETS="${TEXT_ALL}"
export MAX_CONCURRENT="${EGMAP_MAX_CONCURRENT}"
bash scripts/run_egmap_official_phase2a.sh || true

export RERUN_MASPO=1
bash scripts/fair_all_pairs.sh >>"$LOG" 2>&1 || true
"$PY" scripts/update_result_ledger.py --seed "${SEED}" --graph llm_agg || true
echo "=== m4b TEXT DONE $(date -Is) ==="
