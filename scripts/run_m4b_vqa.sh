#!/usr/bin/env bash
# m4b VQA only — run AFTER multimodal 4B vLLM @ :8005 is ready.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG="logs/campaign_m4b_vqa.log"
exec >>"$LOG" 2>&1

echo "=== m4b VQA $(date -Is) ==="

if ! curl -sf http://127.0.0.1:8005/v1/models >/dev/null 2>&1; then
  echo "ERROR: multimodal vLLM :8005 not ready"
  echo "  bash scripts/restart_vllm_4b_multimodal.sh"
  exit 1
fi

export MODEL_PROFILE="${MODEL_PROFILE:-single_4b}"
export SEED="${SEED:-123}"
export RUN_VQA=1
export MAX_CONCURRENT="${MAX_CONCURRENT:-12}"
export EGMAP_MAX_CONCURRENT="${EGMAP_MAX_CONCURRENT:-8}"
export SKIP_PREFLIGHT=1
PY=".venv/bin/python"
VQA_DS="vqarad slake chartqa"

echo "[phase] MASPO VQA: ${VQA_DS}"
export DATASETS="${VQA_DS}"
bash scripts/run_maspo_official_phase1.sh || true

echo "[phase] EGMAP VQA: ${VQA_DS}"
export MAX_CONCURRENT="${EGMAP_MAX_CONCURRENT}"
bash scripts/run_egmap_official_phase2a.sh || true

export DATASETS="${VQA_DS}"
export RERUN_MASPO=1
bash scripts/fair_all_pairs.sh >>"$LOG" 2>&1 || true
"$PY" scripts/update_result_ledger.py --seed "${SEED}" --graph llm_agg || true
echo "=== m4b VQA DONE $(date -Is) ==="
