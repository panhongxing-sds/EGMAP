#!/usr/bin/env bash
# m4b campaign — same flow as the successful 2026-06-15 run:
#   standard vLLM @ :8005  →  run_maspo_official_phase1.sh  →  run_egmap_official_phase2a.sh
set -uo pipefail
cd "$(dirname "$0")/.."
LOG="logs/campaign_m4b_now.log"
exec >>"$LOG" 2>&1

echo "=== campaign_m4b_now $(date -Is) ==="

if ! curl -sf http://127.0.0.1:8005/v1/models >/dev/null 2>&1; then
  echo "[vLLM] :8005 not ready — start manually:"
  echo "  bash \${AFS_HOME:-/mnt/afs/L202500372}/bootstrap/serve-qwen35.sh 4b --port 8005"
  exit 1
fi
echo "[vLLM] :8005 ready"

export MODEL_PROFILE=single_4b
export SEED=123
export MAX_CONCURRENT="${MAX_CONCURRENT:-16}"
export EGMAP_MAX_CONCURRENT="${EGMAP_MAX_CONCURRENT:-8}"
export SKIP_PREFLIGHT=1
PY=".venv/bin/python"

export RUN_VQA=0
exec bash scripts/run_m4b_full_pipeline.sh

for f in result/maspo_formal_*_m4b.json result/egmap_formal_*_m4b.json; do
  [[ -f "$f" ]] && "$PY" scripts/prune_unscoreable_formal.py --write "$f" 2>/dev/null || true
done
"$PY" scripts/update_result_ledger.py --seed 123 --graph llm_agg || true
echo "=== campaign_m4b_now DONE $(date -Is) ==="
