#!/usr/bin/env bash
# Wait for the in-flight EGMAP batch, then backfill all paired MASPO baselines + export table.
set -euo pipefail
cd "$(dirname "$0")/.."
chmod +x scripts/run_maspo_formal_all.sh

EGMAP_PID="${1:-}"
LOG="logs/wait_egmap_then_maspo.log"

if [[ -n "${EGMAP_PID}" ]] && kill -0 "${EGMAP_PID}" 2>/dev/null; then
  echo "[$(date '+%F %T')] waiting for EGMAP batch pid=${EGMAP_PID}" | tee -a "${LOG}"
  while kill -0 "${EGMAP_PID}" 2>/dev/null; do
    sleep 60
  done
  echo "[$(date '+%F %T')] EGMAP batch exited" | tee -a "${LOG}"
else
  echo "[$(date '+%F %T')] no live EGMAP pid; proceed to MASPO backfill" | tee -a "${LOG}"
fi

bash scripts/run_maspo_formal_all.sh 2>&1 | tee -a "${LOG}"
.venv/bin/python scripts/export_egmap_maspo_table.py --auto -o result/comparison_table.md 2>&1 | tee -a "${LOG}"
echo "[$(date '+%F %T')] pipeline follow-up complete -> result/comparison_table.md" | tee -a "${LOG}"
