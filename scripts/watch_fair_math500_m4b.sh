#!/usr/bin/env bash
# Wait for MASPO math500 fair rerun, then sync/rescore/prune EGMAP+MASPO pair.
set -euo pipefail
cd "$(dirname "$0")/.."
LOG=logs/maspo_math500_m4b_fair_rerun.log
PY=.venv/bin/python

echo "[fair-watch] waiting for MASPO math500 200-eval rerun..."
while [[ ! -f "$LOG" ]] || ! grep -q "saved eval ->" "$LOG"; do
  sleep 30
done
echo "[fair-watch] MASPO done $(date -Is)"

"$PY" scripts/fair_pair_postprocess.py --dataset math500 --seed 123 --model-suffix m4b --write \
  | tee logs/fair_math500_m4b_postprocess.log

"$PY" scripts/update_result_ledger.py --seed 123 --graph llm_agg || true
echo "[fair-watch] complete $(date -Is)"
