#!/usr/bin/env bash
# Wait for humaneval EGMAP,补跑 MASPO, fair postprocess, ledger update.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/formal_model_profiles.sh
formal_apply_model_profile single_4b
formal_apply_tok8192_env

SEED=123
SUF=m4b
LOG=logs/fair_humaneval_wait_${SUF}.log
exec >>"${LOG}" 2>&1
echo "=== wait_fair_humaneval $(date -Is) ==="

EG="result/egmap_formal_humaneval_llm_agg_na3_d3s200o100seed${SEED}_b100k3_${SUF}.json"
while [[ ! -s "${EG}" ]]; do
  if ! pgrep -f "run_egmap_formal_one_seed.py --dataset humaneval" >/dev/null 2>&1; then
    echo "[warn] humaneval EGMAP process gone but ${EG} missing"
    break
  fi
  echo "[wait] $(date -Is) EGMAP humaneval still running..."
  sleep 300
done

if [[ ! -s "${EG}" ]]; then
  echo "[FAIL] EGMAP result missing: ${EG}"
  exit 1
fi
echo "[ok] EGMAP humaneval done: ${EG}"

echo "[rerun] MASPO humaneval fair eval"
bash scripts/rerun_maspo_fair_eval.sh humaneval

echo "[fair] humaneval pair postprocess"
.venv/bin/python scripts/fair_pair_postprocess.py --dataset humaneval --seed "${SEED}" \
  --model-suffix "${SUF}" --write

.venv/bin/python scripts/update_result_ledger.py --seed "${SEED}" --graph llm_agg
echo "=== wait_fair_humaneval DONE $(date -Is) ==="
