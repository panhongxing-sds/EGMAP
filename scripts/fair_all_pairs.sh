#!/usr/bin/env bash
# Fair EGMAP vs MASPO for all text formal pairs: manifest sync, rescore, union-prune.
#
# Usage:
#   bash scripts/fair_all_pairs.sh                    # postprocess only (ready pairs)
#   RERUN_MASPO=1 bash scripts/fair_all_pairs.sh      #补跑 MASPO eval then postprocess
#   SEED=123 MODEL_SUFFIX=m4b DATASETS="math500 aqua" bash scripts/fair_all_pairs.sh
set -euo pipefail
cd "$(dirname "$0")/.."

SEED="${SEED:-123}"
MODEL_SUFFIX="${MODEL_SUFFIX:-m4b}"
DATASETS="${DATASETS:-math500 aqua gpqa agieval humaneval}"
RERUN_MASPO="${RERUN_MASPO:-0}"
PY="${PY:-.venv/bin/python}"
LOG="logs/fair_all_pairs_seed${SEED}_${MODEL_SUFFIX}.log"
mkdir -p logs

exec >>"${LOG}" 2>&1
echo "=== fair_all_pairs $(date -Is) seed=${SEED} suffix=${MODEL_SUFFIX} RERUN_MASPO=${RERUN_MASPO} ==="

need_maspo=()
for ds in ${DATASETS}; do
  ms="result/maspo_formal_${ds}_llm_agg_na3_d3s200o100seed${SEED}_${MODEL_SUFFIX}.json"
  eg="result/egmap_formal_${ds}_llm_agg_na3_d3s200o100seed${SEED}_b100k3_${MODEL_SUFFIX}.json"
  man="splits/egmap_formal_${ds}_llm_agg_na3_d3s200o100seed${SEED}_b100k3_split.json"
  [[ -f "${man}" ]] || { echo "[skip] ${ds}: no manifest"; continue; }
  if [[ ! -f "${eg}" ]]; then
    echo "[wait] ${ds}: EGMAP result missing"
    continue
  fi
  if [[ ! -f "${ms}" ]]; then
    echo "[need] ${ds}: MASPO result missing"
    need_maspo+=("${ds}")
    continue
  fi
  miss=$("${PY}" -c "
import json, sys
man = json.load(open(sys.argv[1]))
run = man.get('eval_unique_ids_run') or []
excl = set(man.get('excluded_unscoreable') or [])
expected = [u for u in run if u not in excl]
ms = {x['unique_id'] for x in json.load(open(sys.argv[2]))['detailed']}
fair = json.load(open(sys.argv[2])).get('fair_eval') or {}
# Already fair-processed with full manifest coverage (post-prune denominator).
if fair.get('policy') and fair.get('eval_ids_present') == len(expected) and not (set(expected) - ms):
    print(0)
else:
    print(len(set(expected) - ms))
" "${man}" "${ms}")
  if [[ "${miss}" != "0" ]]; then
    echo "[need] ${ds}: MASPO missing ${miss} manifest id(s)"
    need_maspo+=("${ds}")
  fi
done

if [[ "${RERUN_MASPO}" == "1" && ${#need_maspo[@]} -gt 0 ]]; then
  echo "[rerun] MASPO fair eval: ${need_maspo[*]}"
  export SEED MODEL_PROFILE=single_4b
  bash scripts/rerun_maspo_fair_eval.sh "${need_maspo[@]}"
fi

for ds in ${DATASETS}; do
  ms="result/maspo_formal_${ds}_llm_agg_na3_d3s200o100seed${SEED}_${MODEL_SUFFIX}.json"
  eg="result/egmap_formal_${ds}_llm_agg_na3_d3s200o100seed${SEED}_b100k3_${MODEL_SUFFIX}.json"
  if [[ -f "${ms}" && -f "${eg}" ]]; then
    echo "[fair] ${ds}"
    "${PY}" scripts/fair_pair_postprocess.py --dataset "${ds}" --seed "${SEED}" \
      --model-suffix "${MODEL_SUFFIX}" --write || echo "[FAIL] fair ${ds}"
  fi
done

"${PY}" scripts/update_result_ledger.py --seed "${SEED}" --graph llm_agg || true
echo "=== fair_all_pairs DONE $(date -Is) ==="
