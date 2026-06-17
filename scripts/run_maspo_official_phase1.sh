#!/usr/bin/env bash
# Phase 1: Official MASPO baseline — seed 123 only, all main-table datasets.
# Optimize node prompts + single-path frozen eval (no handoff/residual/experience).
#
# Usage:
#   bash scripts/run_maspo_official_phase1.sh              # all datasets seed 123
#   DATASETS=math500 bash scripts/run_maspo_official_phase1.sh
#   FORCE=1 bash scripts/run_maspo_official_phase1.sh    # overwrite invalid old json
set -euo pipefail
cd "$(dirname "$0")/.."

source scripts/formal_model_profiles.sh
MODEL_PROFILE="${MODEL_PROFILE:-single_4b}"
formal_apply_model_profile "${MODEL_PROFILE}"
formal_apply_tok8192_env

AFS_HOME="/mnt/afs/L202500372"

SEED="${SEED:-123}"
DATASETS="${DATASETS:-math500 aqua gpqa agieval humaneval vqarad slake chartqa}"
GRAPH="${GRAPH:-llm_agg}"
NA="${NA:-3}"
NR="${NR:-1}"
DEPTH="${DEPTH:-3}"
OPT_SIZE="${OPT_SIZE:-100}"
SAMPLE_SIZE="${SAMPLE_SIZE:-200}"
MAX_CONCURRENT="${MAX_CONCURRENT:-16}"
FORCE="${FORCE:-0}"

BANK_SIZE="${BANK_SIZE:-100}"
TOP_K="${TOP_K:-3}"

PY="${PY:-.venv/bin/python}"
mkdir -p logs result prompt result/_invalid_pseudo_maspo

if ! formal_check_vllm_profile; then
  echo "ERROR: vLLM not ready for profile ${FORMAL_MODEL_PROFILE}" >&2
  echo "  dual: bash scripts/start_vllm_dual_4b9b.sh" >&2
  echo "  single: bash ${AFS_HOME}/bootstrap/serve-qwen35.sh 9b --port 8001" >&2
  exit 1
fi

echo "[MASPO-P1] profile=${FORMAL_MODEL_PROFILE} (${FORMAL_MODEL_LABEL}) suffix=${FORMAL_TAG_SUFFIX}"
echo "[MASPO-P1] seed=${SEED} datasets=[${DATASETS}] tok8192 official baseline"
echo "[MASPO-P1] WORK_MAX_TOKENS=${MASPO_WORK_MAX_TOKENS} PROMPT_CHARS=${MASPO_WORK_MAX_PROMPT_CHARS}"

for dataset in ${DATASETS}; do
  if formal_skip_vqa_unless_enabled "${dataset}"; then
    continue
  fi
  tag="maspo_formal_${dataset}_${GRAPH}_na${NA}_d${DEPTH}s${SAMPLE_SIZE}o${OPT_SIZE}seed${SEED}${FORMAL_TAG_SUFFIX}"
  out="result/${tag}.json"
  log="logs/${tag}_official.log"
  prompt="prompt/${tag}_prompts.json"

  if [[ -s "${out}" && "${FORCE}" != "1" ]]; then
    if "${PY}" -c "
import json, sys
d = json.load(open(sys.argv[1]))
si = d.get('split_info') or {}
bad = d.get('residual_selector') or si.get('residual_selector') or d.get('disagreement_handoff') or si.get('handoff')
sys.exit(1 if bad else 0)
" "${out}" 2>/dev/null; then
      echo "[skip] ${out} (valid official result exists; FORCE=1 to rerun)"
      continue
    fi
    echo "[archive] invalid pseudo-MASPO -> result/_invalid_pseudo_maspo/$(basename "${out}")"
    mv -f "${out}" "result/_invalid_pseudo_maspo/$(basename "${out}")"
  elif [[ -s "${out}" && "${FORCE}" == "1" ]]; then
    echo "[archive] FORCE rerun -> result/_invalid_pseudo_maspo/$(basename "${out}")"
    mv -f "${out}" "result/_invalid_pseudo_maspo/$(basename "${out}")"
  fi

  formal_set_vqa_mode "${dataset}"
  echo "[$(date '+%F %T')] [run] official MASPO dataset=${dataset} seed=${SEED} -> ${log}"
  if "${PY}" run_maspo_formal_one_seed.py \
    --dataset "${dataset}" \
    --graph "${GRAPH}" \
    --na "${NA}" \
    --nr "${NR}" \
    --seed "${SEED}" \
    --opt-size "${OPT_SIZE}" \
    --sample-size "${SAMPLE_SIZE}" \
    --depth "${DEPTH}" \
    --max-concurrent "${MAX_CONCURRENT}" \
    > "${log}" 2>&1; then
    acc=$("${PY}" -c "import json; d=json.load(open('${out}')); print(d['graph_types']['llm_agg']['accuracy'])" 2>/dev/null || echo "?")
    echo "[$(date '+%F %T')] [done] ${out} acc=${acc}"
    eg="result/egmap_formal_${dataset}_${GRAPH}_na${NA}_d${DEPTH}s${SAMPLE_SIZE}o${OPT_SIZE}seed${SEED}_b${BANK_SIZE}k${TOP_K}${FORMAL_TAG_SUFFIX}.json"
    if [[ -f "${eg}" ]]; then
      echo "[fair] MASPO+EGMAP ${dataset} seed=${SEED}"
      "${PY}" scripts/fair_pair_postprocess.py --dataset "${dataset}" --seed "${SEED}" \
        --model-suffix "${FORMAL_MODEL_PROFILE}" --write \
        >> "logs/fair_${dataset}_seed${SEED}${FORMAL_TAG_SUFFIX}.log" 2>&1 || true
    fi
    "${PY}" scripts/update_result_ledger.py --seed "${SEED}" --graph "${GRAPH}" || true
  else
    echo "[$(date '+%F %T')] [FAIL] ${dataset} seed=${SEED} (see ${log})" >&2
  fi
done

echo "[MASPO-P1] final RESULT ledger sync"
"${PY}" scripts/update_result_ledger.py --seed "${SEED}" --graph "${GRAPH}" || true

echo "[MASPO-P1] complete"
