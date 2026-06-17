#!/usr/bin/env bash
# Phase 2a: EGMAP formal — seed 123, llm_agg na3, all main-table datasets.
# Requires: preflight_egmap.py PASS (static + --smoke) per dataset before run.
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
MAX_CONCURRENT="${MAX_CONCURRENT:-8}"
BANK_SIZE="${BANK_SIZE:-100}"
TOP_K="${TOP_K:-3}"
FORCE="${FORCE:-0}"
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"

PY="${PY:-.venv/bin/python}"
mkdir -p logs result prompt memory splits

if ! formal_check_vllm_profile; then
  echo "ERROR: vLLM not ready for profile ${FORMAL_MODEL_PROFILE}" >&2
  exit 1
fi

echo "[EGMAP-P2a] profile=${FORMAL_MODEL_PROFILE} seed=${SEED} datasets=[${DATASETS}] graph=${GRAPH}"

for dataset in ${DATASETS}; do
  if formal_skip_vqa_unless_enabled "${dataset}"; then
    continue
  fi
  base_tag="egmap_formal_${dataset}_${GRAPH}_na${NA}_d${DEPTH}s${SAMPLE_SIZE}o${OPT_SIZE}seed${SEED}_b${BANK_SIZE}k${TOP_K}"
  tag="${base_tag}${FORMAL_TAG_SUFFIX}"
  out="result/${tag}.json"
  log="logs/${tag}_official.log"

  if [[ -s "${out}" && "${FORCE}" != "1" ]]; then
    echo "[skip] ${out} exists (FORCE=1 to rerun)"
    continue
  fi

  formal_set_vqa_mode "${dataset}"

  if [[ "${SKIP_PREFLIGHT}" != "1" ]]; then
    echo "[preflight] ${dataset} seed=${SEED}"
  if ! "${PY}" scripts/preflight_egmap.py --dataset "${dataset}" --seed "${SEED}" \
      --graph "${GRAPH}" --na "${NA}" --smoke --fast \
      > "logs/preflight_${dataset}_seed${SEED}.log" 2>&1; then
      echo "[FAIL] preflight ${dataset} — see logs/preflight_${dataset}_seed${SEED}.log" >&2
      continue
    fi
  fi

  echo "[$(date '+%F %T')] [run] EGMAP dataset=${dataset} seed=${SEED} -> ${log}"
  if "${PY}" run_egmap_formal_one_seed.py \
    --dataset "${dataset}" \
    --graph "${GRAPH}" \
    --na "${NA}" \
    --nr "${NR}" \
    --seed "${SEED}" \
    --opt-size "${OPT_SIZE}" \
    --sample-size "${SAMPLE_SIZE}" \
    --depth "${DEPTH}" \
    --max-concurrent "${MAX_CONCURRENT}" \
    --bank-size "${BANK_SIZE}" \
    --top-k "${TOP_K}" \
    > "${log}" 2>&1; then
    "${PY}" scripts/preflight_egmap.py --dataset "${dataset}" --seed "${SEED}" \
      --graph "${GRAPH}" --check-eval \
      > "logs/postflight_${dataset}_seed${SEED}.log" 2>&1 || true
    ms="result/maspo_formal_${dataset}_${GRAPH}_na${NA}_d${DEPTH}s${SAMPLE_SIZE}o${OPT_SIZE}seed${SEED}${FORMAL_TAG_SUFFIX}.json"
    if [[ -f "${ms}" ]]; then
      echo "[fair] EGMAP+MASPO ${dataset} seed=${SEED}"
      "${PY}" scripts/fair_pair_postprocess.py --dataset "${dataset}" --seed "${SEED}" \
        --model-suffix "${FORMAL_MODEL_PROFILE}" --write \
        >> "logs/fair_${dataset}_seed${SEED}${FORMAL_TAG_SUFFIX}.log" 2>&1 || true
    fi
    "${PY}" scripts/update_result_ledger.py --seed "${SEED}" --graph "${GRAPH}"
    acc=$("${PY}" -c "import json; print(json.load(open('${out}'))['graph_types']['llm_agg']['accuracy'])" 2>/dev/null || echo "?")
    echo "[$(date '+%F %T')] [done] ${out} acc=${acc}"
  else
    echo "[$(date '+%F %T')] [FAIL] EGMAP ${dataset} seed=${SEED} (see ${log})" >&2
  fi
done

echo "[EGMAP-P2a] complete"
