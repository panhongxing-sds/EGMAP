#!/usr/bin/env bash
# Re-run official MASPO frozen eval only (--skip-optimize), aligned to EGMAP split manifest.
# Usage:
#   bash scripts/rerun_maspo_fair_eval.sh math500
#   bash scripts/rerun_maspo_fair_eval.sh aqua gpqa
#   SEED=123 MODEL_PROFILE=single_4b bash scripts/rerun_maspo_fair_eval.sh math500 aqua
set -euo pipefail
cd "$(dirname "$0")/.."

source scripts/formal_model_profiles.sh
MODEL_PROFILE="${MODEL_PROFILE:-single_4b}"
formal_apply_model_profile "${MODEL_PROFILE}"
formal_apply_tok8192_env

SEED="${SEED:-123}"
DATASETS="${*:-math500 aqua gpqa agieval humaneval}"
MAX_CONCURRENT="${MAX_CONCURRENT:-8}"
PY="${PY:-.venv/bin/python}"

if ! formal_check_vllm_profile; then
  echo "ERROR: vLLM not ready for ${FORMAL_MODEL_PROFILE}" >&2
  exit 1
fi

for dataset in ${DATASETS}; do
  if [[ "${dataset}" =~ ^(vqarad|slake|chartqa|textvqa|pmcvqa)$ ]] && [[ "${RUN_VQA:-0}" != "1" ]]; then
    echo "[skip] VQA ${dataset} (RUN_VQA=0)"
    continue
  fi
  log="logs/maspo_fair_eval_${dataset}_seed${SEED}${FORMAL_TAG_SUFFIX}.log"
  echo "[$(date '+%F %T')] MASPO fair eval ${dataset} seed=${SEED} -> ${log}"
  formal_set_vqa_mode "${dataset}" 2>/dev/null || true
  "${PY}" run_maspo_formal_one_seed.py \
    --dataset "${dataset}" \
    --graph llm_agg \
    --na 3 \
    --seed "${SEED}" \
    --opt-size 100 \
    --sample-size 200 \
    --depth 3 \
    --max-concurrent "${MAX_CONCURRENT}" \
    --skip-optimize \
    > "${log}" 2>&1
  echo "[$(date '+%F %T')] done ${dataset}"
done
