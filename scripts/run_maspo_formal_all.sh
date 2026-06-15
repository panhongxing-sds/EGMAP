#!/usr/bin/env bash
# MASPO formal baseline for all cells that already have EGMAP formal prompts + split manifest.
# Paired eval: same frozen eval ids as EGMAP (no experience).
set -euo pipefail
cd "$(dirname "$0")/.."

source scripts/env_unified.sh
# shellcheck source=formal_common.sh
source scripts/formal_common.sh

AFS_HOME="/mnt/afs/L202500372"
MODEL_PATH="${MODEL_PATH:-${AFS_HOME}/models/Qwen3.5-9B}"
PORT="${PORT:-8001}"

formal_apply_env "${AFS_HOME}" "${MODEL_PATH}" "${PORT}"

DATASETS="${DATASETS:-math500 agieval aqua gpqa humaneval}"
SEEDS="${SEEDS:-123 42 456}"
GRAPH="${GRAPH:-llm_agg}"
NA="${NA:-3}"
DEPTH="${DEPTH:-3}"
OPT_SIZE="${OPT_SIZE:-100}"
SAMPLE_SIZE="${SAMPLE_SIZE:-200}"
MAX_CONCURRENT="${MAX_CONCURRENT:-4}"
BANK_SIZE="${BANK_SIZE:-100}"
TOP_K="${TOP_K:-3}"

PY="${PY:-.venv/bin/python}"
mkdir -p logs result prompt splits

if ! curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
  echo "ERROR: vLLM not responding on port ${PORT}" >&2
  exit 1
fi

echo "[MASPO-ALL] datasets=[${DATASETS}] seeds=[${SEEDS}] graph=${GRAPH}"
echo "[MASPO-ALL] HANDOFF_DATASET_ROOT=${HANDOFF_DATASET_ROOT} (paired EGMAP formal, frozen eval manifest, no experience)"

for seed in ${SEEDS}; do
  for dataset in ${DATASETS}; do
    egmap_tag="egmap_formal_${dataset}_${GRAPH}_na${NA}_d${DEPTH}s${SAMPLE_SIZE}o${OPT_SIZE}seed${seed}_b${BANK_SIZE}k${TOP_K}"
    maspo_tag="maspo_formal_${dataset}_${GRAPH}_na${NA}_d${DEPTH}s${SAMPLE_SIZE}o${OPT_SIZE}seed${seed}"
    egmap_out="result/${egmap_tag}.json"
    maspo_out="result/${maspo_tag}.json"
  maspo_log="logs/${maspo_tag}.log"
    prompt_path="prompt/${egmap_tag}_prompts.json"
    handoff_path="prompt/${egmap_tag}_handoffs.json"

    if [[ -s "${maspo_out}" ]]; then
      echo "[skip] ${maspo_out} (exists)"
      continue
    fi
    if [[ ! -s "${egmap_out}" ]]; then
      echo "[wait] ${dataset} seed=${seed}: EGMAP result missing (${egmap_out}), skip MASPO"
      continue
    fi
    if [[ ! -f "${prompt_path}" || ! -f "${handoff_path}" ]]; then
      echo "[wait] ${dataset} seed=${seed}: EGMAP prompts missing, skip MASPO"
      continue
    fi
    split_manifest="splits/${egmap_tag}_split.json"
    if [[ ! -f "${split_manifest}" ]]; then
      echo "[FAIL] ${dataset} seed=${seed}: missing split manifest ${split_manifest}"
      continue
    fi

    formal_set_vqa_mode "${dataset}"
    echo "[$(date '+%F %T')] [run] MASPO dataset=${dataset} seed=${seed} manifest=${split_manifest} -> ${maspo_log}"
    "${PY}" run_maspo_formal_baseline.py \
      --dataset "${dataset}" \
      --graph "${GRAPH}" \
      --na "${NA}" \
      --seed "${seed}" \
      --opt-size "${OPT_SIZE}" \
      --sample-size "${SAMPLE_SIZE}" \
      --depth "${DEPTH}" \
      --max-concurrent "${MAX_CONCURRENT}" \
      --bank-size "${BANK_SIZE}" \
      --top-k "${TOP_K}" \
      > "${maspo_log}" 2>&1 \
      && echo "[$(date '+%F %T')] [done] ${maspo_out}" \
      || echo "[$(date '+%F %T')] [FAIL] MASPO dataset=${dataset} seed=${seed} (see ${maspo_log})"
  done
done

echo "[MASPO-ALL] complete; exporting comparison table + leakage audit"
"${PY}" scripts/export_egmap_maspo_table.py --auto -o result/comparison_table.md
if [[ -x scripts/verify_formal_no_leakage.py ]] || [[ -f scripts/verify_formal_no_leakage.py ]]; then
  for seed in ${SEEDS}; do
    for dataset in ${DATASETS}; do
      egmap_tag="egmap_formal_${dataset}_${GRAPH}_na${NA}_d${DEPTH}s${SAMPLE_SIZE}o${OPT_SIZE}seed${seed}_b${BANK_SIZE}k${TOP_K}"
      maspo_tag="maspo_formal_${dataset}_${GRAPH}_na${NA}_d${DEPTH}s${SAMPLE_SIZE}o${OPT_SIZE}seed${seed}"
      manifest="splits/${egmap_tag}_split.json"
      egmap_out="result/${egmap_tag}.json"
      maspo_out="result/${maspo_tag}.json"
      if [[ -f "${manifest}" && -s "${egmap_out}" && -s "${maspo_out}" ]]; then
        "${PY}" scripts/verify_formal_no_leakage.py \
          --manifest "${manifest}" \
          --egmap-result "${egmap_out}" \
          --maspo-result "${maspo_out}" \
          || echo "[warn] leakage check failed for ${dataset} seed=${seed}"
      fi
    done
  done
fi
