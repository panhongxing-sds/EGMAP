#!/usr/bin/env bash
# Formal EGMAP across all text datasets x all seeds, using run_egmap_formal_one_seed.py.
# Single-9B serving config (work=strong=judge @ port 8001).
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
SKIP_OPTIMIZE="${SKIP_OPTIMIZE:-0}"

PY="${PY:-.venv/bin/python}"
mkdir -p logs result prompt stats memory splits

if ! curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
  echo "ERROR: vLLM not responding on port ${PORT}" >&2
  exit 1
fi

echo "[ALL] datasets=[${DATASETS}] seeds=[${SEEDS}] graph=${GRAPH} na=${NA} depth=${DEPTH} opt=${OPT_SIZE} sample=${SAMPLE_SIZE} bank=${BANK_SIZE} top_k=${TOP_K} model=${MODEL_PATH}"

for seed in ${SEEDS}; do
  for dataset in ${DATASETS}; do
    tag="egmap_formal_${dataset}_${GRAPH}_na${NA}_d${DEPTH}s${SAMPLE_SIZE}o${OPT_SIZE}seed${seed}_b${BANK_SIZE}k${TOP_K}"
    out="result/${tag}.json"
    log="logs/${tag}.log"
    if [[ -s "${out}" ]]; then
      echo "[skip] ${out} (exists)"
      continue
    fi
    formal_set_vqa_mode "${dataset}"
    echo "[$(date '+%F %T')] [run] dataset=${dataset} seed=${seed} top_k=${TOP_K} -> ${log}"
    EGMAP_EXTRA=()
    if [[ "${SKIP_OPTIMIZE}" == "1" ]]; then
      EGMAP_EXTRA+=(--skip-optimize)
    fi
    "${PY}" run_egmap_formal_one_seed.py \
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
      "${EGMAP_EXTRA[@]}" \
      > "${log}" 2>&1 \
      && echo "[$(date '+%F %T')] [done] ${out}" \
      || echo "[$(date '+%F %T')] [FAIL] dataset=${dataset} seed=${seed} (see ${log})"

    # Paired MASPO baseline (same frozen eval, no experience) — only if EGMAP succeeded.
    if [[ -s "${out}" ]]; then
      maspo_tag="maspo_formal_${dataset}_${GRAPH}_na${NA}_d${DEPTH}s${SAMPLE_SIZE}o${OPT_SIZE}seed${seed}"
      maspo_out="result/${maspo_tag}.json"
      maspo_log="logs/${maspo_tag}.log"
      if [[ -s "${maspo_out}" ]]; then
        echo "[skip] ${maspo_out} (MASPO exists)"
      elif [[ -f "prompt/${tag}_prompts.json" && -f "prompt/${tag}_handoffs.json" ]]; then
        formal_set_vqa_mode "${dataset}"
        echo "[$(date '+%F %T')] [run] MASPO dataset=${dataset} seed=${seed} -> ${maspo_log}"
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
      fi
    fi
  done
done

echo "[ALL] EGMAP complete"
