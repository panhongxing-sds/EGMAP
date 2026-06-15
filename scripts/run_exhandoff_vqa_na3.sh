#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env_unified.sh

mkdir -p logs result prompt stats memory

DATASETS="${DATASETS:-vqarad slake chartqa textvqa}"
SEEDS="${SEEDS:-123 42 456}"
SAMPLE_SIZE="${SAMPLE_SIZE:-200}"
OPT_SIZE="${OPT_SIZE:-100}"
DEPTH="${DEPTH:-3}"
MAX_CONCURRENT="${MAX_CONCURRENT:-2}"
export MASPO_FIXED_DEPTH="$DEPTH"
export MASPO_FIXED_ROUNDS_PER_TURN="${ROUNDS_PER_TURN:-3}"

for seed in $SEEDS; do
  for dataset in $DATASETS; do
    out="result/egmap_${dataset}_llm_agg_na3_d${DEPTH}s${SAMPLE_SIZE}o${OPT_SIZE}seed${seed}.json"
    log="logs/egmap_${dataset}_vqa_na3_seed${seed}.log"
    if [[ -s "$out" ]]; then
      echo "[skip] $out"
      continue
    fi
    echo "[$(date '+%F %T')] ExHandoff VQA na3 dataset=$dataset seed=$seed"
    V11_1_VQA_MODE="$dataset" python run_maspo.py \
      --dataset "$dataset" \
      --graph llm_agg --na 3 \
      --optimize --fixed-rounds --beam-refresh --lookahead-score --misleading-sampling \
      --experience-guided \
      --seed "$seed" --sample-size "$SAMPLE_SIZE" --opt-size "$OPT_SIZE" --depth "$DEPTH" \
      --max-concurrent "$MAX_CONCURRENT" \
      > "$log" 2>&1
    latest=$(ls -t result/${dataset}_llm_agg_*egmap.json 2>/dev/null | head -1 || true)
    [[ -n "$latest" ]] && cp "$latest" "$out"
  done
done
