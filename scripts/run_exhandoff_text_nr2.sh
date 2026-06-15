#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env_unified.sh

mkdir -p logs result prompt stats memory

DATASETS="${DATASETS:-math500 agieval aqua gpqa humaneval}"
SEEDS="${SEEDS:-123 42 456}"
SAMPLE_SIZE="${SAMPLE_SIZE:-200}"
OPT_SIZE="${OPT_SIZE:-100}"
DEPTH="${DEPTH:-3}"
MAX_CONCURRENT="${MAX_CONCURRENT:-4}"
export MASPO_FIXED_DEPTH="$DEPTH"
export MASPO_FIXED_ROUNDS_PER_TURN="${ROUNDS_PER_TURN:-3}"

for seed in $SEEDS; do
  for dataset in $DATASETS; do
    out="result/egmap_${dataset}_reflect_nr2_d${DEPTH}s${SAMPLE_SIZE}o${OPT_SIZE}seed${seed}.json"
    log="logs/egmap_${dataset}_nr2_seed${seed}.log"
    if [[ -s "$out" ]]; then
      echo "[skip] $out"
      continue
    fi
    echo "[$(date '+%F %T')] ExHandoff nr2 dataset=$dataset seed=$seed"
    python run_maspo.py \
      --dataset "$dataset" \
      --graph reflect --nr 2 \
      --optimize --fixed-rounds --beam-refresh --lookahead-score --misleading-sampling \
      --experience-guided \
      --seed "$seed" --sample-size "$SAMPLE_SIZE" --opt-size "$OPT_SIZE" --depth "$DEPTH" \
      --max-concurrent "$MAX_CONCURRENT" \
      > "$log" 2>&1
    latest=$(ls -t result/${dataset}_reflect_*egmap.json 2>/dev/null | head -1 || true)
    [[ -n "$latest" ]] && cp "$latest" "$out"
  done
done
