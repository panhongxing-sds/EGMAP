#!/usr/bin/env bash
# Validation: re-run agieval seed42 (worst regression cell) with max_tokens=8192
# for BOTH EGMAP and MASPO (fair), reusing the already-optimized prompts.
# Backs up the existing (token-4096, cleaned) result JSONs first.
set -euo pipefail
cd "$(dirname "$0")/.."

export HANDOFF_DATASET_ROOT="/mnt/afs/L202500372/data/egmap_handoff"
source scripts/env_unified.sh
AFS_HOME="/mnt/afs/L202500372"
MODEL_PATH="${AFS_HOME}/models/Qwen3.5-9B"
source scripts/formal_common.sh
formal_apply_env "${AFS_HOME}" "${MODEL_PATH}" 8001

export MASPO_WORK_MAX_TOKENS=8192   # the fix under test (was default 4096)

DS=agieval; SEED=42
EG=result/egmap_formal_${DS}_llm_agg_na3_d3s200o100seed${SEED}_b100k3.json
MS=result/maspo_formal_${DS}_llm_agg_na3_d3s200o100seed${SEED}.json

echo "[$(date '+%F %T')] backup token-4096 results -> *.tok4096bak"
[[ -f "$EG" ]] && cp -f "$EG" "${EG}.tok4096bak"
[[ -f "$MS" ]] && cp -f "$MS" "${MS}.tok4096bak"

echo "[$(date '+%F %T')] EGMAP agieval seed42 @ tok8192 (skip-optimize, reuse prompts)"
.venv/bin/python run_egmap_formal_one_seed.py \
  --dataset "$DS" --graph llm_agg --na 3 --seed "$SEED" \
  --opt-size 100 --sample-size 200 --depth 3 --max-concurrent 4 \
  --bank-size 100 --top-k 3 --skip-optimize

echo "[$(date '+%F %T')] MASPO agieval seed42 @ tok8192"
.venv/bin/python run_maspo_formal_baseline.py \
  --dataset "$DS" --graph llm_agg --na 3 --seed "$SEED" \
  --opt-size 100 --sample-size 200 --depth 3 --max-concurrent 4 \
  --bank-size 100 --top-k 3

echo "[$(date '+%F %T')] clean re-score the two new files"
.venv/bin/python scripts/rescore_formal_clean.py "$EG" "$MS" --write

echo "[$(date '+%F %T')] === validation done ==="
