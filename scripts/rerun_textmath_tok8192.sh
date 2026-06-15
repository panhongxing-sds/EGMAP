#!/usr/bin/env bash
# Full re-run: math500 / agieval / aqua / gpqa × seeds 123/42/456 @ max_tokens=8192.
# EGMAP + MASPO fair (same MASPO_WORK_MAX_TOKENS), reuse optimized prompts (--skip-optimize).
set -euo pipefail
cd "$(dirname "$0")/.."

export HANDOFF_DATASET_ROOT="/mnt/afs/L202500372/data/egmap_handoff"
source scripts/env_unified.sh
AFS_HOME="/mnt/afs/L202500372"
MODEL_PATH="${AFS_HOME}/models/Qwen3.5-9B"
source scripts/formal_common.sh
formal_apply_env "${AFS_HOME}" "${MODEL_PATH}" 8001
formal_apply_tok8192_env

DATASETS="${DATASETS:-math500 agieval aqua gpqa}"
SEEDS="${SEEDS:-123 42 456}"
LOG=logs/rerun_textmath_tok8192_master.log
mkdir -p logs result

if ! curl -sf "http://127.0.0.1:8001/v1/models" >/dev/null; then
  echo "[$(date '+%F %T')] ERROR: vLLM not on :8001" | tee -a "$LOG"
  exit 1
fi

echo "[$(date '+%F %T')] === full tok8192 rerun datasets=[${DATASETS}] seeds=[${SEEDS}] ===" | tee -a "$LOG"

for seed in ${SEEDS}; do
  for ds in ${DATASETS}; do
    EG=result/egmap_formal_${ds}_llm_agg_na3_d3s200o100seed${seed}_b100k3.json
    MS=result/maspo_formal_${ds}_llm_agg_na3_d3s200o100seed${seed}.json
    cell="${ds}:seed${seed}"
    echo "[$(date '+%F %T')] --- cell ${cell} @ tok8192 ---" | tee -a "$LOG"
    [[ -f "$EG" && ! -f "${EG}.tok4096bak" ]] && cp -f "$EG" "${EG}.tok4096bak"
    [[ -f "$MS" && ! -f "${MS}.tok4096bak" ]] && cp -f "$MS" "${MS}.tok4096bak"

    .venv/bin/python run_egmap_formal_one_seed.py \
      --dataset "$ds" --graph llm_agg --na 3 --seed "$seed" \
      --opt-size 100 --sample-size 200 --depth 3 --max-concurrent 4 \
      --bank-size 100 --top-k 3 --skip-optimize \
      >> "$LOG" 2>&1 \
      && echo "[$(date '+%F %T')] [done EGMAP] ${cell}" | tee -a "$LOG" \
      || { echo "[$(date '+%F %T')] [FAIL EGMAP] ${cell}" | tee -a "$LOG"; continue; }

    .venv/bin/python run_maspo_formal_baseline.py \
      --dataset "$ds" --graph llm_agg --na 3 --seed "$seed" \
      --opt-size 100 --sample-size 200 --depth 3 --max-concurrent 4 \
      --bank-size 100 --top-k 3 \
      >> "$LOG" 2>&1 \
      && echo "[$(date '+%F %T')] [done MASPO] ${cell}" | tee -a "$LOG" \
      || echo "[$(date '+%F %T')] [FAIL MASPO] ${cell}" | tee -a "$LOG"

    .venv/bin/python scripts/rescore_formal_clean.py "$EG" "$MS" --write >> "$LOG" 2>&1 || true
  done
done

echo "[$(date '+%F %T')] === full tok8192 rerun complete ===" | tee -a "$LOG"
.venv/bin/python scripts/export_egmap_maspo_table.py --auto -o result/comparison_table.md >> "$LOG" 2>&1 || true
