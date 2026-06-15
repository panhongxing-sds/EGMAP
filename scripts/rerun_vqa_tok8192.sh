#!/usr/bin/env bash
# vqarad / slake / chartqa × 3 seeds @ max_tokens=8192. Requires multimodal vLLM on :8001.
set -euo pipefail
cd "$(dirname "$0")/.."

export HANDOFF_DATASET_ROOT="/mnt/afs/L202500372/data/egmap_handoff"
source scripts/env_unified.sh
AFS_HOME="/mnt/afs/L202500372"
MODEL_PATH="${AFS_HOME}/models/Qwen3.5-9B"
source scripts/formal_common.sh
formal_apply_env "${AFS_HOME}" "${MODEL_PATH}" 8001
export MASPO_WORK_MAX_TOKENS=8192

DATASETS="${DATASETS:-vqarad slake chartqa}"
SEEDS="${SEEDS:-123 42 456}"
MAX_CONCURRENT="${VQA_MAX_CONCURRENT:-2}"
LOG=logs/rerun_vqa_tok8192_master.log
mkdir -p logs result

curl -sf "http://127.0.0.1:8001/v1/models" >/dev/null || { echo "vLLM :8001 down" | tee -a "$LOG"; exit 1; }

echo "[$(date '+%F %T')] === VQA tok8192 datasets=[${DATASETS}] seeds=[${SEEDS}] ===" | tee -a "$LOG"

for seed in ${SEEDS}; do
  for ds in ${DATASETS}; do
    EG=result/egmap_formal_${ds}_llm_agg_na3_d3s200o100seed${seed}_b100k3.json
    MS=result/maspo_formal_${ds}_llm_agg_na3_d3s200o100seed${seed}.json
    cell="${ds}:seed${seed}"
    echo "[$(date '+%F %T')] --- ${cell} ---" | tee -a "$LOG"
    formal_set_vqa_mode "$ds"
    [[ -f "$EG" && ! -f "${EG}.tok4096bak" ]] && cp -f "$EG" "${EG}.tok4096bak"
    [[ -f "$MS" && ! -f "${MS}.tok4096bak" ]] && cp -f "$MS" "${MS}.tok4096bak"

    .venv/bin/python run_egmap_formal_one_seed.py \
      --dataset "$ds" --graph llm_agg --na 3 --seed "$seed" \
      --opt-size 100 --sample-size 200 --depth 3 --max-concurrent "$MAX_CONCURRENT" \
      --bank-size 100 --top-k 3 --skip-optimize >> "$LOG" 2>&1 \
      && echo "[done EGMAP] $cell" | tee -a "$LOG" \
      || { echo "[FAIL EGMAP] $cell" | tee -a "$LOG"; continue; }

    formal_set_vqa_mode "$ds"
    .venv/bin/python run_maspo_formal_baseline.py \
      --dataset "$ds" --graph llm_agg --na 3 --seed "$seed" \
      --opt-size 100 --sample-size 200 --depth 3 --max-concurrent "$MAX_CONCURRENT" \
      --bank-size 100 --top-k 3 >> "$LOG" 2>&1 \
      && echo "[done MASPO] $cell" | tee -a "$LOG" \
      || echo "[FAIL MASPO] $cell" | tee -a "$LOG"
  done
done

echo "[$(date '+%F %T')] === VQA tok8192 complete ===" | tee -a "$LOG"
.venv/bin/python scripts/export_egmap_maspo_table.py --auto -o result/comparison_table_vqa.md \
  --title "## EGMAP vs MASPO (Parallel, VQA)" >> "$LOG" 2>&1 || true
