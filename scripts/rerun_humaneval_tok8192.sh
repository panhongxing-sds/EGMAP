#!/usr/bin/env bash
# humaneval × 3 seeds @ max_tokens=8192, no prompt middle-truncation, EGMAP+MASPO fair.
set -euo pipefail
cd "$(dirname "$0")/.."

export HANDOFF_DATASET_ROOT="/mnt/afs/L202500372/data/egmap_handoff"
source scripts/env_unified.sh
AFS_HOME="/mnt/afs/L202500372"
MODEL_PATH="${AFS_HOME}/models/Qwen3.5-9B"
source scripts/formal_common.sh
formal_apply_env "${AFS_HOME}" "${MODEL_PATH}" 8001
formal_apply_tok8192_env

SEEDS="${SEEDS:-123 42 456}"
LOG=logs/rerun_humaneval_tok8192_master.log
mkdir -p logs result

echo "[$(date '+%F %T')] tok8192 env: WORK_MAX_TOKENS=${MASPO_WORK_MAX_TOKENS} WORK_MAX_PROMPT_CHARS=${MASPO_WORK_MAX_PROMPT_CHARS}" | tee -a "$LOG"
.venv/bin/python scripts/verify_humaneval_scoring.py | tee -a "$LOG"
curl -sf "http://127.0.0.1:8001/v1/models" >/dev/null || { echo "vLLM :8001 down" | tee -a "$LOG"; exit 1; }

echo "[$(date '+%F %T')] === humaneval tok8192 seeds=[${SEEDS}] ===" | tee -a "$LOG"

humaneval_cell_ok() {
  local eg="$1" ms="$2"
  [[ -s "$eg" && -s "$ms" ]] || return 1
  .venv/bin/python - <<PY
import json, sys
for p in ("$eg", "$ms"):
    d = json.load(open(p))
    gt = d["graph_types"]["llm_agg"]
    if gt.get("total", 0) <= 0:
        sys.exit(1)
    errs = sum(1 for it in d.get("detailed", []) if (it.get("models", {}).get("llm_agg") or {}).get("error"))
    if errs >= gt["total"]:
        sys.exit(1)
sys.exit(0)
PY
}

for seed in ${SEEDS}; do
  EG=result/egmap_formal_humaneval_llm_agg_na3_d3s200o100seed${seed}_b100k3.json
  MS=result/maspo_formal_humaneval_llm_agg_na3_d3s200o100seed${seed}.json
  cell="humaneval:seed${seed}"
  if humaneval_cell_ok "$EG" "$MS"; then
    echo "[$(date '+%F %T')] [skip] ${cell} (EGMAP+MASPO already valid)" | tee -a "$LOG"
    continue
  fi
  echo "[$(date '+%F %T')] --- ${cell} ---" | tee -a "$LOG"
  [[ -f "$EG" && ! -f "${EG}.pre_tok8192.bak" ]] && cp -f "$EG" "${EG}.pre_tok8192.bak"
  [[ -f "$MS" && ! -f "${MS}.pre_tok8192.bak" ]] && cp -f "$MS" "${MS}.pre_tok8192.bak"
  rm -f "result/egmap_formal_humaneval_llm_agg_na3_d3s200o100seed${seed}_b100k3_stage1_opt_memory_build.json"

  .venv/bin/python run_egmap_formal_one_seed.py \
    --dataset humaneval --graph llm_agg --na 3 --seed "$seed" \
    --opt-size 100 --sample-size 200 --depth 3 --max-concurrent 4 \
    --bank-size 100 --top-k 3 --skip-optimize >> "$LOG" 2>&1 \
    && echo "[done EGMAP] $cell" | tee -a "$LOG" \
    || { echo "[FAIL EGMAP] $cell" | tee -a "$LOG"; continue; }

  .venv/bin/python run_maspo_formal_baseline.py \
    --dataset humaneval --graph llm_agg --na 3 --seed "$seed" \
    --opt-size 100 --sample-size 200 --depth 3 --max-concurrent 4 \
    --bank-size 100 --top-k 3 >> "$LOG" 2>&1 \
    && echo "[done MASPO] $cell" | tee -a "$LOG" \
    || echo "[FAIL MASPO] $cell" | tee -a "$LOG"

  .venv/bin/python scripts/rescore_humaneval_formal.py "$EG" "$MS" --write >> "$LOG" 2>&1 || true
done

.venv/bin/python scripts/export_egmap_maspo_table.py --auto -o result/comparison_table.md >> "$LOG" 2>&1 || true
echo "[$(date '+%F %T')] === humaneval tok8192 complete ===" | tee -a "$LOG"
