#!/usr/bin/env bash
# GPU smoke: build experience bank on historically failed math500 seed123 opt items.
# Requires vLLM Qwen3.5-9B on :8001 (GPU). Stops other formal jobs first.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[$(date '+%F %T')] stopping competing formal/smoke jobs..."
pkill -f 'run_egmap_formal_one_seed.py' 2>/dev/null || true
pkill -f 'run_maspo_formal_baseline.py' 2>/dev/null || true
pkill -f 'smoke_bank_build.py' 2>/dev/null || true
sleep 2

export HANDOFF_DATASET_ROOT="/mnt/afs/L202500372/data/egmap_handoff"
source scripts/env_unified.sh
AFS_HOME="/mnt/afs/L202500372"
MODEL_PATH="${AFS_HOME}/models/Qwen3.5-9B"
source scripts/formal_common.sh
formal_apply_env "${AFS_HOME}" "${MODEL_PATH}" 8001
formal_apply_tok8192_env
export MASPO_NA=3
export MASPO_FIXED_DEPTH=3

if ! curl -sf "http://127.0.0.1:8001/v1/models" >/dev/null; then
  echo "Starting vLLM 9B on GPU :8001..."
  export VLLM_ENGINE_READY_TIMEOUT_S=1800
  nohup bash "${AFS_HOME}/bootstrap/serve-qwen35.sh" 9b >> "${AFS_HOME}/logs/vllm-qwen35-9b-8001-restart.log" 2>&1 &
  for i in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:8001/v1/models" >/dev/null; then
      echo "vLLM ready after ${i}0s"
      break
    fi
    sleep 10
  done
fi
if ! curl -sf "http://127.0.0.1:8001/v1/models" >/dev/null; then
  echo "ERROR: vLLM not on :8001 after wait"
  exit 1
fi

LOG=logs/smoke_bank_math500_seed123.log
echo "[$(date '+%F %T')] GPU smoke bank build @ :8001 tok8192 (exclusive)" | tee "$LOG"
echo "MODEL=${MASPO_MODEL} WORK_MAX_TOKENS=${MASPO_WORK_MAX_TOKENS}" | tee -a "$LOG"

# 3 curated items: 2 old timeouts + 1 old wrong answer
exec .venv/bin/python scripts/smoke_bank_build.py \
  --dataset math500 --seed 123 --max-concurrent 1 --fast \
  --uids \
    test/counting_and_probability/525.json \
    test/algebra/297.json \
    test/precalculus/920.json \
  2>&1 | tee -a "$LOG"
