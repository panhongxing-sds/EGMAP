#!/usr/bin/env bash
# Campaign 2 (m9b): wait for 9B vLLM, then MASPO + EGMAP on another server.
set -uo pipefail
cd "$(dirname "$0")/.."
AFS="${AFS_HOME:-/mnt/afs/L202500372}"
LOG="logs/campaign_m9b_now.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== campaign_m9b_now $(date -Is) AFS_HOME=${AFS} ==="

wait_vllm() {
  local port="${1:-8001}"
  for i in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      echo "[vLLM] ready @ :${port} (${i} checks)"
      return 0
    fi
    sleep 5
  done
  return 1
}

start_vllm_9b() {
  if wait_vllm 8001; then return 0; fi
  echo "[vLLM] starting standard 9B @ :8001 ..."
  pkill -f 'vllm.*8001' 2>/dev/null || true
  pkill -f 'nvcc.*flashinfer' 2>/dev/null || true
  sleep 3
  nohup bash "${AFS}/bootstrap/serve-qwen35.sh" 9b --port 8001 \
    >> "${AFS}/logs/vllm-Qwen3.5-9B-8001.log" 2>&1 &
  wait_vllm 8001
}

start_vllm_9b || { echo "FATAL: vLLM 9B failed"; exit 1; }

export AFS_HOME="${AFS}"
export MODEL_PROFILE=single_9b
export SEED=123
export MAX_CONCURRENT="${MAX_CONCURRENT:-12}"
export EGMAP_MAX_CONCURRENT="${EGMAP_MAX_CONCURRENT:-6}"
export SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-1}"
PY=".venv/bin/python"

echo ">>> MASPO full m9b (8 datasets)"
export DATASETS="math500 aqua gpqa agieval humaneval vqarad slake chartqa"
bash scripts/run_maspo_official_phase1.sh || true

echo ">>> EGMAP full m9b"
export MAX_CONCURRENT="${EGMAP_MAX_CONCURRENT}"
bash scripts/run_egmap_official_phase2a.sh || true

for f in result/maspo_formal_*_m9b.json result/egmap_formal_*_m9b.json; do
  [[ -f "$f" ]] && "$PY" scripts/prune_unscoreable_formal.py --write "$f" 2>/dev/null || true
done
"$PY" scripts/update_result_ledger.py --seed 123 --graph llm_agg || true
echo "=== campaign_m9b_now DONE $(date -Is) ==="
