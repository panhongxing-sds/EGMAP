#!/usr/bin/env bash
# Reliable m4b campaign: wait for vLLM, then MASPO (remaining) + EGMAP.
set -uo pipefail
cd "$(dirname "$0")/.."
AFS="${AFS_HOME:-/mnt/afs/L202500372}"
LOG="logs/campaign_m4b_now.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== campaign_m4b_now $(date -Is) ==="

wait_vllm() {
  local port="${1:-8005}"
  for i in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      echo "[vLLM] ready @ :${port} (${i} checks)"
      return 0
    fi
    sleep 5
  done
  return 1
}

start_vllm_4b() {
  if wait_vllm 8005; then return 0; fi
  echo "[vLLM] starting standard 4B @ :8005 ..."
  pkill -f 'vllm.*8005' 2>/dev/null || true
  pkill -f 'nvcc.*flashinfer' 2>/dev/null || true
  sleep 3
  nohup bash "${AFS}/bootstrap/serve-qwen35.sh" 4b --port 8005 \
    >> "${AFS}/logs/vllm-Qwen3.5-4B-8005.log" 2>&1 &
  wait_vllm 8005
}

start_vllm_4b || { echo "FATAL: vLLM failed"; exit 1; }

export MODEL_PROFILE=single_4b
export SEED=123
export MAX_CONCURRENT="${MAX_CONCURRENT:-16}"
export EGMAP_MAX_CONCURRENT="${EGMAP_MAX_CONCURRENT:-8}"
export SKIP_PREFLIGHT=1
PY=".venv/bin/python"

echo "[phase] MASPO remaining text+VQA (skip math500/aqua if done)"
export DATASETS="gpqa agieval humaneval vqarad slake chartqa"
bash scripts/run_maspo_official_phase1.sh || true

echo "[phase] EGMAP full m4b"
export DATASETS="math500 aqua gpqa agieval humaneval vqarad slake chartqa"
bash scripts/run_egmap_official_phase2a.sh || true

for f in result/maspo_formal_*_m4b.json result/egmap_formal_*_m4b.json; do
  [[ -f "$f" ]] && "$PY" scripts/prune_unscoreable_formal.py --write "$f" 2>/dev/null || true
done
"$PY" scripts/update_result_ledger.py --seed 123 --graph llm_agg || true
echo "=== campaign_m4b_now DONE $(date -Is) ==="
