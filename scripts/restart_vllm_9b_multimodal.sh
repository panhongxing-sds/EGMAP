#!/usr/bin/env bash
# Restart Qwen3.5-9B on :8001 WITH vision (VQA). Stops text-only --language-model-only instance.
set -euo pipefail
AFS_HOME="/mnt/afs/L202500372"
MODEL_PATH="${AFS_HOME}/models/Qwen3.5-9B"
PORT=8001
LOG="${AFS_HOME}/logs/vllm-qwen35-9b-${PORT}-multimodal.log"

# shellcheck source=/dev/null
source "${AFS_HOME}/bootstrap/common.sh"

VLLM_PY=""
for venv in "${CONDA_TMP}" /tmp/vllm-cu124 /tmp/vllm-cu124-build; do
  [[ -x "${venv}/bin/python3" ]] && VLLM_PY="${venv}/bin/python3" && break
done
[[ -n "${VLLM_PY}" ]] || { echo "vLLM env missing" >&2; exit 1; }

pkill -f "vllm.*serve.*${PORT}" 2>/dev/null || true
pkill -f "vllm.entrypoints.*${PORT}" 2>/dev/null || true
sleep 5

mkdir -p "${AFS_HOME}/logs"
echo "[$(date '+%F %T')] starting multimodal 9B @ :${PORT}" | tee -a "${LOG}"
nohup "${VLLM_PY}" -m vllm.entrypoints.cli.main serve "${MODEL_PATH}" \
  --host 0.0.0.0 --port "${PORT}" \
  --tensor-parallel-size 1 --max-model-len 32768 \
  --reasoning-parser qwen3 \
  >> "${LOG}" 2>&1 &

for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "[$(date '+%F %T')] multimodal 9B ready @ :${PORT}" | tee -a "${LOG}"
    exit 0
  fi
  sleep 15
done
echo "[$(date '+%F %T')] ERROR: multimodal vLLM failed to start" | tee -a "${LOG}"
exit 1
