#!/usr/bin/env bash
# 4B vLLM tuned for higher throughput on H100/MIG (text or multimodal).
set -euo pipefail
AFS_HOME="${AFS_HOME:-/mnt/afs/L202500372}"
MODEL_PATH="${AFS_HOME}/models/Qwen3.5-4B"
PORT="${PORT:-8005}"
MULTIMODAL="${MULTIMODAL:-0}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
GPU_UTIL="${GPU_UTIL:-0.92}"
LOG="${AFS_HOME}/logs/vllm-qwen35-4b-${PORT}-turbo.log"

# shellcheck source=/dev/null
source "${AFS_HOME}/bootstrap/common.sh"

VLLM_PY=""
for venv in "${CONDA_TMP}" /tmp/vllm-cu124 /tmp/vllm-cu124-build; do
  [[ -x "${venv}/bin/python3" ]] && VLLM_PY="${venv}/bin/python3" && break
done
[[ -n "${VLLM_PY}" ]] || { echo "vLLM env missing" >&2; exit 1; }

pkill -f "vllm.*serve.*${PORT}" 2>/dev/null || true
pkill -f "vllm.entrypoints.*${PORT}" 2>/dev/null || true
sleep 4

EXTRA=()
if [[ "${MULTIMODAL}" != "1" ]]; then
  EXTRA+=(--language-model-only)
fi

mkdir -p "${AFS_HOME}/logs"
echo "[$(date '+%F %T')] turbo 4B @ :${PORT} max-num-seqs=${MAX_NUM_SEQS} gpu-util=${GPU_UTIL} mm=${MULTIMODAL}" | tee -a "${LOG}"

nohup "${VLLM_PY}" -m vllm.entrypoints.cli.main serve "${MODEL_PATH}" \
  --host 0.0.0.0 --port "${PORT}" \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --reasoning-parser qwen3 \
  "${EXTRA[@]}" \
  >> "${LOG}" 2>&1 &

for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "[$(date '+%F %T')] turbo 4B ready @ :${PORT}" | tee -a "${LOG}"
    exit 0
  fi
  sleep 4
done
echo "ERROR: turbo 4B failed to start" | tee -a "${LOG}"
exit 1
