#!/usr/bin/env bash
# Start vLLM for formal dual profile: 4B @8005 + 9B @8004 (background).
set -euo pipefail
AFS_HOME="${AFS_HOME:-/mnt/afs/L202500372}"
BOOT="${AFS_HOME}/bootstrap/serve-qwen35.sh"
mkdir -p "${AFS_HOME}/logs"

start_one() {
  local key="$1" port="$2"
  if curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
    echo "[vLLM] ${key} already up @ :${port}"
    return 0
  fi
  echo "[vLLM] starting ${key} @ :${port} ..."
  nohup bash "${BOOT}" "${key}" --port "${port}" \
    >> "${AFS_HOME}/logs/vllm-${key}-${port}.nohup.log" 2>&1 &
  for _ in $(seq 1 90); do
    sleep 4
    if curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      echo "[vLLM] ${key} ready @ :${port}"
      return 0
    fi
  done
  echo "[vLLM] TIMEOUT waiting for ${key} @ :${port}" >&2
  return 1
}

# Free ports if occupied by wrong model (optional: user may manage manually)
start_one 4b 8005
start_one 9b 8004
echo "[vLLM] dual profile ready (4B:8005, 9B:8004)"
