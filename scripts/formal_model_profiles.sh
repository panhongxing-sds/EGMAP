#!/usr/bin/env bash
# Formal experiment model profiles (run campaigns separately).
#
#   m4b9b  — Dual (default paper): work Qwen3.5-4B @ :8005, strong/judge 9B @ :8004
#   m9b    — Single 9B ablation: work+strong+judge all Qwen3.5-9B @ :8001
#
# Usage:
#   source scripts/formal_model_profiles.sh dual_4b_9b
#   source scripts/formal_model_profiles.sh single_9b
#   formal_model_tag_suffix   # prints _m4b9b or _m9b
set -euo pipefail

AFS_HOME="${AFS_HOME:-/mnt/afs/L202500372}"

formal_model_tag_suffix() {
  case "${FORMAL_MODEL_PROFILE:-}" in
    m4b9b|dual_4b_9b|dual) echo "_m4b9b" ;;
    m4b|single_4b) echo "_m4b" ;;
    m9b|single_9b|single) echo "_m9b" ;;
    "") echo "" ;;
    *) echo "_${FORMAL_MODEL_PROFILE}" ;;
  esac
}

formal_apply_model_profile() {
  local profile="${1:-dual_4b_9b}"
  # shellcheck source=formal_common.sh
  source "$(dirname "${BASH_SOURCE[0]}")/formal_common.sh"

  case "${profile}" in
    m4b9b|dual_4b_9b|dual)
      export FORMAL_MODEL_PROFILE=m4b9b
      export FORMAL_MODEL_LABEL="Qwen3.5-4B work + 9B strong"
      formal_apply_env "${AFS_HOME}" "${AFS_HOME}/models/Qwen3.5-4B" 8005
      export MASPO_WORK_PORT=8005
      export MASPO_STRONG_PORT=8004
      export MASPO_BASE_URL="http://127.0.0.1:8005/v1"
      export MASPO_EVALUATOR_BASE_URL="http://127.0.0.1:8004/v1"
      export MASPO_JUDGE_BASE_URL="http://127.0.0.1:8004/v1"
      export MASPO_MODEL="${AFS_HOME}/models/Qwen3.5-4B"
      export MASPO_EVALUATOR_MODEL="${AFS_HOME}/models/Qwen3.5-9B"
      export MASPO_JUDGE_MODEL="${AFS_HOME}/models/Qwen3.5-9B"
      ;;
    m4b|single_4b)
      export FORMAL_MODEL_PROFILE=m4b
      export FORMAL_MODEL_LABEL="Qwen3.5-4B single"
      formal_apply_env "${AFS_HOME}" "${AFS_HOME}/models/Qwen3.5-4B" 8005
      export MASPO_WORK_PORT=8005
      export MASPO_STRONG_PORT=8005
      export MASPO_BASE_URL="http://127.0.0.1:8005/v1"
      export MASPO_EVALUATOR_BASE_URL="http://127.0.0.1:8005/v1"
      export MASPO_JUDGE_BASE_URL="http://127.0.0.1:8005/v1"
      export MASPO_MODEL="${AFS_HOME}/models/Qwen3.5-4B"
      export MASPO_EVALUATOR_MODEL="${AFS_HOME}/models/Qwen3.5-4B"
      export MASPO_JUDGE_MODEL="${AFS_HOME}/models/Qwen3.5-4B"
      ;;
    m9b|single_9b|single)
      export FORMAL_MODEL_PROFILE=m9b
      export FORMAL_MODEL_LABEL="Qwen3.5-9B single"
      formal_apply_env "${AFS_HOME}" "${AFS_HOME}/models/Qwen3.5-9B" 8001
      export MASPO_WORK_PORT=8001
      export MASPO_STRONG_PORT=8001
      export MASPO_BASE_URL="http://127.0.0.1:8001/v1"
      export MASPO_EVALUATOR_BASE_URL="http://127.0.0.1:8001/v1"
      export MASPO_JUDGE_BASE_URL="http://127.0.0.1:8001/v1"
      export MASPO_MODEL="${AFS_HOME}/models/Qwen3.5-9B"
      export MASPO_EVALUATOR_MODEL="${AFS_HOME}/models/Qwen3.5-9B"
      export MASPO_JUDGE_MODEL="${AFS_HOME}/models/Qwen3.5-9B"
      ;;
    *)
      echo "Unknown profile: ${profile} (use single_4b, dual_4b_9b, or single_9b)" >&2
      return 1
      ;;
  esac
  export FORMAL_TAG_SUFFIX
  FORMAL_TAG_SUFFIX="$(formal_model_tag_suffix)"
  echo "[MODEL] profile=${FORMAL_MODEL_PROFILE} label=${FORMAL_MODEL_LABEL}"
  echo "[MODEL] work=${MASPO_MODEL} @ ${MASPO_BASE_URL}"
  echo "[MODEL] strong=${MASPO_EVALUATOR_MODEL} @ ${MASPO_EVALUATOR_BASE_URL}"
  echo "[MODEL] tag_suffix=${FORMAL_TAG_SUFFIX}"
}

# EGMAP split manifest is model-agnostic; strip model suffix for split lookup.
formal_egmap_split_tag_base() {
  local dataset="$1" graph="$2" na="$3" depth="$4" sample="$5" opt="$6" seed="$7" bank="$8" topk="$9"
  echo "egmap_formal_${dataset}_${graph}_na${na}_d${depth}s${sample}o${opt}seed${seed}_b${bank}k${topk}"
}

formal_check_vllm_profile() {
  local ok=0
  case "${FORMAL_MODEL_PROFILE:-}" in
    m4b9b)
      curl -sf "http://127.0.0.1:8005/v1/models" >/dev/null || { echo "MISSING vLLM 4B @ :8005"; ok=1; }
      curl -sf "http://127.0.0.1:8004/v1/models" >/dev/null || { echo "MISSING vLLM 9B @ :8004"; ok=1; }
      ;;
    m4b)
      curl -sf "http://127.0.0.1:8005/v1/models" >/dev/null || { echo "MISSING vLLM 4B @ :8005"; ok=1; }
      ;;
    m9b)
      curl -sf "http://127.0.0.1:8001/v1/models" >/dev/null || { echo "MISSING vLLM 9B @ :8001"; ok=1; }
      ;;
    *)
      echo "FORMAL_MODEL_PROFILE not set" >&2
      return 1
      ;;
  esac
  return "${ok}"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  formal_apply_model_profile "${1:-single_4b}"
  formal_apply_tok8192_env
  formal_check_vllm_profile
fi
