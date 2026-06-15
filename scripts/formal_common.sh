#!/usr/bin/env bash
# Shared helpers for formal EGMAP / MASPO batch scripts.
set -euo pipefail

formal_is_vqa_dataset() {
  case "${1:-}" in
    vqarad|slake|chartqa|textvqa|pmcvqa) return 0 ;;
    *) return 1 ;;
  esac
}

formal_set_vqa_mode() {
  local dataset="${1:-}"
  if formal_is_vqa_dataset "${dataset}"; then
    export V11_1_VQA_MODE="${dataset}"
    export V11_VQA_MODE="${dataset}"
  else
    unset V11_1_VQA_MODE V11_VQA_MODE
  fi
}

formal_apply_env() {
  local afs_home="${1:-/mnt/afs/L202500372}"
  local model_path="${2:-${afs_home}/models/Qwen3.5-9B}"
  local port="${3:-8001}"

  export HANDOFF_DATASET_ROOT="${afs_home}/data/egmap_handoff"
  export MASPO_MODEL="${model_path}"
  export MASPO_EVALUATOR_MODEL="${model_path}"
  export MASPO_JUDGE_MODEL="${model_path}"
  export MASPO_WORK_PORT="${port}"
  export MASPO_STRONG_PORT="${port}"
  export MASPO_BASE_URL="http://127.0.0.1:${port}/v1"
  export MASPO_EVALUATOR_BASE_URL="http://127.0.0.1:${port}/v1"
  export MASPO_JUDGE_BASE_URL="http://127.0.0.1:${port}/v1"
}

# No middle-of-prompt truncation; pair with MASPO_WORK_MAX_TOKENS=8192 for full answers.
formal_apply_tok8192_env() {
  export MASPO_WORK_MAX_TOKENS="${MASPO_WORK_MAX_TOKENS:-8192}"
  export MASPO_WORK_MAX_PROMPT_CHARS=0
  export MASPO_STRONG_MAX_PROMPT_CHARS=0
}
