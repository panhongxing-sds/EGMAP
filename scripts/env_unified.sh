#!/usr/bin/env bash
# Unified local-vLLM/runtime configuration shared by MASPO baselines and ExHandoff.
set -euo pipefail

unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"

# Local vLLM endpoints in these experiments expect the dummy key. Force it so
# a real external OPENAI_API_KEY from the shell does not cause 401s.
export MASPO_API_KEY="${MASPO_API_KEY:-dummy}"
export OPENAI_API_KEY="$MASPO_API_KEY"
# Prefer local AFS mirror when present; fall back to legacy cluster path.
if [[ -z "${HANDOFF_DATASET_ROOT:-}" ]]; then
  if [[ -d /mnt/afs/L202500372/data/egmap_handoff ]]; then
    export HANDOFF_DATASET_ROOT="/mnt/afs/L202500372/data/egmap_handoff"
  else
    export HANDOFF_DATASET_ROOT="/public2/TangXiaoying/agentv5/datasets"
  fi
fi

export MASPO_WORK_PORT="${MASPO_WORK_PORT:-8005}"
export MASPO_STRONG_PORT="${MASPO_STRONG_PORT:-8004}"
export MASPO_BASE_URL="${MASPO_BASE_URL:-http://127.0.0.1:${MASPO_WORK_PORT}/v1}"
export MASPO_JUDGE_BASE_URL="${MASPO_JUDGE_BASE_URL:-http://127.0.0.1:${MASPO_STRONG_PORT}/v1}"
export MASPO_EVALUATOR_BASE_URL="${MASPO_EVALUATOR_BASE_URL:-http://127.0.0.1:${MASPO_STRONG_PORT}/v1}"

export MASPO_MODEL="${MASPO_MODEL:-Qwen/Qwen3.5-4B}"
export MASPO_EVALUATOR_MODEL="${MASPO_EVALUATOR_MODEL:-Qwen/Qwen3.5-9B}"
export MASPO_JUDGE_MODEL="${MASPO_JUDGE_MODEL:-Qwen/Qwen3.5-9B}"

export MASPO_STRONG_MAX_TOKENS="${MASPO_STRONG_MAX_TOKENS:-16384}"
export MASPO_WORK_MAX_PROMPT_CHARS="${MASPO_WORK_MAX_PROMPT_CHARS:-24000}"
export MASPO_STRONG_MAX_PROMPT_CHARS="${MASPO_STRONG_MAX_PROMPT_CHARS:-8000}"
