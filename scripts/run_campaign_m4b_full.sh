#!/usr/bin/env bash
# Full Campaign 1 (Qwen3.5-4B / m4b): MASPO then EGMAP, seed=123, all 8 datasets.
# Logs: logs/campaign_m4b_full_<timestamp>.log
set -uo pipefail
cd "$(dirname "$0")/.."

STAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="logs/campaign_m4b_full_${STAMP}.log"
mkdir -p logs

exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "========== Campaign m4b full START ${STAMP} =========="
export MODEL_PROFILE=single_4b
export SEED=123
# H100 4B: eval 16-way; EGMAP residual 8-way (2 paths per sample)
export MAX_CONCURRENT="${MAX_CONCURRENT:-16}"
export EGMAP_MAX_CONCURRENT="${EGMAP_MAX_CONCURRENT:-8}"
PY="${PY:-.venv/bin/python}"

source scripts/formal_model_profiles.sh
formal_apply_model_profile single_4b
formal_apply_tok8192_env

TEXT_DS="math500 aqua gpqa agieval humaneval"
VQA_DS="vqarad slake chartqa"

_run_maspo() {
  local label="$1"
  shift
  export DATASETS="$*"
  echo ""
  echo ">>> [MASPO] ${label}: ${DATASETS}"
  bash scripts/run_maspo_official_phase1.sh || echo "[WARN] MASPO ${label} had failures"
}

_run_egmap() {
  local label="$1"
  shift
  export DATASETS="$*"
  export SKIP_PREFLIGHT=1
  export MAX_CONCURRENT="${EGMAP_MAX_CONCURRENT:-8}"
  echo ""
  echo ">>> [EGMAP] ${label}: ${DATASETS}"
  bash scripts/run_egmap_official_phase2a.sh || echo "[WARN] EGMAP ${label} had failures"
}

_wait_4b_text() {
  for i in $(seq 1 90); do
    curl -sf http://127.0.0.1:8005/v1/models >/dev/null 2>&1 && return 0
    sleep 4
  done
  echo "ERROR: 4B text vLLM not ready" >&2
  return 1
}

# --- Phase 1: Official MASPO (text; math500 auto-skipped if valid) ---
bash scripts/restart_vllm_4b_turbo.sh || {
  echo "WARN: turbo restart failed, waiting for existing :8005"
}

_run_maspo "text-remaining" ${TEXT_DS}

# --- Phase 1: MASPO VQA (multimodal 4B) ---
echo ">>> Restarting 4B multimodal turbo for VQA..."
MULTIMODAL=1 bash scripts/restart_vllm_4b_turbo.sh || {
  echo "[WARN] multimodal 4B restart failed; skipping VQA MASPO"
}
_run_maspo "vqa" ${VQA_DS}

# --- Phase 2: EGMAP (text; fresh _m4b prompts/bank/eval) ---
_wait_4b_text || bash scripts/restart_vllm_4b_multimodal.sh
# multimodal 4B can serve text too; ensure :8005 up
for i in $(seq 1 60); do
  curl -sf http://127.0.0.1:8005/v1/models >/dev/null 2>&1 && break
  sleep 4
done

_run_egmap "text-all" ${TEXT_DS}

# --- Phase 2: EGMAP VQA ---
bash scripts/restart_vllm_4b_multimodal.sh || true
_run_egmap "vqa" ${VQA_DS}

echo ""
echo ">>> Prune unscoreable + ledger sync"
shopt -s nullglob
for f in result/maspo_formal_*_m4b.json result/egmap_formal_*_m4b.json; do
  "${PY}" scripts/prune_unscoreable_formal.py --write "$f" 2>/dev/null || true
done
shopt -u nullglob
"${PY}" scripts/update_result_ledger.py --seed 123 --graph llm_agg || true

echo "========== Campaign m4b full DONE ${STAMP} =========="
