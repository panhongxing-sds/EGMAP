#!/usr/bin/env bash
# m4b seed123 全链路（顺序执行，可重复跑，已完成格自动 skip）:
#   1. MASPO 文本 5 集
#   2. EGMAP 文本 5 集
#   3. 切换多模态 vLLM @ :8005
#   4. MASPO VQA 3 集
#   5. EGMAP VQA 3 集
#   → prune + RESULT 台账
set -uo pipefail
cd "$(dirname "$0")/.."
AFS="${AFS_HOME:-/mnt/afs/L202500372}"
LOG="logs/campaign_m4b_pipeline.log"
exec >>"$LOG" 2>&1

export AFS_HOME="${AFS}"

wait_for() {
  local pattern="$1"
  local label="$2"
  while pgrep -f "${pattern}" >/dev/null 2>&1; do
    echo "[wait] ${label} ..."
    sleep 30
  done
}

echo "========== m4b PIPELINE START $(date -Is) =========="

# 若已有文本阶段在跑，先等它结束（避免双开 math500 等）
wait_for 'run_m4b_text\.sh' 'text campaign'
wait_for 'run_egmap_official_phase2a\.sh' 'EGMAP phase2a'
wait_for 'run_egmap_formal_one_seed\.py' 'EGMAP job'
wait_for 'run_maspo_formal_one_seed\.py' 'MASPO job'

echo "[phase 1-2] MASPO text + EGMAP text (skip if done)"
bash scripts/run_m4b_text.sh || true

wait_for 'run_egmap_formal_one_seed\.py' 'EGMAP job'
wait_for 'run_maspo_formal_one_seed\.py' 'MASPO job'

echo "[phase 3] multimodal vLLM @ :8005"
if bash scripts/restart_vllm_4b_multimodal.sh; then
  echo "[phase 3] multimodal ready"
else
  echo "[FATAL] multimodal vLLM failed — VQA aborted" >&2
  exit 1
fi

echo "[phase 4-5] MASPO VQA + EGMAP VQA"
bash scripts/run_m4b_vqa.sh || true

PY=".venv/bin/python"
shopt -s nullglob
for f in result/maspo_formal_*_m4b.json result/egmap_formal_*_m4b.json; do
  [[ -f "$f" ]] && "$PY" scripts/prune_unscoreable_formal.py --write "$f" 2>/dev/null || true
done
shopt -u nullglob
"$PY" scripts/update_result_ledger.py --seed 123 --graph llm_agg || true

echo "========== m4b PIPELINE DONE $(date -Is) =========="
