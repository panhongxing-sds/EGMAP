#!/usr/bin/env bash
# Queue: wait for text-math tok8192 → humaneval tok8192 → restart multimodal vLLM → VQA tok8192.
set -euo pipefail
cd "$(dirname "$0")/.."
LOG=logs/rerun_remaining_queue.log

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "waiting for rerun_textmath_tok8192.sh to finish..."
while pgrep -f "bash scripts/rerun_textmath_tok8192.sh" >/dev/null 2>&1; do
  sleep 60
done
# Also wait if a stray egmap/maspo from text phase is still running (not humaneval/vqa)
while pgrep -f "run_egmap_formal_one_seed.py --dataset (math500|agieval|aqua|gpqa)" >/dev/null 2>&1; do
  sleep 30
done
log "text-math phase done (or not running)"

log "=== phase 2: humaneval @ tok8192 ==="
bash scripts/rerun_humaneval_tok8192.sh

log "=== phase 3: restart vLLM multimodal for VQA ==="
bash scripts/restart_vllm_9b_multimodal.sh

log "=== phase 4: VQA @ tok8192 ==="
bash scripts/rerun_vqa_tok8192.sh

log "=== all remaining datasets complete ==="
.venv/bin/python scripts/export_egmap_maspo_table.py --auto -o result/comparison_table.md >> "$LOG" 2>&1 || true
