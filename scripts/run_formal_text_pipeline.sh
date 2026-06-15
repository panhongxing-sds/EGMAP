#!/usr/bin/env bash
# Text formal pipeline: EGMAP (optimize + bank + eval) then paired MASPO baseline per cell.
# Phase 1 (current): 5 text datasets x 3 seeds = 15 cells.
# Phase 2 (later):   add vqarad slake chartqa via DATASETS=... on both scripts.
set -euo pipefail
cd "$(dirname "$0")/.."

MASTER_LOG="${MASTER_LOG:-logs/run_formal_text_pipeline_master.log}"
mkdir -p logs

{
  echo "[$(date '+%F %T')] === formal text pipeline start ==="
  bash scripts/run_egmap_formal_all.sh
  echo "[$(date '+%F %T')] === EGMAP done, starting MASPO baseline ==="
  bash scripts/run_maspo_formal_all.sh
  echo "[$(date '+%F %T')] === exporting comparison table ==="
  .venv/bin/python scripts/export_egmap_maspo_table.py --auto -o result/comparison_table.md
  echo "[$(date '+%F %T')] === formal text pipeline complete ==="
} 2>&1 | tee -a "${MASTER_LOG}"
