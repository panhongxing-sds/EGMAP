#!/usr/bin/env bash
# Formal VQA phase: vqarad + slake + chartqa x 3 seeds (EGMAP + paired MASPO per cell).
set -euo pipefail
cd "$(dirname "$0")/.."

MASTER_LOG="${MASTER_LOG:-logs/run_formal_vqa_pipeline_master.log}"
mkdir -p logs

{
  echo "[$(date '+%F %T')] === formal VQA pipeline start ==="
  DATASETS="vqarad slake chartqa" \
  MAX_CONCURRENT="${VQA_MAX_CONCURRENT:-2}" \
  bash scripts/run_egmap_formal_all.sh
  echo "[$(date '+%F %T')] === formal VQA pipeline complete ==="
  .venv/bin/python scripts/export_egmap_maspo_table.py --auto -o result/comparison_table.md
  .venv/bin/python scripts/export_egmap_maspo_table.py --auto -o result/comparison_table_vqa.md \
    --title "## EGMAP vs MASPO (Parallel, VQA)"
} 2>&1 | tee -a "${MASTER_LOG}"
