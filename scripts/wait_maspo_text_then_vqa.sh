#!/usr/bin/env bash
# After text MASPO backfill finishes, run VQA formal EGMAP+MASPO (vqarad/slake/chartqa).
set -euo pipefail
cd "$(dirname "$0")/.."
chmod +x scripts/run_formal_vqa_pipeline.sh scripts/run_maspo_formal_all.sh

MASPO_PID="${1:-}"
LOG="logs/wait_maspo_text_then_vqa.log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"; }

if [[ -n "${MASPO_PID}" ]] && kill -0 "${MASPO_PID}" 2>/dev/null; then
  log "waiting for text MASPO batch pid=${MASPO_PID}"
  while kill -0 "${MASPO_PID}" 2>/dev/null; do
    sleep 120
  done
  log "text MASPO batch exited"
else
  log "no live MASPO pid (${MASPO_PID:-none}); checking pending text cells"
  pending=0
  for seed in 123 42 456; do
    for ds in math500 agieval aqua gpqa humaneval; do
      ms="result/maspo_formal_${ds}_llm_agg_na3_d3s200o100seed${seed}.json"
      eg="result/egmap_formal_${ds}_llm_agg_na3_d3s200o100seed${seed}_b100k3.json"
      if [[ -s "${eg}" && ! -s "${ms}" ]]; then
        pending=$((pending + 1))
      fi
    done
  done
  if [[ "${pending}" -gt 0 ]]; then
    log "ERROR: ${pending} text MASPO cells still missing; start run_maspo_formal_all.sh first" >&2
    exit 1
  fi
  log "all text MASPO cells present; proceed to VQA"
fi

log "starting VQA formal pipeline (vqarad slake chartqa x 3 seeds)"
bash scripts/run_formal_vqa_pipeline.sh 2>&1 | tee -a "${LOG}"
log "full pipeline done -> result/comparison_table.md"
