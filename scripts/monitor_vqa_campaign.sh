#!/usr/bin/env bash
# One monitor tick: vLLM health, launch VQA pipeline if ready, log dataset progress.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=logs/vqa_monitor.log
mkdir -p logs

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

vllm_ready() { curl -sf --max-time 5 http://127.0.0.1:8005/v1/models >/dev/null 2>&1; }

log "=== monitor tick ==="

if vllm_ready; then
  if pgrep -af "vllm.*8005" | grep -q "language-model-only"; then
    log "WARN: vLLM text-only — need multimodal restart before VQA"
  else
    log "vLLM ready (multimodal) @ :8005"
  fi
elif pgrep -af "vllm.*serve.*8005" >/dev/null 2>&1; then
  log "vLLM starting… (process up, waiting for API)"
  tail -1 /mnt/afs/L202500372/logs/vllm-qwen35-4b-8005-multimodal.log 2>/dev/null | tee -a "$LOG"
else
  if [[ ! -f logs/.vllm_restart_lock ]] || [[ $(find logs/.vllm_restart_lock -mmin +20 2>/dev/null) ]]; then
    touch logs/.vllm_restart_lock
    log "vLLM down — kick restart_vllm_4b_multimodal.sh"
    nohup bash scripts/restart_vllm_4b_multimodal.sh >>logs/vllm_multimodal_restart.nohup.log 2>&1 &
  else
    log "vLLM restart already in progress (lock)"
  fi
fi

vqa_done() { grep -q "m4b VQA DONE" logs/campaign_m4b_vqa.log 2>/dev/null; }
vqa_running() { pgrep -f "run_m4b_vqa.sh|run_maspo_formal_one_seed|run_egmap_formal_one_seed" >/dev/null 2>&1; }

if vllm_ready && ! vqa_done && ! vqa_running && ! pgrep -f "run_m4b_vqa.sh" >/dev/null 2>&1; then
  need=0
  for ds in vqarad slake chartqa; do
    f="result/egmap_formal_${ds}_llm_agg_na3_d3s200o100seed123_b100k3_m4b.json"
    d="result/egmap_formal_${ds}_llm_agg_na3_d3s200o100seed123_b100k3_m4b.json"
    if [[ ! -f "$f" ]]; then need=1; fi
    if [[ -f "$f" ]]; then
      fair=$(.venv/bin/python -c "import json;print(bool(json.load(open('$f')).get('fair_eval',{}).get('policy')))" 2>/dev/null || echo False)
      [[ "$fair" == "True" ]] || need=1
    fi
  done
  if [[ "$need" == "1" ]]; then
    log "launch run_m4b_vqa.sh"
    nohup bash scripts/run_m4b_vqa.sh >>logs/campaign_m4b_vqa.nohup.log 2>&1 &
  fi
fi

pgrep -af "run_maspo_formal_one_seed|run_egmap_formal_one_seed|run_m4b_vqa" 2>/dev/null | grep -v monitor | head -2 | tee -a "$LOG" || log "  no active eval job"

for ds in vqarad slake chartqa; do
  eg="result/egmap_formal_${ds}_llm_agg_na3_d3s200o100seed123_b100k3_m4b.json"
  ms="result/maspo_formal_${ds}_llm_agg_na3_d3s200o100seed123_m4b.json"
  eg_s=—; ms_s=—; fair=N
  if [[ -f "$eg" ]]; then
    eg_s=$(.venv/bin/python -c "import json;d=json.load(open('$eg'));g=d['graph_types']['llm_agg'];print(f\"{g['correct']}/{g['total']}={g['accuracy']*100:.1f}%\")")
    [[ $(.venv/bin/python -c "import json;print(json.load(open('$eg')).get('fair_eval',{}).get('policy',''))") ]] && fair=Y
  fi
  [[ -f "$ms" ]] && ms_s=$(.venv/bin/python -c "import json;d=json.load(open('$ms'));g=d['graph_types']['llm_agg'];print(f\"{g['correct']}/{g['total']}={g['accuracy']*100:.1f}%\")")
  log "  ${ds}: EG=${eg_s} MS=${ms_s} fair=${fair}"
  for lf in logs/maspo_formal_${ds}_*m4b*.log logs/egmap_formal_${ds}_*m4b*.log; do
    [[ -f "$lf" ]] || continue
    line=$(tail -1 "$lf" | tr '\r' '\n' | tail -1)
    [[ -n "$line" ]] && log "    $(basename "$lf"): ${line:0:100}"
  done
done

log "=== tick end ==="
