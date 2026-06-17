#!/usr/bin/env bash
# EGMAP eval ablation: same prompts/handoff, experience bank OFF.
set -euo pipefail
cd "$(dirname "$0")/.."

source scripts/formal_model_profiles.sh
MODEL_PROFILE="${MODEL_PROFILE:-single_4b}"
formal_apply_model_profile "${MODEL_PROFILE}"
formal_apply_tok8192_env

SEED="${SEED:-123}"
DATASETS="${DATASETS:-math500 aqua gpqa}"
MAX_CONCURRENT="${MAX_CONCURRENT:-6}"
PY="${PY:-.venv/bin/python}"
LOG="logs/egmap_nobank_ablation_seed${SEED}${FORMAL_TAG_SUFFIX}.log"

mkdir -p logs result
exec >>"${LOG}" 2>&1

echo "=== EGMAP no-bank ablation $(date -Is) seed=${SEED} datasets=[${DATASETS}] ==="

if ! formal_check_vllm_profile; then
  echo "ERROR: vLLM not ready"
  exit 1
fi

for dataset in ${DATASETS}; do
  echo "[$(date '+%F %T')] run ${dataset} no-bank eval"
  "${PY}" run_egmap_formal_one_seed.py \
    --dataset "${dataset}" \
    --graph llm_agg \
    --na 3 \
    --seed "${SEED}" \
    --opt-size 100 \
    --sample-size 200 \
    --depth 3 \
    --max-concurrent "${MAX_CONCURRENT}" \
    --bank-size 100 \
    --top-k 3 \
    --skip-optimize \
    --no-bank
  out="result/egmap_formal_${dataset}_llm_agg_na3_d3s200o100seed${SEED}_b100k3${FORMAL_TAG_SUFFIX}_nobank.json"
  acc=$("${PY}" -c "import json; d=json.load(open('${out}')); print(f\"{d['graph_types']['llm_agg']['accuracy']*100:.2f}%\")" 2>/dev/null || echo "?")
  echo "[$(date '+%F %T')] done ${dataset} -> ${out} acc=${acc}"
done

echo "[compare] with-bank vs no-bank"
"${PY}" <<'PY'
import json
from pathlib import Path

seed = 123
suf = "_m4b"
for ds in ["math500", "aqua", "gpqa"]:
    wb = Path(f"result/egmap_formal_{ds}_llm_agg_na3_d3s200o100seed{seed}_b100k3{suf}.json")
    nb = Path(f"result/egmap_formal_{ds}_llm_agg_na3_d3s200o100seed{seed}_b100k3{suf}_nobank.json")
    ms = Path(f"result/maspo_formal_{ds}_llm_agg_na3_d3s200o100seed{seed}{suf}.json")
    if not nb.is_file():
        print(f"{ds}: nobank missing")
        continue
    ng = json.loads(nb.read_text())["graph_types"]["llm_agg"]
    line = f"{ds}: nobank {ng['correct']}/{ng['total']}={ng['accuracy']*100:.1f}%"
    if wb.is_file():
        wg = json.loads(wb.read_text())["graph_types"]["llm_agg"]
        line += f" | with-bank {wg['correct']}/{wg['total']}={wg['accuracy']*100:.1f}% | Δbank {((ng['accuracy']-wg['accuracy'])*100):+.1f}pp"
    if ms.is_file():
        mg = json.loads(ms.read_text())["graph_types"]["llm_agg"]
        line += f" | MASPO {mg['accuracy']*100:.1f}%"
    print(line)
PY

echo "=== no-bank ablation DONE $(date -Is) ==="
