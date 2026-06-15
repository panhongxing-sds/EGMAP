#!/usr/bin/env python3
"""Smoke-test formal experience bank build on historically problematic opt items.

Runs stage-1-style memory build (write_experience=True, no eval) on a small,
curated subset — typically items that previously timed out or produced garbage
bank rows — then finalizes the bank and prints a quality audit.

Usage:
    source scripts/env_unified.sh
    source scripts/formal_common.sh
    formal_apply_env /mnt/afs/L202500372 /mnt/afs/L202500372/models/Qwen3.5-9B 8001
    formal_apply_tok8192_env
    export MASPO_NA=3 MASPO_FIXED_DEPTH=3

    python scripts/smoke_bank_build.py --dataset math500 --seed 123
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import GraphType, TaskType
from data_loaders import get_default_use_judge, get_task_type, split_opt_eval_items
from experience import ExperienceMemoryBank, finalize_experience_bank, _is_bankable_model_answer
from run_maspo import arun_test_suite

# math500 seed123: 67× timeout + 1× wrong (precalculus/920) in old stage1 @ tok4096.
DEFAULT_SMOKE_UIDS: Dict[str, Dict[int, List[str]]] = {
    "math500": {
        123: [
            "test/counting_and_probability/525.json",
            "test/algebra/297.json",
            "test/number_theory/1257.json",
            "test/prealgebra/1558.json",
            "test/precalculus/920.json",
            "test/algebra/1035.json",
        ],
    },
    "aqua": {
        123: ["104", "152", "164", "82"],
    },
}


def formal_tag(dataset: str, seed: int, graph: str, na: int, depth: int, sample: int, opt: int, bank: int, topk: int) -> str:
    return (
        f"egmap_formal_{dataset}_{graph}_na{na}_d{depth}s{sample}o{opt}seed{seed}"
        f"_b{bank}k{topk}"
    )


def audit_bank(path: Path, task_type: TaskType) -> Dict[str, object]:
    if not path.is_file():
        return {"exists": False, "entries": 0}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    err_types = Counter(r.get("error_type") for r in rows)
    advice_types = Counter(r.get("advice") for r in rows)
    correct_true = sum(1 for r in rows if r.get("correct") is True)
    bad_ma = 0
    for r in rows:
        err = (r.get("metadata") or {}).get("runtime_error") or None
        if not _is_bankable_model_answer(str(r.get("model_answer") or ""), task_type=task_type, error=err or None):
            bad_ma += 1
    return {
        "exists": True,
        "entries": len(rows),
        "error_types": dict(err_types),
        "unique_advice": len(advice_types),
        "correct_true": correct_true,
        "bad_model_answer": bad_ma,
        "uids": [(r.get("metadata") or {}).get("unique_id") for r in rows],
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test experience bank build.")
    parser.add_argument("--dataset", default="math500")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--graph", default="llm_agg")
    parser.add_argument("--na", type=int, default=3)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--opt-size", type=int, default=100)
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--bank-size", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-concurrent", type=int, default=1)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Single-path MAS (no residual) with MASPO_NA=1 for quicker GPU smoke",
    )
    parser.add_argument("--uids", nargs="*", help="Explicit unique_id list (opt pool)")
    parser.add_argument("--output-bank", type=Path, help="Bank JSONL path (default: memory/<tag>_smoke_bank.jsonl)")
    args = parser.parse_args()

    dataset = args.dataset
    seed = args.seed
    graph_type = GraphType(args.graph)
    task_type = get_task_type(dataset)
    use_judge = get_default_use_judge(dataset)

    tag = formal_tag(dataset, seed, graph_type.value, args.na, args.depth, args.sample_size, args.opt_size, args.bank_size, args.top_k)
    prompt_path = ROOT / "prompt" / f"{tag}_prompts.json"
    handoff_path = ROOT / "prompt" / f"{tag}_handoffs.json"
    if not prompt_path.is_file() or not handoff_path.is_file():
        raise FileNotFoundError(f"Missing formal prompts: {prompt_path} / {handoff_path}")

    prompt_map = {int(k): v for k, v in json.loads(prompt_path.read_text(encoding="utf-8")).items()}
    handoff_map = json.loads(handoff_path.read_text(encoding="utf-8"))

    opt_items, _ = split_opt_eval_items(dataset, args.opt_size, seed)
    by_id = {it["unique_id"]: it for it in opt_items}

    uid_list = args.uids or DEFAULT_SMOKE_UIDS.get(dataset, {}).get(seed)
    if not uid_list:
        raise ValueError(f"No default smoke uids for {dataset} seed{seed}; pass --uids")
    missing = [u for u in uid_list if u not in by_id]
    if missing:
        raise ValueError(f"Smoke uids not in opt pool: {missing}")

    smoke_items = [by_id[u] for u in uid_list]
    bank_path = args.output_bank or (ROOT / "memory" / f"{tag}_smoke_bank.jsonl")
    result_path = ROOT / "result" / f"{tag}_smoke_bank_build.json"
    if bank_path.exists():
        bank_path.unlink()

    print(f"[SMOKE BANK] dataset={dataset} seed={seed} items={len(smoke_items)}", flush=True)
    print(f"[SMOKE BANK] prompts={prompt_path.name} bank_out={bank_path}", flush=True)
    print(f"[SMOKE BANK] WORK_MAX_TOKENS={os.environ.get('MASPO_WORK_MAX_TOKENS','?')} "
          f"PROMPT_CHARS={os.environ.get('MASPO_WORK_MAX_PROMPT_CHARS','?')}", flush=True)

    if args.fast:
        os.environ["MASPO_NA"] = "1"
        os.environ["MASPO_FIXED_DEPTH"] = "1"
        use_residual = False
        print("[SMOKE BANK] fast mode: MASPO_NA=1 depth=1 no residual selector", flush=True)
    else:
        os.environ.setdefault("MASPO_NA", str(args.na))
        os.environ.setdefault("MASPO_FIXED_DEPTH", str(args.depth))
        use_residual = True

    build_bank = ExperienceMemoryBank(str(bank_path), top_k=args.top_k)
    await arun_test_suite(
        smoke_items,
        task_type=task_type,
        graph_types=[graph_type],
        sample_size=None,
        seed=seed,
        split_info={
            "stage": "smoke_bank_build",
            "dataset": dataset,
            "seed": seed,
            "smoke_uids": uid_list,
            "fast_mode": args.fast,
            "no_eval_leakage": True,
        },
        output_file=str(result_path),
        max_concurrent=args.max_concurrent,
        prompt_map=prompt_map,
        handoff_map=handoff_map,
        use_handoff=True,
        use_judge=use_judge,
        nr=1,
        use_disagreement_handoff=not args.fast,
        use_residual_selector=use_residual,
        experience_bank=build_bank,
        experience_top_k=args.top_k,
        write_experience=True,
    )

    raw_entries = sum(1 for _ in bank_path.open()) if bank_path.exists() else 0
    kept = finalize_experience_bank(bank_path, args.bank_size)
    audit = audit_bank(bank_path, task_type)

    run = json.loads(result_path.read_text(encoding="utf-8"))
    errors = sum(1 for it in run["detailed"] if it["models"]["llm_agg"].get("error"))
    wrong = sum(1 for it in run["detailed"] if not it["models"]["llm_agg"].get("correct"))
    exp_new = run.get("experience_stats", {}).get("new_entries", 0)

    print("\n=== SMOKE RUN SUMMARY ===")
    print(f"completed={len(run['detailed'])}/{len(smoke_items)}  errors={errors}  wrong={wrong}")
    print(f"experience_entries_appended(raw)={raw_entries}  after_finalize={kept}")
    print(f"bank_audit: {json.dumps(audit, ensure_ascii=False, indent=2)}")

    # Per-item trace
    print("\n=== PER-ITEM ===")
    for item in run["detailed"]:
        m = item["models"]["llm_agg"]
        uid = item["unique_id"]
        err = m.get("error")
        out = (m.get("output") or "")[:60]
        print(f"  {uid}: correct={m.get('correct')} error={err!r} output={out!r}")

    ok = (
        kept >= max(1, wrong // 2)
        and audit.get("correct_true", 0) == 0
        and audit.get("bad_model_answer", 0) == 0
        and errors < len(smoke_items)
    )
    print(f"\nSMOKE {'PASS' if ok else 'FAIL'}: bank has {kept} valuable failure entries "
          f"(expected roughly {wrong} wrong + some timeouts as runtime_timeout)")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
