#!/usr/bin/env python3
"""EGMAP preflight: static artifact checks + optional GPU smoke before formal runs.

Checks (no GPU):
  - egmap prompts + handoffs exist
  - handoff keys cover all MAS edges for the graph
  - split manifest opt/eval disjoint; bank uids ⊆ opt pool
  - bank JSONL schema (failure-only, no correct=True rows)
  - optional: existing result json has experience/residual/handoff trace fields

Checks (GPU, --smoke):
  - scripts/smoke_bank_build.py --fast on curated opt items

Usage:
  python scripts/preflight_egmap.py --dataset math500 --seed 123
  python scripts/preflight_egmap.py --dataset math500 --seed 123 --smoke --fast
  python scripts/preflight_egmap.py --all-text --seed 123
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if not os.environ.get("HANDOFF_DATASET_ROOT"):
    os.environ["HANDOFF_DATASET_ROOT"] = "/mnt/afs/L202500372/data/egmap_handoff"

from agent import MAS
from config import GraphType, TaskType
from data_loaders import (
    get_task_type,
    load_formal_split_manifest,
    split_opt_eval_items,
    verify_bank_from_opt_only,
)
from experience import Experience
from handoff import edge_key, normalize_handoff_map

TEXT_DATASETS = ["math500", "aqua", "gpqa", "agieval", "humaneval"]
VQA_DATASETS = ["vqarad", "slake", "chartqa"]
ALL_DATASETS = TEXT_DATASETS + VQA_DATASETS

BANK_REQUIRED = {"dataset", "task_type", "problem", "error_type", "advice", "source", "correct"}


def formal_tag(
    dataset: str,
    graph: str,
    na: int,
    depth: int,
    sample: int,
    opt: int,
    seed: int,
    bank: int,
    topk: int,
) -> str:
    if graph == "reflect":
        return (
            f"egmap_formal_{dataset}_{graph}_nr{na}_d{depth}s{sample}o{opt}seed{seed}"
            f"_b{bank}k{topk}"
        )
    return (
        f"egmap_formal_{dataset}_{graph}_na{na}_d{depth}s{sample}o{opt}seed{seed}"
        f"_b{bank}k{topk}"
    )


def maspo_tag(dataset: str, graph: str, na: int, depth: int, sample: int, opt: int, seed: int, nr: int = 1) -> str:
    if graph == "reflect":
        return f"maspo_formal_{dataset}_{graph}_nr{nr}_d{depth}s{sample}o{opt}seed{seed}"
    return f"maspo_formal_{dataset}_{graph}_na{na}_d{depth}s{sample}o{opt}seed{seed}"


def check_handoff_coverage(
    dataset: str,
    graph: str,
    handoff_map: dict[str, str],
    *,
    na: int,
    nr: int,
) -> list[str]:
    errors: list[str] = []
    task_type = get_task_type(dataset)
    g = GraphType(graph)
    mas = MAS(g, task_type, Nr=nr, Na=na, use_handoff=True)
    handoff_map = normalize_handoff_map(handoff_map)
    expected = set()
    for src, dsts in mas.edges.items():
        for dst in dsts:
            expected.add(edge_key(src, dst))
    missing = sorted(expected - set(handoff_map))
    extra = sorted(set(handoff_map) - expected)
    empty = [k for k, v in handoff_map.items() if len((v or "").strip()) < 40]
    if missing:
        errors.append(f"handoff missing edges: {missing[:6]}{'...' if len(missing) > 6 else ''}")
    if extra:
        errors.append(f"handoff extra keys (non-fatal): {extra[:4]}")
    for k in empty:
        errors.append(f"handoff contract too short: {k}")
    for k in expected & set(handoff_map):
        text = handoff_map[k]
        if "Sender rule" not in text and "sender" not in text.lower():
            errors.append(f"handoff {k}: missing sender rule marker")
        if "Receiver rule" not in text and "receiver" not in text.lower():
            errors.append(f"handoff {k}: missing receiver rule marker")
    return errors


def check_bank(path: Path, opt_ids: set[str], task_type: TaskType) -> list[str]:
    errors: list[str] = []
    if not path.is_file():
        return [f"bank missing: {path}"]
    try:
        verify_bank_from_opt_only(str(path), opt_ids)
    except ValueError as exc:
        errors.append(str(exc))
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"bank {path}:{line_no} invalid json: {exc}")
            continue
        rows.append(obj)
        missing = BANK_REQUIRED - set(obj)
        if missing:
            errors.append(f"bank {path}:{line_no} missing fields: {sorted(missing)}")
        if obj.get("correct") is True:
            errors.append(f"bank {path}:{line_no} stores correct=True (failure-only violation)")
        try:
            Experience.from_json(obj)
        except Exception as exc:
            errors.append(f"bank {path}:{line_no} Experience.from_json: {exc}")
    if path.stat().st_size > 0 and not rows:
        errors.append(f"bank {path} non-empty file but no valid rows")
    return errors


def check_result_json(path: Path, *, expect_residual: bool = True) -> list[str]:
    if not path.is_file():
        return []
    errors: list[str] = []
    data = json.loads(path.read_text(encoding="utf-8"))
    detailed = data.get("detailed") or []
    if not detailed:
        errors.append(f"{path.name}: empty detailed")
        return errors
    sample = detailed[0]
    models = sample.get("models") or {}
    m = next(iter(models.values()), {})
    if "experience" not in m and expect_residual:
        pass  # field may be stripped in export; check full_output
    fo = m.get("full_output") or {}
    if isinstance(fo, dict):
        trace = fo.get("raw_trace") or {}
        if expect_residual and not trace and "residual" not in m:
            errors.append(f"{path.name}: no raw_trace in full_output (handoff path unverified)")
    if expect_residual and "residual" not in m:
        errors.append(f"{path.name}: missing residual block in sample item")
    else:
        res = m.get("residual") or {}
        for key in ("base_output", "challenger_output", "selection"):
            if key not in res:
                errors.append(f"{path.name}: residual missing {key}")
    return errors


def run_smoke(dataset: str, seed: int, fast: bool) -> tuple[bool, str]:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "smoke_bank_build.py"),
        "--dataset",
        dataset,
        "--seed",
        str(seed),
        "--max-concurrent",
        "1",
    ]
    if fast:
        cmd.append("--fast")
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    tail = "\n".join(out.strip().splitlines()[-15:])
    return proc.returncode == 0, tail


def preflight_one(
    dataset: str,
    *,
    seed: int,
    graph: str,
    na: int,
    nr: int,
    depth: int,
    sample: int,
    opt: int,
    bank: int,
    topk: int,
    smoke: bool,
    fast_smoke: bool,
    check_eval_result: bool,
) -> dict[str, Any]:
    tag = formal_tag(dataset, graph, na, depth, sample, opt, seed, bank, topk)
    prompt_path = ROOT / "prompt" / f"{tag}_prompts.json"
    handoff_path = ROOT / "prompt" / f"{tag}_handoffs.json"
    manifest_path = ROOT / "splits" / f"{tag}_split.json"
    bank_path = ROOT / "memory" / f"{tag}_bank.jsonl"
    eval_path = ROOT / "result" / f"{tag}.json"

    errors: list[str] = []
    warnings: list[str] = []

    if not prompt_path.is_file():
        errors.append(f"missing prompts: {prompt_path}")
    if not handoff_path.is_file():
        errors.append(f"missing handoffs: {handoff_path}")

    opt_items, _ = split_opt_eval_items(dataset, opt, seed)
    opt_ids = {x["unique_id"] for x in opt_items}

    if manifest_path.is_file():
        m = load_formal_split_manifest(str(manifest_path))
        if set(m.get("opt_unique_ids", [])) != opt_ids:
            errors.append("split manifest opt_unique_ids mismatch")
        leaked = set(m.get("eval_unique_ids_run", [])) & opt_ids
        if leaked:
            errors.append(f"manifest eval ids overlap opt: {len(leaked)}")
    else:
        warnings.append(f"no split manifest yet: {manifest_path}")

    if prompt_path.is_file() and handoff_path.is_file():
        handoffs = json.loads(handoff_path.read_text(encoding="utf-8"))
        errors.extend(check_handoff_coverage(dataset, graph, handoffs, na=na, nr=nr))

    if bank_path.is_file() and bank_path.stat().st_size > 0:
        errors.extend(check_bank(bank_path, opt_ids, get_task_type(dataset)))
    elif check_eval_result and eval_path.is_file():
        warnings.append(f"eval exists but bank empty/missing: {bank_path}")

    if check_eval_result and eval_path.is_file():
        errors.extend(check_result_json(eval_path, expect_residual=True))

    smoke_ok = None
    smoke_tail = ""
    if smoke and not errors:
        smoke_ok, smoke_tail = run_smoke(dataset, seed, fast_smoke)
        if not smoke_ok:
            errors.append(f"GPU smoke failed ({'fast' if fast_smoke else 'full'})\n{smoke_tail}")

    return {
        "dataset": dataset,
        "seed": seed,
        "graph": graph,
        "tag": tag,
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "smoke_ok": smoke_ok,
        "smoke_tail": smoke_tail,
        "paths": {
            "prompts": str(prompt_path),
            "handoffs": str(handoff_path),
            "bank": str(bank_path),
            "eval": str(eval_path),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="EGMAP preflight checks")
    ap.add_argument("--dataset")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--graph", default="llm_agg")
    ap.add_argument("--na", type=int, default=3)
    ap.add_argument("--nr", type=int, default=1)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--sample-size", type=int, default=200)
    ap.add_argument("--opt-size", type=int, default=100)
    ap.add_argument("--bank-size", type=int, default=100)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--smoke", action="store_true", help="Run GPU smoke_bank_build after static checks")
    ap.add_argument("--fast", action="store_true", help="Fast smoke (na=1, no residual)")
    ap.add_argument("--all-text", action="store_true")
    ap.add_argument("--all", action="store_true", dest="all_datasets")
    ap.add_argument("--check-eval", action="store_true", help="Validate existing eval json structure")
    ap.add_argument("--json-out", type=Path, help="Write report JSON")
    args = ap.parse_args()

    datasets = []
    if args.all_datasets:
        datasets = ALL_DATASETS
    elif args.all_text:
        datasets = TEXT_DATASETS
    elif args.dataset:
        datasets = [args.dataset]
    else:
        ap.error("pass --dataset, --all-text, or --all")

    reports = []
    for ds in datasets:
        nr = args.nr if args.graph == "reflect" else 1
        na = args.na if args.graph != "reflect" else args.nr
        r = preflight_one(
            ds,
            seed=args.seed,
            graph=args.graph,
            na=na,
            nr=nr,
            depth=args.depth,
            sample=args.sample_size,
            opt=args.opt_size,
            bank=args.bank_size,
            topk=args.top_k,
            smoke=args.smoke,
            fast_smoke=args.fast,
            check_eval_result=args.check_eval,
        )
        reports.append(r)
        status = "PASS" if r["ok"] else "FAIL"
        print(f"\n[{status}] {ds} seed={args.seed} graph={args.graph}")
        for w in r["warnings"]:
            print(f"  WARN: {w}")
        for e in r["errors"]:
            print(f"  ERR: {e}")
        if r["smoke_ok"] is True:
            print("  smoke: PASS")
        elif r["smoke_ok"] is False:
            print("  smoke: FAIL")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")

    return 0 if all(r["ok"] for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
