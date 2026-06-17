#!/usr/bin/env python3
"""Fair EGMAP vs MASPO post-processing for one dataset×seed pair.

Enforces:
  1. Both result JSONs use the same eval ``unique_id`` set (from split manifest).
  2. Deterministic re-scoring (``rescore_formal_clean``) fixes false negatives.
  3. Union of truncated/unscoreable items is removed from **both** files.

Usage:
    python scripts/fair_pair_postprocess.py --dataset math500 --seed 123 --model-suffix m4b
    python scripts/fair_pair_postprocess.py --dataset math500 --seed 123 --model-suffix m4b --write
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import DATASET_CONFIG
from scripts.prune_unscoreable_formal import (
    item_reasons,
    prune_result_file,
    recompute_graph_stats,
    union_exclude_for_pair,
)
from scripts.rescore_formal_clean import clean_correct, dataset_of, ELIGIBLE

GRAPH = "llm_agg"
NA, DEPTH, SAMPLE, OPT, BANK, TOPK = 3, 3, 200, 100, 100, 3


def _suffix(model_suffix: str) -> str:
    if not model_suffix:
        return ""
    return model_suffix if model_suffix.startswith("_") else f"_{model_suffix}"


def _pair_paths(dataset: str, seed: int, model_suffix: str) -> Tuple[Path, Path, Path]:
    suf = _suffix(model_suffix)
    eg_base = (
        f"egmap_formal_{dataset}_{GRAPH}_na{NA}_d{DEPTH}s{SAMPLE}o{OPT}seed{seed}"
        f"_b{BANK}k{TOPK}"
    )
    ms_base = f"maspo_formal_{dataset}_{GRAPH}_na{NA}_d{DEPTH}s{SAMPLE}o{OPT}seed{seed}"
    manifest = ROOT / "splits" / f"{eg_base}_split.json"
    eg = ROOT / "result" / f"{eg_base}{suf}.json"
    ms = ROOT / "result" / f"{ms_base}{suf}.json"
    return eg, ms, manifest


def _load_manifest_ids(manifest: Path) -> List[str]:
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing split manifest: {manifest}")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    ids = data.get("eval_unique_ids_run") or []
    if not ids:
        raise ValueError(f"Empty eval_unique_ids_run in {manifest}")
    return ids


def _uids_in(path: Path) -> Set[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {item["unique_id"] for item in data.get("detailed") or []}


def _sync_to_manifest(
    path: Path,
    manifest_ids: List[str],
    *,
    write: bool,
) -> Dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    graph_key = next(iter(data.get("graph_types") or {"llm_agg": {}}))
    by_uid = {item["unique_id"]: item for item in data.get("detailed") or []}
    missing = [uid for uid in manifest_ids if uid not in by_uid]
    extra = sorted(set(by_uid) - set(manifest_ids))
    kept = [by_uid[uid] for uid in manifest_ids if uid in by_uid]
    before = len(data.get("detailed") or [])
    data["detailed"] = kept
    recompute_graph_stats(data, graph_key)
    fair = data.setdefault("fair_eval", {})
    fair["eval_ids_target"] = len(manifest_ids)
    fair["eval_ids_present"] = len(kept)
    fair["missing_manifest_ids"] = missing
    fair["dropped_extra_ids"] = extra
    if write:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    gt = data["graph_types"][graph_key]
    return {
        "path": path.name,
        "before": before,
        "after": gt["total"],
        "missing": len(missing),
        "extra_dropped": len(extra),
        "missing_ids": missing[:8],
    }


def _rescore_inplace(path: Path, *, write: bool) -> Dict[str, int]:
    from data_loaders import load_test_data

    dataset = dataset_of(path)
    if dataset is None or dataset not in ELIGIBLE:
        return {"to_correct": 0, "to_wrong": 0}
    task_type = DATASET_CONFIG[dataset]["task_type"]
    gold_by_id = {it["unique_id"]: it["answer"] for it in load_test_data(dataset)}
    data = json.loads(path.read_text(encoding="utf-8"))
    graph_key = next(iter(data.get("graph_types") or {"llm_agg": {}}))
    to_correct = to_wrong = 0
    correct = 0
    for item in data.get("detailed") or []:
        uid = item["unique_id"]
        gold = gold_by_id.get(uid, item.get("correct_answer", ""))
        model = item["models"][graph_key]
        prev = bool(model.get("correct"))
        now = clean_correct(model.get("output") or "", gold, task_type)
        if now and not prev:
            to_correct += 1
        elif prev and not now:
            to_wrong += 1
        correct += int(now)
        if write:
            model["correct"] = now
    if write:
        gt = data["graph_types"][graph_key]
        total = len(data.get("detailed") or [])
        gt["correct"] = correct
        gt["total"] = total
        gt["accuracy"] = (correct / total) if total else 0.0
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"to_correct": to_correct, "to_wrong": to_wrong}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fair EGMAP/MASPO pair post-process.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model-suffix", default="m4b", help="e.g. m4b, m9b, or empty")
    parser.add_argument("--write", action="store_true")
    parser.add_argument(
        "--allow-partial-maspo",
        action="store_true",
        help="Do not abort when MASPO is missing manifest eval ids (not recommended)",
    )
    args = parser.parse_args()

    eg_path, ms_path, manifest_path = _pair_paths(args.dataset, args.seed, args.model_suffix)
    for p in (eg_path, ms_path):
        if not p.is_file():
            print(f"ERROR: missing result file: {p}", file=sys.stderr)
            sys.exit(1)

    manifest_ids = _load_manifest_ids(manifest_path)
    ms_missing = [uid for uid in manifest_ids if uid not in _uids_in(ms_path)]
    eg_missing = [uid for uid in manifest_ids if uid not in _uids_in(eg_path)]
    eg_already_pruned = False
    if ms_missing and not args.allow_partial_maspo:
        print(
            f"ERROR: MASPO missing {len(ms_missing)} manifest eval id(s). "
            f"Re-run eval first, e.g.:\n"
            f"  python run_maspo_formal_one_seed.py --dataset {args.dataset} "
            f"--seed {args.seed} --skip-optimize\n"
            f"Missing sample: {ms_missing[:5]}",
            file=sys.stderr,
        )
        sys.exit(2)
    if eg_missing:
        fair_prev = json.loads(eg_path.read_text(encoding="utf-8")).get("fair_eval") or {}
        excluded_prev = set(fair_prev.get("excluded_unscoreable") or [])
        if fair_prev.get("policy") and set(eg_missing) <= excluded_prev:
            print(
                f"  note: EGMAP already fair-pruned; {len(eg_missing)} id(s) in excluded_unscoreable"
            )
            eg_already_pruned = True
            eg_missing = []
        else:
            print(
                f"ERROR: EGMAP missing {len(eg_missing)} manifest eval id(s): {eg_missing[:5]}",
                file=sys.stderr,
            )
            sys.exit(2)

    mode = "WRITE" if args.write else "DRY-RUN"
    print(f"[{mode}] fair pair: {args.dataset} seed={args.seed} suffix={args.model_suffix or '(none)'}")
    print(f"  manifest: {manifest_path.name} ({len(manifest_ids)} eval ids)")

    for label, path in [("EGMAP", eg_path), ("MASPO", ms_path)]:
        s = _sync_to_manifest(path, manifest_ids, write=args.write)
        print(
            f"  sync {label}: {s['before']} -> {s['after']} "
            f"(missing={s['missing']}, extra_dropped={s['extra_dropped']})"
        )

    for label, path in [("EGMAP", eg_path), ("MASPO", ms_path)]:
        r = _rescore_inplace(path, write=args.write)
        print(f"  rescore {label}: +{r['to_correct']} / -{r['to_wrong']}")

    union_exclude = union_exclude_for_pair(eg_path, ms_path)
    if eg_already_pruned:
        fair_prev = json.loads(eg_path.read_text(encoding="utf-8")).get("fair_eval") or {}
        excluded_prev = fair_prev.get("excluded_unscoreable") or []
        if excluded_prev:
            union_exclude = sorted(set(union_exclude) | set(excluded_prev))
    trunc_only: List[str] = []
    dataset = args.dataset
    task_type = DATASET_CONFIG[dataset]["task_type"]
    for uid in union_exclude:
        for path in (eg_path, ms_path):
            data = json.loads(path.read_text(encoding="utf-8"))
            item = next((x for x in data["detailed"] if x["unique_id"] == uid), None)
            if item and any(
                r.startswith("truncated") for r in item_reasons(item, dataset, task_type)
            ):
                trunc_only.append(uid)
                break
    print(f"  union unscoreable: {len(union_exclude)} (truncation-related: {len(set(trunc_only))})")
    if union_exclude:
        print(f"    ids: {union_exclude}")

    for path in (eg_path, ms_path):
        summary = prune_result_file(path, union_exclude, write=args.write)
        acc = f"{summary['accuracy']*100:.1f}% ({summary['correct']}/{summary['after']})"
        side = "EGMAP" if path.name.startswith("egmap") else "MASPO"
        print(f"  prune {side}: removed {summary['removed']} -> {acc}")

    if args.write:
        for path in (eg_path, ms_path):
            data = json.loads(path.read_text(encoding="utf-8"))
            gk = next(iter(data["graph_types"]))
            gt = data["graph_types"][gk]
            fair = data.setdefault("fair_eval", {})
            fair["policy"] = "manifest_sync + rescore + union_unscoreable_prune"
            fair["manifest"] = str(manifest_path.relative_to(ROOT))
            fair["excluded_unscoreable"] = union_exclude
            fair["eval_denominator"] = gt["total"]
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

        eg = json.loads(eg_path.read_text(encoding="utf-8"))
        ms = json.loads(ms_path.read_text(encoding="utf-8"))
        gk = "llm_agg"
        e_acc = eg["graph_types"][gk]["accuracy"] * 100
        m_acc = ms["graph_types"][gk]["accuracy"] * 100
        n = eg["graph_types"][gk]["total"]
        print(
            f"\n[Fair comparison] n={n}  MASPO={m_acc:.1f}%  EGMAP={e_acc:.1f}%  "
            f"Δ={e_acc - m_acc:+.1f}pp"
        )
    else:
        print("\nRe-run with --write to persist.")


if __name__ == "__main__":
    main()
