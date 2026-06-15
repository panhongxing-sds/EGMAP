#!/usr/bin/env python3
"""Audit formal EGMAP/MASPO runs for opt/eval disjointness and bank isolation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_loaders import (
    load_formal_split_manifest,
    load_test_data,
    select_eval_subset,
    split_opt_eval_items,
    verify_bank_from_opt_only,
    verify_disjoint,
)


def _ids_from_result(path: Path) -> set[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {row["unique_id"] for row in data.get("detailed", [])}


def audit_manifest(manifest_path: Path) -> list[str]:
    errors: list[str] = []
    m = load_formal_split_manifest(str(manifest_path))
    dataset = m["dataset"]
    seed = int(m["seed"])
    opt_size = int(m["opt_size"])
    sample_size = m.get("sample_size")
    opt_items, eval_pool = split_opt_eval_items(dataset, opt_size=opt_size, seed=seed)
    opt_ids = {x["unique_id"] for x in opt_items}
    expected_eval = select_eval_subset(
        eval_pool, sample_size=sample_size, seed=seed
    )
    expected_eval_ids = [x["unique_id"] for x in expected_eval]
    if expected_eval_ids != m.get("eval_unique_ids_run"):
        errors.append(
            f"{manifest_path}: eval_unique_ids_run mismatch "
            f"(manifest {len(m.get('eval_unique_ids_run', []))} vs recomputed {len(expected_eval_ids)})"
        )
    if set(m.get("opt_unique_ids", [])) != opt_ids:
        errors.append(f"{manifest_path}: opt_unique_ids mismatch")
    bank = m.get("bank_path")
    if bank:
        try:
            verify_bank_from_opt_only(bank, opt_ids)
        except ValueError as exc:
            errors.append(str(exc))
    return errors


def audit_result_pair(egmap_json: Path, maspo_json: Path, manifest_path: Path | None) -> list[str]:
    errors: list[str] = []
    egmap_ids = _ids_from_result(egmap_json)
    maspo_ids = _ids_from_result(maspo_json)
    if egmap_ids != maspo_ids:
        errors.append(
            f"EGMAP vs MASPO eval id mismatch: "
            f"only_egmap={len(egmap_ids - maspo_ids)} only_maspo={len(maspo_ids - egmap_ids)}"
        )
    if manifest_path and manifest_path.is_file():
        m = load_formal_split_manifest(str(manifest_path))
        opt_ids = set(m.get("opt_unique_ids", []))
        leaked = egmap_ids & opt_ids
        if leaked:
            errors.append(f"eval contains {len(leaked)} opt unique_id(s): {sorted(leaked)[:5]}")
        manifest_eval = set(m.get("eval_unique_ids_run", []))
        if manifest_eval and egmap_ids != manifest_eval:
            errors.append("result detailed ids != manifest eval_unique_ids_run")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify formal split has no eval leakage.")
    parser.add_argument("--manifest", type=Path, help="splits/egmap_formal_*_split.json")
    parser.add_argument("--egmap-result", type=Path)
    parser.add_argument("--maspo-result", type=Path)
    parser.add_argument(
        "--dataset", help="Recompute split only (no result files)"
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--opt-size", type=int, default=100)
    parser.add_argument("--sample-size", type=int, default=200)
    args = parser.parse_args()

    errors: list[str] = []
    if args.dataset:
        opt_items, eval_items = split_opt_eval_items(
            args.dataset, opt_size=args.opt_size, seed=args.seed
        )
        run_items = select_eval_subset(eval_items, sample_size=args.sample_size, seed=args.seed)
        verify_disjoint(opt_items, run_items, dataset=args.dataset, seed=args.seed)
        print(
            f"OK split {args.dataset} seed={args.seed}: "
            f"opt={len(opt_items)} eval_pool={len(eval_items)} eval_run={len(run_items)}"
        )
    if args.manifest:
        errors.extend(audit_manifest(args.manifest))
    if args.egmap_result and args.maspo_result:
        errors.extend(
            audit_result_pair(args.egmap_result, args.maspo_result, args.manifest)
        )
    if errors:
        for e in errors:
            print(f"FAIL: {e}")
        return 1
    if args.manifest or (args.egmap_result and args.maspo_result):
        print("OK: no leakage detected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
