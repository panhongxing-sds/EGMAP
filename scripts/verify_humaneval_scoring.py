#!/usr/bin/env python3
"""Preflight: HumanEval harness + scoring helpers must pass on gold solutions."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import DATASET_CONFIG
from data_loaders import load_test_data
from utils import execute_code_with_tests


def main() -> int:
    items = load_test_data("humaneval")
    missing = sum(1 for it in items if not it.get("test_list"))
    if missing:
        print(f"FAIL: {missing}/{len(items)} items missing test_list")
        return 1

    path = DATASET_CONFIG["humaneval"]["path"]
    raw_by_id = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            raw_by_id[row["task_id"]] = row

    failed = []
    for item in items:
        uid = item["unique_id"]
        row = raw_by_id[uid]
        gold = (row.get("prompt") or "") + (row.get("canonical_solution") or "")
        if not execute_code_with_tests(gold, item["test_list"])["success"]:
            failed.append((uid, "canonical_solution"))

    if failed:
        print(f"FAIL: {len(failed)} checks failed, e.g. {failed[:5]}")
        return 1

    print(f"OK: humaneval scoring verified on {len(items)} tasks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
