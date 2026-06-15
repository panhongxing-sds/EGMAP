#!/usr/bin/env python3
"""Re-score saved formal MATH/CHOICE result JSONs with the current clean grader.

Why: some result JSONs were written with a stale grader and contain false
negatives where the stored answer is mathematically correct but a different
surface form (e.g. ``0.5`` vs ``1/2``, ``C`` vs ``(c)``). This re-applies the
*current* deterministic scoring (normalize + math_equivalent for MATH,
normalize + containment for CHOICE) on the already-saved ``output`` field, so
format-only mismatches are fixed while genuine errors and truncations (output
is a mid-reasoning fragment) correctly stay wrong.

Judge-based (LLM) scoring is intentionally NOT replicated: only datasets whose
formal runs used deterministic grading (default_use_judge=False) are eligible.
HumanEval/code keep using scripts/rescore_humaneval_formal.py.

Usage:
    # Dry-run every math/choice formal eval JSON (no file changes)
    python scripts/rescore_formal_clean.py

    # Write cleaned correctness/accuracy back into the JSONs
    python scripts/rescore_formal_clean.py --write

    # Specific files
    python scripts/rescore_formal_clean.py result/egmap_formal_agieval_*.json --write
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import DATASET_CONFIG, TaskType
from data_loaders import load_test_data
from utils import normalize_answer, math_equivalent

CHOICE_TYPES = {TaskType.MATH_CHOICE, TaskType.REASONING_CHOICE, TaskType.VQA_CHOICE}

# Datasets eligible for deterministic re-scoring (no LLM judge in formal runs).
ELIGIBLE = {
    name
    for name, cfg in DATASET_CONFIG.items()
    if cfg["task_type"] in ({TaskType.MATH} | CHOICE_TYPES)
    and not cfg.get("default_use_judge", False)
}


def clean_correct(output: str, gold: str, task_type: TaskType) -> bool:
    """Deterministic clean scoring mirroring run_maspo.score_answer's non-judge path."""
    correct_answer = normalize_answer(gold)
    model_answer = normalize_answer(output or "")
    is_correct = model_answer == correct_answer
    if task_type in CHOICE_TYPES:
        is_correct = is_correct or (correct_answer and correct_answer in model_answer)
    if not is_correct and task_type == TaskType.MATH:
        is_correct = math_equivalent(output or "", gold)
    return bool(is_correct)


def dataset_of(path: Path) -> str | None:
    stem = path.stem
    for name in sorted(DATASET_CONFIG, key=len, reverse=True):
        if f"_{name}_" in stem:
            return name
    return None


def rescore_file(path: Path, write: bool) -> Dict[str, object] | None:
    dataset = dataset_of(path)
    if dataset is None or dataset not in ELIGIBLE:
        return None
    task_type = DATASET_CONFIG[dataset]["task_type"]
    gold_by_id = {it["unique_id"]: it["answer"] for it in load_test_data(dataset)}

    data = json.loads(path.read_text(encoding="utf-8"))
    graph_key = next(iter(data.get("graph_types") or {"llm_agg": {}}))
    gt = data["graph_types"].get(graph_key, {})
    old_correct = gt.get("correct", 0)

    total = 0
    new_correct = 0
    flipped_to_correct = 0
    flipped_to_wrong = 0
    for item in data.get("detailed", []):
        uid = item["unique_id"]
        gold = gold_by_id.get(uid, item.get("correct_answer", ""))
        model = item["models"][graph_key]
        out = model.get("output") or ""
        prev = bool(model.get("correct"))
        now = clean_correct(out, gold, task_type)
        total += 1
        new_correct += int(now)
        if now and not prev:
            flipped_to_correct += 1
        elif prev and not now:
            flipped_to_wrong += 1
        if write:
            model["correct"] = now

    new_acc = new_correct / total if total else 0.0
    if write:
        gt["correct"] = new_correct
        gt["total"] = total
        gt["accuracy"] = new_acc
        data["graph_types"][graph_key] = gt
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "path": path.name,
        "dataset": dataset,
        "total": total,
        "old_correct": old_correct,
        "new_correct": new_correct,
        "old_acc": (old_correct / total if total else 0.0),
        "new_acc": new_acc,
        "to_correct": flipped_to_correct,
        "to_wrong": flipped_to_wrong,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--write", action="store_true", help="Write cleaned scores back into JSONs")
    args = parser.parse_args()

    paths: List[Path] = args.paths or sorted(
        p
        for p in (ROOT / "result").glob("*_formal_*.json")
        if "stage1" not in p.name and "preaudit" not in p.name
    )

    print(f"{'file':70} {'old':>14} {'new':>14}  flips(+/-)")
    print("-" * 116)
    agg = {"to_correct": 0, "to_wrong": 0, "files": 0}
    for path in paths:
        if not path.is_file():
            continue
        summary = rescore_file(path, args.write)
        if summary is None:
            continue
        agg["to_correct"] += summary["to_correct"]
        agg["to_wrong"] += summary["to_wrong"]
        agg["files"] += 1
        old = f"{summary['old_acc']*100:.1f}% ({summary['old_correct']}/{summary['total']})"
        new = f"{summary['new_acc']*100:.1f}% ({summary['new_correct']}/{summary['total']})"
        print(f"{summary['path']:70} {old:>14} {new:>14}  +{summary['to_correct']}/-{summary['to_wrong']}")

    print("-" * 116)
    mode = "WROTE" if args.write else "DRY-RUN"
    print(f"[{mode}] files={agg['files']}  total flips: +{agg['to_correct']} to_correct, -{agg['to_wrong']} to_wrong")
    if not args.write:
        print("Re-run with --write to persist cleaned scores.")


if __name__ == "__main__":
    main()
