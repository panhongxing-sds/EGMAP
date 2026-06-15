#!/usr/bin/env python3
"""Re-score a saved result JSON using updated extract_answer on raw_output.

Shows before/after accuracy without re-running inference. Updates output/correct
fields and graph_types accuracy in place when --write is set.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import TaskType
from utils import extract_answer, normalize_answer, math_equivalent


def extract_answer_legacy(raw: str) -> str:
    """Pre-fix extract_answer (last-30-char fallback only after tags/boxed)."""
    raw = raw.strip()
    for _ in range(3):
        new_raw = html.unescape(raw)
        if new_raw == raw:
            break
        raw = new_raw

    matches = re.findall(r"<answer>(.*?)</answer>", raw, re.S)
    if matches:
        return matches[-1].strip()

    m = re.search(r"<answer>([^<]{0,300})$", raw.strip(), re.S)
    if m:
        val = m.group(1).strip()
        if val and val not in ("UNVERIFIED", "UNDEFINED"):
            return val

    box_pat = re.compile(r"\\boxed\s*\{((?:[^{}]|\{[^{}]*\})*)\}", re.S)
    m = box_pat.search(raw)
    if m:
        return m.group(1).strip()

    sentences = re.split(r"[。\n;]+", raw)
    last = sentences[-1].strip()
    return last[-30:] if len(last) > 30 else last


def score_sync(
    gold: str,
    output: str,
    task_type: TaskType,
) -> bool:
    correct_answer = normalize_answer(gold)
    model_answer = normalize_answer(output or "")
    is_correct = model_answer == correct_answer
    if task_type in (TaskType.MATH_CHOICE, TaskType.REASONING_CHOICE, TaskType.VQA_CHOICE):
        is_correct = is_correct or correct_answer in model_answer
    if not is_correct and task_type == TaskType.MATH:
        is_correct = math_equivalent(output, gold)
    return is_correct


def rescore_file(path: Path, write: bool = False) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    task_type = TaskType(data.get("task_type", "math"))
    graph_keys = list((data.get("graph_types") or {}).keys())
    if not graph_keys:
        graph_keys = ["llm_agg"]

    summary: Dict[str, Any] = {"path": str(path), "graphs": {}}

    for graph_key in graph_keys:
        old_correct = 0
        new_correct = 0
        total = 0
        changed = 0

        for item in data.get("detailed", []):
            model = (item.get("models") or {}).get(graph_key)
            if not model:
                continue
            raw = model.get("raw_output") or model.get("final_raw") or ""
            if not raw:
                continue

            total += 1
            gold = item.get("correct_answer") or item.get("answer") or ""
            old_out = model.get("output") or extract_answer_legacy(raw)
            new_out = extract_answer(raw)
            old_ok = score_sync(gold, old_out, task_type)
            new_ok = score_sync(gold, new_out, task_type)
            old_correct += int(old_ok)
            new_correct += int(new_ok)

            if old_out != new_out or old_ok != new_ok:
                changed += 1

            if write:
                model["output"] = new_out
                model["correct"] = new_ok
                if "rescore" not in model:
                    model["rescore"] = {}
                model["rescore"]["legacy_output"] = old_out
                model["rescore"]["legacy_correct"] = old_ok

        old_acc = old_correct / total if total else 0.0
        new_acc = new_correct / total if total else 0.0
        summary["graphs"][graph_key] = {
            "total": total,
            "old_correct": old_correct,
            "new_correct": new_correct,
            "old_accuracy": old_acc,
            "new_accuracy": new_acc,
            "changed_items": changed,
        }

        if write and graph_key in data.get("graph_types", {}):
            gt = data["graph_types"][graph_key]
            gt["correct"] = new_correct
            gt["total"] = total
            gt["accuracy"] = new_acc

    if write:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Re-score result JSON with fixed extract_answer.")
    parser.add_argument("json_path", type=Path, help="Result JSON to rescore")
    parser.add_argument("--write", action="store_true", help="Update JSON in place")
    args = parser.parse_args()

    summary = rescore_file(args.json_path, write=args.write)
    for graph_key, stats in summary["graphs"].items():
        old_pct = stats["old_accuracy"] * 100
        new_pct = stats["new_accuracy"] * 100
        delta = new_pct - old_pct
        sign = "+" if delta >= 0 else ""
        print(
            f"{args.json_path.name} [{graph_key}]: "
            f"old={old_pct:.1f}% ({stats['old_correct']}/{stats['total']}) -> "
            f"new={new_pct:.1f}% ({stats['new_correct']}/{stats['total']}) "
            f"({sign}{delta:.1f}pp, {stats['changed_items']} items changed)"
        )
    if args.write:
        print(f"Updated {args.json_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
