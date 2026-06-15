#!/usr/bin/env python3
"""Re-score saved formal HumanEval JSONs with standard prompt+completion execution."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_loaders import load_test_data
from utils import execute_code_with_tests, humaneval_program_from_fields


def _passes(program: str, tests: list) -> bool:
    return bool(program and "def " in program and execute_code_with_tests(program, tests)["success"])


def _program_for_model(
    model: Dict[str, Any],
    prompt: str,
    tests: list,
    *,
    maspo: bool,
) -> str:
    candidates = [
        (model.get("raw_output") or model.get("final_raw") or "", model.get("output") or ""),
    ]
    if maspo:
        residual = model.get("residual") or {}
        candidates.extend(
            [
                (residual.get("challenger_raw") or "", residual.get("challenger_output") or ""),
                (residual.get("base_raw") or "", residual.get("base_output") or ""),
            ]
        )
    for raw, answer in candidates:
        program = humaneval_program_from_fields(prompt, answer, raw)
        if _passes(program, tests):
            return program
    return humaneval_program_from_fields(
        prompt,
        model.get("output") or "",
        model.get("raw_output") or model.get("final_raw") or "",
    )


def rescore_file(path: Path, tests_by_id: Dict[str, list], write: bool = False) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "stage1" in path.name:
        return {"path": str(path), "skipped": True}
    maspo = "maspo_formal" in path.name
    graph_key = next(iter(data.get("graph_types") or {"llm_agg": {}}))
    old_correct = data["graph_types"][graph_key]["correct"]
    total = 0
    new_correct = 0

    for item in data.get("detailed", []):
        uid = item["unique_id"]
        tests = tests_by_id.get(uid) or item.get("test_list") or []
        if not tests:
            continue
        model = item["models"][graph_key]
        prompt = item.get("problem") or ""
        total += 1
        program = _program_for_model(model, prompt, tests, maspo=maspo)
        ok = _passes(program, tests)
        new_correct += int(ok)
        if write:
            model["output"] = extract_completion(program, prompt) if program else model.get("output")
            model["correct"] = ok
            item["test_list"] = tests

    new_acc = new_correct / total if total else 0.0
    if write:
        gt = data["graph_types"][graph_key]
        gt["correct"] = new_correct
        gt["total"] = total
        gt["accuracy"] = new_acc
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "path": str(path),
        "old_correct": old_correct,
        "new_correct": new_correct,
        "total": total,
        "old_accuracy": old_correct / total if total else 0.0,
        "new_accuracy": new_acc,
    }


def extract_completion(program: str, prompt: str) -> str:
    prompt = (prompt or "").rstrip()
    if program.startswith(prompt):
        return program[len(prompt):].lstrip()
    return program


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", type=Path, help="Result JSON paths (default: formal humaneval eval JSONs)")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    tests_by_id = {
        item["unique_id"]: item["test_list"]
        for item in load_test_data("humaneval")
        if item.get("test_list")
    }
    if not tests_by_id:
        raise SystemExit("No HumanEval tests loaded; check data_loaders mapping.")

    paths = args.paths or sorted(
        p for p in (ROOT / "result").glob("*formal_humaneval*.json")
        if "stage1" not in p.name
    )
    for path in paths:
        if not path.is_file():
            continue
        summary = rescore_file(path, tests_by_id, write=args.write)
        if summary.get("skipped"):
            continue
        old_pct = summary["old_accuracy"] * 100
        new_pct = summary["new_accuracy"] * 100
        print(
            f"{path.name}: {old_pct:.1f}% ({summary['old_correct']}/{summary['total']}) -> "
            f"{new_pct:.1f}% ({summary['new_correct']}/{summary['total']})"
        )


if __name__ == "__main__":
    main()
