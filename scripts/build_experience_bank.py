#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import TaskType  # noqa: E402
from experience import build_memory_entry  # noqa: E402


def infer_dataset(path: Path) -> str:
    known = [
        "math500", "math", "agieval", "agi", "aqua", "gpqa", "mbpp",
        "humaneval", "vqarad", "pmcvqa", "slake", "chartqa", "textvqa",
    ]
    stem = path.stem.lower()
    for name in known:
        if re.search(rf"(^|_){re.escape(name)}($|_)", stem):
            return name
    return stem.split("_")[0]


def iter_entries(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    task_type = TaskType(data.get("task_type", "math"))
    dataset = infer_dataset(path)
    for item in data.get("detailed", []):
        base_item = {
            "problem": item.get("problem", ""),
            "unique_id": item.get("unique_id"),
            "dataset": dataset,
        }
        for graph_type, model in (item.get("models") or {}).items():
            entry = build_memory_entry(
                base_item,
                task_type,
                str(graph_type),
                model.get("output") or "",
                model.get("raw_output") or "",
                bool(model.get("correct")),
                model.get("residual"),
            )
            if entry:
                entry["source_file"] = str(path)
                yield entry


def main():
    parser = argparse.ArgumentParser(description="Build an ExHandoff JSONL memory bank from result JSONs.")
    parser.add_argument("results", nargs="+", help="Result JSON files or glob patterns.")
    parser.add_argument("--output", default="memory/experience_bank.jsonl")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    files = []
    for pattern in args.results:
        matches = sorted(Path().glob(pattern))
        files.extend(matches or [Path(pattern)])
    files = [p for p in files if p.exists()]
    if not files:
        raise SystemExit("No result files found.")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.overwrite else "a"
    count = 0
    with out.open(mode, encoding="utf-8") as f:
        for path in files:
            for entry in iter_entries(path):
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1
    print(f"wrote {count} experience entries -> {out}")


if __name__ == "__main__":
    main()
