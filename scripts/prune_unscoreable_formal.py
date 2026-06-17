#!/usr/bin/env python3
"""Drop formal-eval items whose outputs cannot be scored reliably.

Removes rows from result JSON ``detailed`` lists (and updates ``graph_types``
accuracy) so truncated/garbage/empty/error outputs do not bias benchmarks.

By default, for each EGMAP/MASPO pair on the same dataset×seed, the *union*
of unscoreable ``unique_id``s is removed from **both** files so paired
comparisons share the same denominator.

Also prunes experience bank JSONL rows with unscoreable ``model_answer``.

Usage:
    python scripts/prune_unscoreable_formal.py              # dry-run
    python scripts/prune_unscoreable_formal.py --write
    python scripts/prune_unscoreable_formal.py --write --banks
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import DATASET_CONFIG, TaskType

CHOICE_TYPES = {TaskType.MATH_CHOICE, TaskType.REASONING_CHOICE, TaskType.VQA_CHOICE}

_COMPRESS_GARBAGE_RE = re.compile(
    r"^(?:Let's re-evaluate|Sum: \$-?\d|is such that|minant \$|ne \(which|"
    r"es on the last|scribed in|This implies her|on implies \$|ith \d+)",
    re.I,
)
_TRUNCATED_MC_RE = re.compile(
    r"(?:\(Ke$|implies \$|matches ->|T_A$|distance$|iginal Ring:|If the sum is \$V)",
    re.I,
)


def dataset_of(path: Path) -> Optional[str]:
    stem = path.stem
    for name in sorted(DATASET_CONFIG, key=len, reverse=True):
        if f"_{name}_" in stem:
            return name
    return None


def formal_pair_key(path: Path) -> Optional[Tuple[str, str, str]]:
    """Return (dataset, seed, graph_key) for formal egmap/maspo result files."""
    m = re.search(
        r"^(?:egmap|maspo)_formal_(?P<ds>.+?)_(?P<graph>llm_agg|reflect)_"
        r"na\d+_d\d+s\d+o\d+seed(?P<seed>\d+)",
        path.stem,
    )
    if not m:
        return None
    return m.group("ds"), m.group("seed"), m.group("graph")


def _has_closed_answer_tag(raw: str) -> bool:
    return bool(re.search(r"<answer>.*?</answer>", raw or "", re.I | re.S))


def _truncation_reasons(output: str, raw_output: str, task_type: TaskType) -> List[str]:
    """Detect model outputs cut off before a scoreable final answer."""
    reasons: List[str] = []
    raw = (raw_output or "").strip()
    out = (output or "").strip()
    if task_type != TaskType.MATH or not raw:
        return reasons
    has_open = bool(re.search(r"<answer>", raw, re.I))
    has_close = bool(re.search(r"</answer>", raw, re.I))
    if has_open and not has_close:
        reasons.append("truncated_answer_tag")
    if _has_closed_answer_tag(raw):
        return reasons
    if not out:
        return reasons
    if out.endswith(("\\", "=", "+", "-", "*", "/")):
        reasons.append("truncated_tail")
    if re.search(r"\\(?:sum|frac|epsilon|infty)\b", out) and (
        "\\sum" in out or out.count("(") > out.count(")")
    ):
        reasons.append("truncated_mid_formula")
    if out.count("{") > out.count("}"):
        reasons.append("unclosed_brace")
    return reasons


def unscoreable_reasons(
    output: str,
    *,
    dataset: str,
    task_type: TaskType,
    error: Optional[str] = None,
    correct: Optional[bool] = None,
    raw_output: str = "",
) -> List[str]:
    reasons: List[str] = []
    if error:
        reasons.append("runtime_error")
    if correct is None:
        reasons.append("correct_none")
    out = (output or "").strip()
    if not out:
        reasons.append("empty_output")
        return reasons

    if task_type == TaskType.CODE:
        if "def " not in out and "class " not in out:
            reasons.append("code_not_executable")
        return reasons

    if task_type in CHOICE_TYPES:
        letter = out.strip().upper()
        if len(letter) == 1 and letter.isalpha():
            return reasons
        if len(out) > 6 or _TRUNCATED_MC_RE.search(out):
            reasons.append("choice_output_not_extractable")
        return reasons

    if task_type == TaskType.MATH:
        reasons.extend(_truncation_reasons(out, raw_output, task_type))
        if "\\end{answer}" in out:
            reasons.append("malformed_answer_markup")
        if (
            out.count("(") > out.count(")")
            and not _has_closed_answer_tag(raw_output)
            and not re.match(r"^\([^)]*,\s*[^\]]*\]$", out)
        ):
            reasons.append("unbalanced_parens")
        if _COMPRESS_GARBAGE_RE.search(out):
            reasons.append("compress_garbage")
        # Mid-reasoning fragment: long-ish prose without math finality.
        if len(out) > 40 and not re.search(
            r"\\boxed\{|\$[^$]+\$|^[0-9./()-]+$|^\\begin\{",
            out,
        ):
            if re.search(r"^[A-Za-z][a-z]+ ", out):
                reasons.append("reasoning_fragment")
        return reasons

    return reasons


def item_reasons(item: Dict, dataset: str, task_type: TaskType) -> List[str]:
    graph_key = next(iter(item.get("models") or {"llm_agg": {}}))
    model = item["models"][graph_key]
    return unscoreable_reasons(
        model.get("output") or "",
        dataset=dataset,
        task_type=task_type,
        error=model.get("error"),
        correct=model.get("correct"),
        raw_output=model.get("raw_output") or "",
    )


def bank_entry_reasons(entry: Dict, dataset: str, task_type: TaskType) -> List[str]:
    return unscoreable_reasons(
        entry.get("model_answer") or "",
        dataset=dataset,
        task_type=task_type,
        error=None,
        correct=entry.get("correct"),
    )


def recompute_graph_stats(data: Dict, graph_key: str) -> None:
    st = data.setdefault("graph_types", {}).setdefault(graph_key, {})
    detailed = data.get("detailed") or []
    correct = 0
    total = 0
    rt = 0.0
    for item in detailed:
        m = item["models"][graph_key]
        total += 1
        if m.get("correct"):
            correct += 1
        rt += float(m.get("response_time") or 0.0)
    st["total"] = total
    st["correct"] = correct
    st["accuracy"] = (correct / total) if total else 0.0
    st["avg_response_time"] = (rt / total) if total else 0.0


def prune_result_file(
    path: Path,
    exclude_ids: Sequence[str],
    *,
    write: bool,
) -> Dict[str, object]:
    dataset = dataset_of(path)
    if dataset is None:
        raise ValueError(f"cannot infer dataset: {path}")
    task_type = DATASET_CONFIG[dataset]["task_type"]
    data = json.loads(path.read_text(encoding="utf-8"))
    graph_key = next(iter(data.get("graph_types") or {"llm_agg": {}}))
    exclude = set(exclude_ids)
    before = len(data.get("detailed") or [])
    kept = []
    removed = []
    for item in data.get("detailed") or []:
        uid = item["unique_id"]
        if uid in exclude:
            removed.append({"unique_id": uid, "reasons": item_reasons(item, dataset, task_type)})
            continue
        kept.append(item)
    data["detailed"] = kept
    recompute_graph_stats(data, graph_key)
    excl = data.setdefault("excluded_unscoreable", {})
    excl["ids"] = sorted(exclude)
    excl["count"] = len(exclude)
    excl["policy"] = "union_pair_or_self"
    if write:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        if path.name.startswith("egmap_formal_"):
            _update_split_manifest(path, sorted(exclude))
    gt = data["graph_types"][graph_key]
    return {
        "path": path.name,
        "dataset": dataset,
        "before": before,
        "after": gt["total"],
        "removed": len(removed),
        "accuracy": gt["accuracy"],
        "correct": gt["correct"],
        "removed_sample": removed[:5],
    }


def _update_split_manifest(result_path: Path, exclude_ids: List[str]) -> None:
    if not exclude_ids:
        return
    tag = result_path.stem
    manifest_path = ROOT / "splits" / f"{tag}_split.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    excl = set(exclude_ids)
    run_ids = manifest.get("eval_unique_ids_run") or []
    manifest["eval_unique_ids_run"] = [uid for uid in run_ids if uid not in excl]
    manifest["excluded_unscoreable"] = sorted(excl)
    manifest["eval_run_size"] = len(manifest["eval_unique_ids_run"])
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def collect_pairs(paths: Sequence[Path]) -> Dict[Tuple[str, str, str], Dict[str, Path]]:
    pairs: Dict[Tuple[str, str, str], Dict[str, Path]] = {}
    for path in paths:
        key = formal_pair_key(path)
        if key is None:
            continue
        side = "egmap" if path.name.startswith("egmap") else "maspo"
        pairs.setdefault(key, {})[side] = path
    return pairs


def union_exclude_for_pair(eg_path: Path, ms_path: Path) -> List[str]:
    dataset = dataset_of(eg_path)
    assert dataset is not None
    task_type = DATASET_CONFIG[dataset]["task_type"]
    exclude: set[str] = set()
    for path in (eg_path, ms_path):
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in data.get("detailed") or []:
            if item_reasons(item, dataset, task_type):
                exclude.add(item["unique_id"])
    return sorted(exclude)


def self_exclude(path: Path) -> List[str]:
    dataset = dataset_of(path)
    if dataset is None:
        return []
    task_type = DATASET_CONFIG[dataset]["task_type"]
    data = json.loads(path.read_text(encoding="utf-8"))
    out: List[str] = []
    for item in data.get("detailed") or []:
        if item_reasons(item, dataset, task_type):
            out.append(item["unique_id"])
    return sorted(out)


def prune_banks(write: bool) -> List[Dict[str, object]]:
    summaries: List[Dict[str, object]] = []
    for path in sorted((ROOT / "memory").glob("egmap_formal_*_bank.jsonl")):
        dataset = dataset_of(path)
        if dataset is None:
            continue
        task_type = DATASET_CONFIG[dataset]["task_type"]
        kept_lines: List[str] = []
        removed = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            if bank_entry_reasons(entry, dataset, task_type):
                removed += 1
                continue
            kept_lines.append(line)
        if write and removed:
            path.write_text("\n".join(kept_lines) + ("\n" if kept_lines else ""), encoding="utf-8")
        if removed:
            summaries.append({"path": path.name, "removed": removed, "kept": len(kept_lines)})
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Prune unscoreable formal eval items.")
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--banks", action="store_true", help="Also prune memory/*_bank.jsonl")
    parser.add_argument(
        "--no-union-pair",
        action="store_true",
        help="Prune per-file only (do not union EGMAP+MASPO exclusions)",
    )
    args = parser.parse_args()

    paths: List[Path] = args.paths or sorted(
        p
        for p in (ROOT / "result").glob("*_formal_*.json")
        if "stage1" not in p.name and "preaudit" not in p.name and "bak" not in p.name
    )

    pairs = collect_pairs(paths)
    processed: set[Path] = set()
    print(f"{'file':70} {'removed':>7} {'new_acc':>12}")
    print("-" * 95)

    for key, sides in sorted(pairs.items()):
        eg = sides.get("egmap")
        ms = sides.get("maspo")
        targets = [p for p in (eg, ms) if p is not None]
        if not targets:
            continue
        if args.no_union_pair or eg is None or ms is None:
            exclude_map = {p: self_exclude(p) for p in targets}
        else:
            union = union_exclude_for_pair(eg, ms)
            exclude_map = {p: union for p in targets}
        for path in targets:
            if path in processed:
                continue
            summary = prune_result_file(path, exclude_map[path], write=args.write)
            processed.add(path)
            acc = f"{summary['accuracy']*100:.1f}% ({summary['correct']}/{summary['after']})"
            print(f"{summary['path']:70} {summary['removed']:7d} {acc:>12}")

    for path in paths:
        if path in processed:
            continue
        summary = prune_result_file(path, self_exclude(path), write=args.write)
        acc = f"{summary['accuracy']*100:.1f}% ({summary['correct']}/{summary['after']})"
        print(f"{summary['path']:70} {summary['removed']:7d} {acc:>12}")

    if args.banks:
        print("\nExperience banks:")
        for s in prune_banks(args.write):
            print(f"  {s['path']}: removed {s['removed']}, kept {s['kept']}")

    mode = "WROTE" if args.write else "DRY-RUN"
    print(f"\n[{mode}] processed {len(processed)} result file(s)")
    if not args.write:
        print("Re-run with --write to persist.")


if __name__ == "__main__":
    main()
