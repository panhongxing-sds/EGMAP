#!/usr/bin/env python3
"""Export EGMAP vs MASPO comparison table from formal result JSONs.

Reads accuracy from ``graph_types.<graph_key>.accuracy`` (e.g. ``llm_agg`` for
Parallel, ``reflect`` for Sequential nr=2).

Auto-discovery pairs ``result/egmap_formal_*.json`` with matching
``result/maspo_formal_*.json`` on dataset/graph/seed/protocol suffix.

MASPO formal baseline (fair paired comparison, same opt prompts, frozen eval):
    python run_maspo_formal_baseline.py \\
      --dataset math500 --graph llm_agg --na 3 --seed 123 \\
      --opt-size 100 --sample-size 200 --depth 3

Or reuse EGMAP prompts without experience via run_maspo.py (non-formal split):
    python run_maspo.py \\
      --dataset math500 --graph llm_agg --na 3 \\
      --prompt-file prompt/egmap_formal_math500_llm_agg_na3_d3s200o100seed123_b100k3_prompts.json \\
      --handoff-file prompt/egmap_formal_math500_llm_agg_na3_d3s200o100seed123_b100k3_handoffs.json \\
      --handoff --sample-size 200 --seed 123 --opt-size 100

Examples:
    # Auto-discover formal runs (Parallel / llm_agg)
    python scripts/export_egmap_maspo_table.py --auto

    # Explicit pair + write markdown
    python scripts/export_egmap_maspo_table.py \\
      --egmap result/egmap_formal_math500_llm_agg_na3_d3s200o100seed123_b100k3.json \\
      --maspo result/maspo_formal_math500_llm_agg_na3_d3s200o100seed123.json \\
      --output result/comparison_table.md

    # EGMAP only when MASPO formal JSON is not present yet (MASPO column left as —)
    python scripts/export_egmap_maspo_table.py --auto -o result/comparison_table.md
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Text benchmarks used for macro average (matches run_exhandoff_text_na3.sh).
TEXT_BENCHMARKS = ("math500", "agieval", "aqua", "gpqa", "humaneval")
VQA_BENCHMARKS = ("vqarad", "slake", "chartqa")
ALL_BENCHMARKS = TEXT_BENCHMARKS + VQA_BENCHMARKS

BENCHMARK_LABELS = {
    "math500": "math500",
    "agieval": "agieval",
    "aqua": "aqua",
    "gpqa": "gpqa",
    "humaneval": "humaneval",
    "vqarad": "vqarad",
    "slake": "slake",
    "chartqa": "chartqa",
    "mbpp": "mbpp",
}

# Illustrative MASPO (Parallel) values from paper — update when paper table is final.
PAPER_MASPO_PARALLEL: Dict[str, float] = {
    "math500": 0.820,
    "agieval": 0.680,
    "aqua": 0.720,
    "gpqa": 0.420,
    "humaneval": 0.780,
}

KNOWN_GRAPHS = ("llm_agg", "reflect")

FORMAL_STEM_RE = re.compile(
    r"^(?P<prefix>egmap|maspo)_formal_"
    r"(?P<rest>.+)_na(?P<na>\d+)_d(?P<depth>\d+)s(?P<sample>\d+)o(?P<opt>\d+)seed(?P<seed>\d+)"
    r"(?:_b(?P<bank>\d+)k(?P<topk>\d+))?$"
)


def split_dataset_graph(rest: str) -> Tuple[str, str]:
    for graph in KNOWN_GRAPHS:
        suffix = f"_{graph}"
        if rest.endswith(suffix):
            return rest[: -len(suffix)], graph
    raise ValueError(f"cannot parse dataset/graph from {rest!r}")


@dataclass
class FormalRun:
    path: Path
    prefix: str
    dataset: str
    graph: str
    match_key: str

    @classmethod
    def from_path(cls, path: Path) -> Optional["FormalRun"]:
        m = FORMAL_STEM_RE.match(path.stem)
        if not m:
            return None
        try:
            dataset, graph = split_dataset_graph(m.group("rest"))
        except ValueError:
            return None
        match_key = (
            f"{dataset}_{graph}_na{m.group('na')}_d{m.group('depth')}"
            f"s{m.group('sample')}o{m.group('opt')}seed{m.group('seed')}"
        )
        return cls(path=path, prefix=m.group("prefix"), dataset=dataset, graph=graph, match_key=match_key)


@dataclass
class Row:
    benchmark: str
    egmap: Optional[float]
    maspo: Optional[float]
    egmap_path: Optional[Path] = None
    maspo_path: Optional[Path] = None
    maspo_is_placeholder: bool = False

    @property
    def gap(self) -> Optional[float]:
        if self.egmap is None or self.maspo is None:
            return None
        return self.egmap - self.maspo


def load_accuracy(path: Path, graph_key: str) -> float:
    data = json.loads(path.read_text(encoding="utf-8"))
    graphs = data.get("graph_types") or {}
    if graph_key not in graphs:
        raise KeyError(f"{path}: missing graph_types.{graph_key}")
    acc = graphs[graph_key].get("accuracy")
    if acc is None:
        raise KeyError(f"{path}: graph_types.{graph_key}.accuracy missing")
    return float(acc)


def fmt_pct(value: Optional[float], placeholder: bool = False) -> str:
    if value is None:
        return "—"
    suffix = "*" if placeholder else ""
    return f"{value * 100:.1f}%{suffix}"


def fmt_gap(value: Optional[float]) -> str:
    if value is None:
        return "—"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value * 100:.1f}%"


def discover_pairs(result_dir: Path, graph_key: str) -> List[Tuple[Optional[Path], Optional[Path], str]]:
    egmap_by_key: Dict[str, Path] = {}
    maspo_by_key: Dict[str, Path] = {}
    for path in sorted(result_dir.glob("*.json")):
        if path.stem.endswith("_stage1_opt_memory_build"):
            continue
        run = FormalRun.from_path(path)
        if run is None or run.graph != graph_key:
            continue
        if run.prefix == "egmap":
            egmap_by_key[run.match_key] = path
        elif run.prefix == "maspo":
            maspo_by_key[run.match_key] = path

    keys = sorted(set(egmap_by_key) | set(maspo_by_key))
    pairs: List[Tuple[Optional[Path], Optional[Path], str]] = []
    for key in keys:
        run = FormalRun.from_path(egmap_by_key.get(key) or maspo_by_key[key])
        dataset = run.dataset if run else key.split("_")[0]
        pairs.append((egmap_by_key.get(key), maspo_by_key.get(key), dataset))
    return pairs


def build_rows(
    pairs: List[Tuple[Optional[Path], Optional[Path], str]],
    graph_key: str,
    use_paper_placeholders: bool,
    paper_maspo: Dict[str, float],
) -> List[Row]:
    rows: List[Row] = []
    for egmap_path, maspo_path, dataset in pairs:
        egmap_acc = load_accuracy(egmap_path, graph_key) if egmap_path else None
        maspo_acc = None
        maspo_placeholder = False
        if maspo_path:
            maspo_acc = load_accuracy(maspo_path, graph_key)
        elif use_paper_placeholders and dataset in paper_maspo:
            maspo_acc = paper_maspo[dataset]
            maspo_placeholder = True
        rows.append(
            Row(
                benchmark=BENCHMARK_LABELS.get(dataset, dataset),
                egmap=egmap_acc,
                maspo=maspo_acc,
                egmap_path=egmap_path,
                maspo_path=maspo_path,
                maspo_is_placeholder=maspo_placeholder,
            )
        )
    return rows


def macro_average(rows: List[Row], attr: str, benchmarks: Tuple[str, ...]) -> Optional[float]:
    values = []
    for row in rows:
        if row.benchmark not in benchmarks:
            continue
        val = getattr(row, attr)
        if val is not None and not (attr == "maspo" and row.maspo_is_placeholder):
            values.append(val)
        elif attr == "maspo" and val is not None and row.maspo_is_placeholder:
            values.append(val)
    if not values:
        return None
    return sum(values) / len(values)


def render_table(
    rows: List[Row],
    graph_key: str,
    title: str,
    notes: List[str],
) -> str:
    lines = [title, ""]
    lines.append("| Benchmark | EGMAP | MASPO | Gap |")
    lines.append("| --- | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            f"| {row.benchmark} | {fmt_pct(row.egmap)} | "
            f"{fmt_pct(row.maspo, row.maspo_is_placeholder)} | {fmt_gap(row.gap)} |"
        )

    macro_egmap = macro_average(rows, "egmap", TEXT_BENCHMARKS)
    macro_maspo_vals = []
    for row in rows:
        if row.benchmark not in TEXT_BENCHMARKS or row.maspo is None:
            continue
        macro_maspo_vals.append(row.maspo)
    macro_maspo = sum(macro_maspo_vals) / len(macro_maspo_vals) if macro_maspo_vals else None
    macro_gap = (macro_egmap - macro_maspo) if macro_egmap is not None and macro_maspo is not None else None

    if macro_egmap is not None or macro_maspo is not None:
        lines.append(
            f"| **Macro avg (text)** | {fmt_pct(macro_egmap)} | "
            f"{fmt_pct(macro_maspo)} | {fmt_gap(macro_gap)} |"
        )

    macro_vqa_eg = macro_average(rows, "egmap", VQA_BENCHMARKS)
    macro_vqa_ms_vals = [
        row.maspo for row in rows if row.benchmark in VQA_BENCHMARKS and row.maspo is not None
    ]
    macro_vqa_ms = sum(macro_vqa_ms_vals) / len(macro_vqa_ms_vals) if macro_vqa_ms_vals else None
    macro_vqa_gap = (macro_vqa_eg - macro_vqa_ms) if macro_vqa_eg is not None and macro_vqa_ms is not None else None
    if macro_vqa_eg is not None or macro_vqa_ms is not None:
        lines.append(
            f"| **Macro avg (vqa)** | {fmt_pct(macro_vqa_eg)} | "
            f"{fmt_pct(macro_vqa_ms)} | {fmt_gap(macro_vqa_gap)} |"
        )

    lines.append("")
    if notes:
        for note in notes:
            lines.append(f"> {note}")
    lines.append("")
    lines.append(f"_Graph: `{graph_key}` · extracted from `graph_types.{graph_key}.accuracy`_")
    return "\n".join(lines)


def collect_notes(rows: List[Row], use_paper_placeholders: bool) -> List[str]:
    notes: List[str] = []
    notes.append(
        "Paired comparison only: local EGMAP formal vs MASPO formal (same optimized prompts, frozen eval; seed=123, opt_size=100, sample_size=200)."
    )
    missing_maspo = [r.benchmark for r in rows if r.egmap is not None and r.maspo is None and not use_paper_placeholders]
    if missing_maspo:
        notes.append(
            "MASPO formal baseline missing (paired local run): "
            + ", ".join(missing_maspo)
            + ". Run `run_maspo_formal_baseline.py` with the same formal protocol as EGMAP."
        )
    placeholder_rows = [r.benchmark for r in rows if r.maspo_is_placeholder]
    if placeholder_rows:
        notes.append(
            "MASPO values marked * are paper placeholders (not local runs): "
            + ", ".join(placeholder_rows)
        )
    egmap_only = [r.benchmark for r in rows if r.egmap is not None and r.maspo is None]
    if egmap_only and not use_paper_placeholders:
        notes.append("Gap shown as — where MASPO baseline is unavailable.")
    return notes


def parse_pair_args(pairs: List[str]) -> List[Tuple[Path, Optional[Path]]]:
    out: List[Tuple[Path, Optional[Path]]] = []
    for spec in pairs:
        if ":" in spec:
            egmap_s, maspo_s = spec.split(":", 1)
            maspo_path = None if maspo_s in ("", "-", "none") else Path(maspo_s)
        else:
            egmap_s, maspo_s = spec, None
            maspo_path = None
        out.append((Path(egmap_s), maspo_path))
    return out


def main():
    parser = argparse.ArgumentParser(description="Export EGMAP vs MASPO markdown comparison table.")
    parser.add_argument("--result-dir", type=Path, default=ROOT / "result")
    parser.add_argument("--graph-key", default="llm_agg", help="graph_types key (llm_agg=Parallel, reflect=Sequential)")
    parser.add_argument("--auto", action="store_true", help="Auto-discover formal egmap/maspo pairs in result-dir")
    parser.add_argument("--egmap", action="append", default=[], help="EGMAP result JSON (repeatable)")
    parser.add_argument("--maspo", action="append", default=[], help="MASPO result JSON aligned with --egmap order")
    parser.add_argument("--pair", action="append", default=[], help="egmap.json:maspo.json (maspo optional)")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Write markdown here (default: stdout only)")
    parser.add_argument("--title", default=None)
    parser.add_argument(
        "--use-paper-placeholders",
        action="store_true",
        help="Fill missing MASPO cells from PAPER_MASPO_PARALLEL (marked with *)",
    )
    parser.add_argument(
        "--paper-maspo-json",
        type=Path,
        default=None,
        help="JSON dict of dataset->accuracy overrides for paper placeholders",
    )
    args = parser.parse_args()

    graph_key = args.graph_key
    topology = "Parallel" if graph_key == "llm_agg" else "Sequential"
    title = args.title or f"## EGMAP vs MASPO ({topology})"

    paper_maspo = dict(PAPER_MASPO_PARALLEL)
    if args.paper_maspo_json:
        paper_maspo.update(json.loads(args.paper_maspo_json.read_text(encoding="utf-8")))

    pair_specs: List[Tuple[Optional[Path], Optional[Path], str]] = []

    if args.auto:
        pair_specs.extend(discover_pairs(args.result_dir, graph_key))

    if args.pair:
        for egmap_path, maspo_path in parse_pair_args(args.pair):
            run = FormalRun.from_path(egmap_path)
            dataset = run.dataset if run else egmap_path.stem.split("_")[2]
            pair_specs.append((egmap_path, maspo_path, dataset))

    if args.egmap:
        maspo_list = list(args.maspo)
        for i, egmap_path in enumerate(args.egmap):
            maspo_path = Path(maspo_list[i]) if i < len(maspo_list) else None
            run = FormalRun.from_path(Path(egmap_path))
            dataset = run.dataset if run else Path(egmap_path).stem.split("_")[2]
            pair_specs.append((Path(egmap_path), maspo_path, dataset))

    if not pair_specs:
        parser.error("No inputs: use --auto, --pair, or --egmap/--maspo")

    # Deduplicate by benchmark label, prefer entries with both sides.
    merged: Dict[str, Tuple[Optional[Path], Optional[Path], str]] = {}
    for egmap_path, maspo_path, dataset in pair_specs:
        key = BENCHMARK_LABELS.get(dataset, dataset)
        prev = merged.get(key)
        if prev is None:
            merged[key] = (egmap_path, maspo_path, dataset)
        else:
            eg = prev[0] or egmap_path
            ms = prev[1] or maspo_path
            merged[key] = (eg, ms, dataset)

    ordered = []
    for bench in ALL_BENCHMARKS:
        if bench in merged:
            ordered.append(merged.pop(bench))
    ordered.extend(merged.values())

    rows = build_rows(ordered, graph_key, args.use_paper_placeholders, paper_maspo)
    notes = collect_notes(rows, args.use_paper_placeholders)
    md = render_table(rows, graph_key, title, notes)

    print(md)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(md, encoding="utf-8")
        print(f"\nWrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
