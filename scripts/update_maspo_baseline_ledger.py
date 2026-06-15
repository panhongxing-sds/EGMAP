#!/usr/bin/env python3
"""Append official MASPO seed-123 baseline accuracies into RESULT.md ledger section."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULT = ROOT / "RESULT.md"

MARKER_START = "<!-- MASPO_BASELINE_LEDGER_START -->"
MARKER_END = "<!-- MASPO_BASELINE_LEDGER_END -->"


def collect(seed: int, graph: str = "llm_agg") -> list[tuple[str, float | None, str]]:
    rows = []
    for p in sorted(ROOT.glob("result/maspo_formal_*.json")):
        if "preaudit" in p.name or "_invalid" in str(p):
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        si = d.get("split_info") or {}
        if d.get("residual_selector") or si.get("residual_selector") or si.get("handoff"):
            continue
        if f"seed{seed}" not in p.name:
            continue
        ds = p.name.split("_")[2] if p.name.startswith("maspo_formal_") else p.stem
        # maspo_formal_{dataset}_{graph}_...
        parts = p.stem.split("_")
        try:
            idx = parts.index("formal")
            dataset = parts[idx + 1]
        except (ValueError, IndexError):
            dataset = p.stem
        acc = d.get("graph_types", {}).get(graph, {}).get("accuracy")
        status = "locked" if acc is not None else "pending"
        rows.append((dataset, acc, status))
    return rows


def render_table(rows: list[tuple[str, float | None, str]], seed: int) -> str:
    lines = [
        MARKER_START,
        f"### Phase 1 锁定 baseline（官方 MASPO，seed={seed}）",
        "",
        "> 跑完一格即写入；**锁定后不再改 protocol**，仅允许 prune 后重算 accuracy。",
        "",
        "| Dataset | Accuracy | Status | Result file |",
        "|---------|:--------:|:------:|-------------|",
    ]
    for dataset, acc, status in rows:
        acc_s = f"{acc * 100:.1f}%" if acc is not None else "—"
        tag = f"maspo_formal_{dataset}_llm_agg_na3_d3s200o100seed{seed}.json"
        lines.append(f"| {dataset} | {acc_s} | {status} | `{tag}` |")
    lines.append("")
    lines.append(MARKER_END)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()
    rows = collect(args.seed)
    if not rows:
        print("no valid official maspo results yet")
        return
    block = render_table(rows, args.seed)
    text = RESULT.read_text(encoding="utf-8") if RESULT.is_file() else ""
    if MARKER_START in text and MARKER_END in text:
        text = re.sub(
            rf"{re.escape(MARKER_START)}.*?{re.escape(MARKER_END)}",
            block,
            text,
            flags=re.DOTALL,
        )
    else:
        anchor = "## 6. 下一步（执行顺序）"
        if anchor in text:
            text = text.replace(anchor, block + "\n\n" + anchor)
        else:
            text = text.rstrip() + "\n\n" + block + "\n"
    RESULT.write_text(text, encoding="utf-8")
    print(f"updated {RESULT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
