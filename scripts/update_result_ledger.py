#!/usr/bin/env python3
"""Update RESULT.md run ledger after each formal seed completes (MASPO or EGMAP)."""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULT = ROOT / "RESULT.md"

MARKER_START = "<!-- RESULT_LEDGER_START -->"
MARKER_END = "<!-- RESULT_LEDGER_END -->"

TEXT_DS = ["math500", "aqua", "gpqa", "agieval", "humaneval", "vqarad", "slake", "chartqa"]


def _is_official_maspo(data: dict) -> bool:
    si = data.get("split_info") or {}
    if si.get("handoff") is True:
        return False
    if data.get("residual_selector") or si.get("residual_selector"):
        return False
    if data.get("disagreement_handoff") or si.get("disagreement_handoff"):
        return False
    if data.get("handoff_source"):
        return False
    return True


def _model_from_name(name: str, data: dict) -> str:
    if "_m4b9b" in name:
        return "m4b9b"
    if "_m4b" in name and "_m4b9b" not in name:
        return "m4b"
    if "_m9b" in name:
        return "m9b"
    si = data.get("split_info") or {}
    return si.get("model_profile") or "legacy"


def _parse_dataset_from_name(name: str, method: str) -> str:
    parts = name.replace(".json", "").split("_")
    try:
        i = parts.index("formal")
        return parts[i + 1]
    except (ValueError, IndexError):
        return "?"


def _collect_runs(seed: int | None, graph: str) -> list[dict]:
    rows: list[dict] = []
    for p in sorted(ROOT.glob("result/*_formal_*.json")):
        if "preaudit" in p.name or "smoke" in p.name or "stage1" in p.name or "_invalid" in str(p):
            continue
        name = p.name
        if name.startswith("maspo_formal_"):
            method = "MASPO"
        elif name.startswith("egmap_formal_"):
            method = "EGMAP"
        else:
            continue
        if graph not in name:
            continue
        if seed is not None and f"seed{seed}" not in name:
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        gkey = graph if graph in (data.get("graph_types") or {}) else next(
            iter(data.get("graph_types") or {graph: {}}), graph
        )
        acc = (data.get("graph_types") or {}).get(gkey, {}).get("accuracy")
        si = data.get("split_info") or {}
        protocol_ok = True
        note = ""
        if method == "MASPO":
            protocol_ok = _is_official_maspo(data)
            if not protocol_ok:
                note = "invalid pseudo-MASPO"
        bank_n = None
        if method == "EGMAP":
            bp = si.get("bank") or ""
            bp_path = ROOT / bp if bp else None
            if bp_path and bp_path.is_file():
                bank_n = sum(1 for ln in bp_path.read_text().splitlines() if ln.strip())
            else:
                tag = p.stem
                alt = ROOT / "memory" / f"{tag}_bank.jsonl"
                if alt.is_file():
                    bank_n = sum(1 for ln in alt.read_text().splitlines() if ln.strip())
        ds = _parse_dataset_from_name(name, method)
        mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        rows.append(
            {
                "dataset": ds,
                "method": method,
                "model": _model_from_name(name, data),
                "seed": si.get("seed") or (seed if seed else "?"),
                "graph": graph,
                "acc": acc,
                "bank_n": bank_n,
                "protocol_ok": protocol_ok,
                "note": note,
                "mtime": mtime,
                "file": name,
            }
        )
    # stable sort: dataset order, then MASPO before EGMAP
    ds_order = {d: i for i, d in enumerate(TEXT_DS)}
    rows.sort(key=lambda r: (ds_order.get(r["dataset"], 99), 0 if r["method"] == "MASPO" else 1))
    return rows


def render(rows: list[dict], seed: int, graph: str) -> str:
    lines = [
        MARKER_START,
        f"### 运行台账（graph=`{graph}`，seed={seed}）",
        "",
        "> 每完成一格 formal run 自动更新。MASPO 需 `protocol_ok=yes` 才可锁定 baseline。",
        "",
        "| Dataset | Method | Model | Acc | Bank | Protocol | Updated | File |",
        "|---------|--------|-------|----:|-----:|:--------:|---------|------|",
    ]
    for r in rows:
        acc_s = f"{r['acc'] * 100:.1f}%" if r.get("acc") is not None else "—"
        bank_s = str(r["bank_n"]) if r.get("bank_n") is not None else "—"
        proto = "yes" if r.get("protocol_ok") else f"**no** {r.get('note','')}"
        lines.append(
            f"| {r['dataset']} | {r['method']} | {r.get('model','?')} | {acc_s} | {bank_s} | {proto} | {r['mtime']} | `{r['file']}` |"
        )
    if not rows:
        lines.append("| — | — | — | — | — | — | — |")
    lines.extend(["", MARKER_END])
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--graph", default="llm_agg")
    args = ap.parse_args()

    rows = _collect_runs(args.seed, args.graph)
    block = render(rows, args.seed, args.graph)
    text = RESULT.read_text(encoding="utf-8") if RESULT.is_file() else ""
    if MARKER_START in text and MARKER_END in text:
        text = re.sub(
            rf"{re.escape(MARKER_START)}.*?{re.escape(MARKER_END)}",
            block,
            text,
            flags=re.DOTALL,
        )
    else:
        # remove legacy maspo-only ledger if present
        text = re.sub(
            r"<!-- MASPO_BASELINE_LEDGER_START -->.*?<!-- MASPO_BASELINE_LEDGER_END -->\n*",
            "",
            text,
            flags=re.DOTALL,
        )
        anchor = "## 6. 下一步（执行顺序）"
        if anchor in text:
            text = text.replace(anchor, block + "\n\n" + anchor)
        else:
            text = text.rstrip() + "\n\n" + block + "\n"
    RESULT.write_text(text, encoding="utf-8")
    print(f"updated {RESULT} ({len(rows)} rows for seed={args.seed} graph={args.graph})")


if __name__ == "__main__":
    main()
