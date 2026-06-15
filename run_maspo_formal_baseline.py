#!/usr/bin/env python3
"""Official MASPO formal baseline — NO handoff, NO residual, NO experience.

Fair comparison protocol:
- Same disjoint opt/eval split manifest as EGMAP formal runs.
- Node prompts from ``maspo_formal_*_prompts.json`` (official MASPO optimization only).
- Evaluation = single-path ``MAS.arun()`` exactly as ``run_maspo.py`` without
  ``--experience-guided`` / ``--handoff`` / ``--disagreement-handoff`` /
  ``--residual-selector``.

Do NOT reuse EGMAP handoff maps or EGMAP co-optimized execution stack.

Usage (after ``run_maspo_formal_one_seed.py --optimize-only`` or full formal run):
    python run_maspo_formal_baseline.py \\
      --dataset math500 --graph llm_agg --na 3 --seed 123 \\
      --opt-size 100 --sample-size 200 --depth 3
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import GraphType, TaskType
from data_loaders import (
    get_default_use_judge,
    get_task_type,
    load_formal_split_manifest,
    select_eval_subset,
    split_opt_eval_items,
)
from run_maspo import arun_test_suite


def _eval_items_from_manifest_or_split(
    dataset: str,
    opt_size: int,
    sample_size: int,
    seed: int,
    manifest_path: Path,
) -> list:
    if manifest_path.is_file():
        manifest = load_formal_split_manifest(str(manifest_path))
        if manifest.get("dataset") != dataset or int(manifest.get("seed", -1)) != seed:
            raise ValueError(
                f"Split manifest mismatch: {manifest_path} "
                f"has dataset={manifest.get('dataset')} seed={manifest.get('seed')}"
            )
        run_ids = manifest.get("eval_unique_ids_run") or []
        _, eval_pool = split_opt_eval_items(dataset, opt_size, seed)
        by_id = {item["unique_id"]: item for item in eval_pool}
        missing = [uid for uid in run_ids if uid not in by_id]
        if missing:
            raise ValueError(
                f"Manifest eval ids not in recomputed eval pool: {missing[:5]} "
                f"(manifest={manifest_path})"
            )
        return [by_id[uid] for uid in run_ids]
    _, eval_pool = split_opt_eval_items(dataset, opt_size, seed)
    return select_eval_subset(eval_pool, sample_size, seed)


def _maspo_tag(dataset: str, graph: str, na: int, depth: int, sample: int, opt: int, seed: int) -> str:
    return f"maspo_formal_{dataset}_{graph}_na{na}_d{depth}s{sample}o{opt}seed{seed}"


def _egmap_tag(dataset: str, graph: str, na: int, depth: int, sample: int, opt: int, seed: int, bank: int, topk: int) -> str:
    return (
        f"egmap_formal_{dataset}_{graph}_na{na}_d{depth}s{sample}o{opt}seed{seed}"
        f"_b{bank}k{topk}"
    )


async def main():
    parser = argparse.ArgumentParser(description="Official MASPO formal baseline (frozen eval, no ExHandoff).")
    parser.add_argument("--dataset", default="math500")
    parser.add_argument("--graph", default="llm_agg")
    parser.add_argument("--na", type=int, default=3)
    parser.add_argument("--nr", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--opt-size", type=int, default=100)
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--max-concurrent", type=int, default=4)
    parser.add_argument("--bank-size", type=int, default=100, help="Match EGMAP tag for split manifest lookup only")
    parser.add_argument("--top-k", type=int, default=3, help="Match EGMAP tag for split manifest lookup only")
    args = parser.parse_args()

    os.environ["MASPO_NA"] = str(args.na)
    os.environ["MASPO_FIXED_DEPTH"] = str(args.depth)

    dataset = args.dataset
    graph_type = GraphType(args.graph)
    task_type = get_task_type(dataset)
    use_judge = True if task_type == TaskType.CODE else get_default_use_judge(dataset)

    maspo_tag = _maspo_tag(
        dataset, graph_type.value, args.na, args.depth, args.sample_size, args.opt_size, args.seed
    )
    egmap_tag = _egmap_tag(
        dataset, graph_type.value, args.na, args.depth, args.sample_size, args.opt_size,
        args.seed, args.bank_size, args.top_k,
    )
    split_manifest_path = Path(f"splits/{egmap_tag}_split.json")
    prompt_path = Path(f"prompt/{maspo_tag}_prompts.json")
    if not prompt_path.is_file():
        raise FileNotFoundError(
            f"Missing official MASPO prompts: {prompt_path}\n"
            "Run: python run_maspo_formal_one_seed.py --dataset ... --seed ...\n"
            "Do NOT point baseline at egmap_formal_* handoffs or experience-guided prompts."
        )

    eval_run_items = _eval_items_from_manifest_or_split(
        dataset, args.opt_size, args.sample_size, args.seed, split_manifest_path
    )
    prompt_map = {int(k): v for k, v in json.loads(prompt_path.read_text(encoding="utf-8")).items()}
    eval_file = f"result/{maspo_tag}.json"

    print(
        f"[OFFICIAL MASPO] dataset={dataset} graph={graph_type.value} seed={args.seed} "
        f"eval_run={len(eval_run_items)} split_manifest={split_manifest_path} "
        f"prompts={prompt_path} | handoff=False residual=False experience=False",
        flush=True,
    )

    Path("result").mkdir(exist_ok=True)
    await arun_test_suite(
        eval_run_items,
        task_type=task_type,
        graph_types=[graph_type],
        sample_size=None,
        seed=args.seed,
        split_info={
            "stage": "maspo_official_frozen_eval",
            "seed": args.seed,
            "opt_size": args.opt_size,
            "eval_run_size": len(eval_run_items),
            "split_manifest": str(split_manifest_path) if split_manifest_path.is_file() else None,
            "prompt_source": str(prompt_path),
            "handoff": False,
            "disagreement_handoff": False,
            "residual_selector": False,
            "experience": False,
            "no_eval_leakage": True,
        },
        output_file=eval_file,
        max_concurrent=args.max_concurrent,
        prompt_map=prompt_map,
        handoff_map=None,
        use_handoff=False,
        use_judge=use_judge,
        nr=args.nr,
        use_disagreement_handoff=False,
        use_residual_selector=False,
        experience_bank=None,
        write_experience=False,
    )
    print(f"[OFFICIAL MASPO] saved eval -> {eval_file}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
