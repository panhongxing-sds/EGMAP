#!/usr/bin/env python3
"""Official MASPO formal pipeline: optimize node prompts only, then frozen eval.

No handoff optimization, no disagreement handoff, no residual selector,
no experience bank. Mirrors EGMAP formal splits but uses ``maspo_formal_*`` tags.

Usage:
    python run_maspo_formal_one_seed.py --dataset math500 --seed 123
    python run_maspo_formal_one_seed.py --dataset math500 --seed 123 --skip-optimize
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

from config import AgentType, GraphType, TaskType, create_evaluator_client
from data_loaders import (
    get_default_use_judge,
    get_task_type,
    load_test_data,
    save_formal_split_manifest,
    select_eval_subset,
    split_opt_eval_items,
)
from agents import MAS
from optimizers import MAPromptOptimizer
from prompts import OPTIMIZATION_REQUIREMENTS
from run_maspo import arun_test_suite
from formal_tags import model_tag_suffix, strip_model_suffix, with_model_suffix

_VQA_DATASETS = frozenset({"vqarad", "slake", "chartqa", "textvqa", "pmcvqa"})


def _maybe_skip_vqa(dataset: str) -> None:
    if dataset in _VQA_DATASETS and os.environ.get("RUN_VQA", "0") != "1":
        print(
            f"[SKIP] VQA dataset={dataset} — text-first mode; "
            "deploy multimodal vLLM then RUN_VQA=1",
            flush=True,
        )
        raise SystemExit(0)


def _tag(dataset: str, graph: str, na: int, depth: int, sample: int, opt: int, seed: int) -> str:
    base = f"maspo_formal_{dataset}_{graph}_na{na}_d{depth}s{sample}o{opt}seed{seed}"
    return with_model_suffix(base)


def _egmap_tag(dataset: str, graph: str, na: int, depth: int, sample: int, opt: int, seed: int, bank: int, topk: int) -> str:
    base = (
        f"egmap_formal_{dataset}_{graph}_na{na}_d{depth}s{sample}o{opt}seed{seed}"
        f"_b{bank}k{topk}"
    )
    return strip_model_suffix(base)  # split manifest is model-agnostic


async def main():
    parser = argparse.ArgumentParser(description="Official MASPO formal one-seed (no ExHandoff).")
    parser.add_argument("--dataset", default="math500")
    parser.add_argument("--graph", default="llm_agg")
    parser.add_argument("--na", type=int, default=3)
    parser.add_argument("--nr", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--opt-size", type=int, default=100)
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--rounds-per-turn", type=int, default=3)
    parser.add_argument("--max-concurrent", type=int, default=4)
    parser.add_argument("--bank-size", type=int, default=100, help="For EGMAP split manifest lookup")
    parser.add_argument("--top-k", type=int, default=3, help="For EGMAP split manifest lookup")
    parser.add_argument("--skip-optimize", action="store_true")
    args = parser.parse_args()
    _maybe_skip_vqa(args.dataset)

    os.environ["MASPO_NA"] = str(args.na)
    os.environ["MASPO_FIXED_DEPTH"] = str(args.depth)
    os.environ["MASPO_FIXED_ROUNDS_PER_TURN"] = str(args.rounds_per_turn)

    dataset = args.dataset
    graph_type = GraphType(args.graph)
    task_type = get_task_type(dataset)
    use_judge = True if task_type == TaskType.CODE else get_default_use_judge(dataset)
    opt_items, eval_pool = split_opt_eval_items(dataset, args.opt_size, args.seed)
    eval_run_items = select_eval_subset(eval_pool, args.sample_size, args.seed)

    tag = _tag(dataset, graph_type.value, args.na, args.depth, args.sample_size, args.opt_size, args.seed)
    egmap_tag = _egmap_tag(
        dataset, graph_type.value, args.na, args.depth, args.sample_size, args.opt_size,
        args.seed, args.bank_size, args.top_k,
    )
    split_manifest_path = Path(f"splits/{egmap_tag}_split.json")
    prompt_path = Path(f"prompt/{tag}_prompts.json")

    print(
        f"[OFFICIAL MASPO] dataset={dataset} graph={graph_type.value} seed={args.seed} "
        f"opt={len(opt_items)} eval_run={len(eval_run_items)} skip_optimize={args.skip_optimize}",
        flush=True,
    )

    for d in ("prompt", "stats", "result", "splits"):
        Path(d).mkdir(exist_ok=True)

    if args.skip_optimize:
        if not prompt_path.is_file():
            raise FileNotFoundError(f"Missing MASPO prompts: {prompt_path}")
        prompt_map = {int(k): v for k, v in json.loads(prompt_path.read_text(encoding="utf-8")).items()}
    else:
        mas = MAS(
            graph_type, task_type, Nr=args.nr, Na=args.na,
            use_handoff=False,
            use_disagreement_handoff=False,
        )
        seed_map = {
            i: mas.agents[i].template
            for i in range(len(mas.agents))
            if mas.agents[i].type != AgentType.AGGREGATOR and mas.agents[i].template
        }
        image_lookup = None
        if task_type in (TaskType.VQA_OPEN, TaskType.VQA_CHOICE):
            full = load_test_data(dataset)
            image_lookup = {it["problem"]: it["image"] for it in full if it.get("image")}

        optimizer = MAPromptOptimizer(
            mas,
            [item["problem"] for item in opt_items],
            seed_map,
            create_evaluator_client(),
            seed_handoff_map=None,
            use_handoff=False,
            use_structured_meta_prompt=False,
            image_lookup=image_lookup,
            use_disagreement_handoff=False,
        )
        requirement = OPTIMIZATION_REQUIREMENTS.get(task_type, "")
        prompt_map, prompt_stats = await optimizer.optimize_all_fixed_rounds(
            requirement=requirement,
            max_total_depth=args.depth,
            rounds_per_turn=args.rounds_per_turn,
            beam_width=2,
            use_dynamic_switching=False,
            use_stochastic_sampling=False,
            use_beam_refresh=True,
            use_feedback=False,
            use_misleading_sampling=True,
            use_lookahead_score=True,
            lookahead_weights=(0.4, 0.4, 0.2),
        )
        prompt_path.write_text(json.dumps(prompt_map, ensure_ascii=False, indent=2), encoding="utf-8")
        Path(f"stats/{tag}_stats.json").write_text(
            json.dumps({"prompt_statistics": prompt_stats}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[OFFICIAL MASPO] saved prompts -> {prompt_path}", flush=True)

    if split_manifest_path.is_file():
        manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
        run_ids = manifest.get("eval_unique_ids_run") or []
        by_id = {item["unique_id"]: item for item in eval_pool}
        missing = [uid for uid in run_ids if uid not in by_id]
        if missing:
            raise ValueError(
                f"Split manifest lists {len(missing)} eval id(s) not in eval_pool "
                f"(seed={args.seed}, dataset={dataset}): {missing[:5]}"
            )
        eval_run_items = [by_id[uid] for uid in run_ids]
        if len(eval_run_items) != len(run_ids):
            raise ValueError(
                f"Eval manifest size mismatch: manifest={len(run_ids)} resolved={len(eval_run_items)}"
            )
        print(
            f"[OFFICIAL MASPO] eval from manifest: {len(eval_run_items)} items -> {split_manifest_path}",
            flush=True,
        )
    elif args.sample_size:
        opt_ids = {item["unique_id"] for item in opt_items}
        save_formal_split_manifest(
            str(split_manifest_path),
            {
                "tag": egmap_tag,
                "dataset": dataset,
                "graph": graph_type.value,
                "seed": args.seed,
                "opt_size": args.opt_size,
                "sample_size": args.sample_size,
                "opt_unique_ids": sorted(opt_ids),
                "eval_pool_size": len(eval_pool),
                "eval_unique_ids_run": [item["unique_id"] for item in eval_run_items],
                "no_eval_leakage": True,
            },
        )
        print(f"[OFFICIAL MASPO] wrote split manifest -> {split_manifest_path}", flush=True)

    eval_file = f"result/{tag}.json"
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
            "model_profile": os.environ.get("FORMAL_MODEL_PROFILE", ""),
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
