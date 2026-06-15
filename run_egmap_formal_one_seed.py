#!/usr/bin/env python3
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
    verify_bank_from_opt_only,
)
from agents import MAS
from optimizers import MAPromptOptimizer
from prompts import OPTIMIZATION_REQUIREMENTS
from run_maspo import arun_test_suite
from experience import ExperienceMemoryBank, finalize_experience_bank
from formal_tags import with_model_suffix, strip_model_suffix


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="math500")
    parser.add_argument("--graph", default="llm_agg")
    parser.add_argument("--na", type=int, default=3)
    parser.add_argument("--nr", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--opt-size", type=int, default=100)
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--rounds-per-turn", type=int, default=3)
    parser.add_argument("--handoff-rounds", type=int, default=1)
    parser.add_argument("--max-concurrent", type=int, default=4)
    parser.add_argument("--bank-size", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--skip-optimize",
        action="store_true",
        help="Load saved prompt/handoff JSON and run stage1+stage2 only",
    )
    args = parser.parse_args()

    os.environ["MASPO_NA"] = str(args.na)
    os.environ["MASPO_FIXED_DEPTH"] = str(args.depth)
    os.environ["MASPO_FIXED_ROUNDS_PER_TURN"] = str(args.rounds_per_turn)
    os.environ.setdefault("V11_MIN_CONFIDENCE", "HIGH")

    dataset = args.dataset
    graph_type = GraphType(args.graph)
    task_type = get_task_type(dataset)
    use_judge = True if task_type == TaskType.CODE else get_default_use_judge(dataset)
    opt_items, eval_pool = split_opt_eval_items(dataset, args.opt_size, args.seed)
    opt_ids = {item["unique_id"] for item in opt_items}
    eval_run_items = select_eval_subset(eval_pool, args.sample_size, args.seed)
    train_problems = [item["problem"] for item in opt_items]

    base_tag = (
        f"egmap_formal_{dataset}_{graph_type.value}_na{args.na}_d{args.depth}"
        f"s{args.sample_size}o{args.opt_size}seed{args.seed}_b{args.bank_size}k{args.top_k}"
    )
    tag = with_model_suffix(base_tag)
    split_manifest_path = Path(f"splits/{strip_model_suffix(base_tag)}_split.json")
    prompt_path = Path(f"prompt/{tag}_prompts.json")
    handoff_path = Path(f"prompt/{tag}_handoffs.json")

    print(
        f"[FORMAL EGMAP] dataset={dataset} graph={graph_type.value} seed={args.seed} "
        f"model_profile={os.environ.get('FORMAL_MODEL_PROFILE', 'default')} "
        f"opt={len(opt_items)} eval_pool={len(eval_pool)} eval_run={len(eval_run_items)} "
        f"bank_size={args.bank_size} top_k={args.top_k} "
        f"skip_optimize={args.skip_optimize}",
        flush=True,
    )

    Path("prompt").mkdir(exist_ok=True)
    Path("stats").mkdir(exist_ok=True)
    Path("result").mkdir(exist_ok=True)
    Path("memory").mkdir(exist_ok=True)
    Path("splits").mkdir(exist_ok=True)

    bank_path = Path(f"memory/{tag}_bank.jsonl")
    save_formal_split_manifest(
        str(split_manifest_path),
        {
            "tag": tag,
            "dataset": dataset,
            "graph": graph_type.value,
            "seed": args.seed,
            "opt_size": args.opt_size,
            "sample_size": args.sample_size,
            "opt_unique_ids": sorted(opt_ids),
            "eval_pool_size": len(eval_pool),
            "eval_unique_ids_run": [item["unique_id"] for item in eval_run_items],
            "bank_path": str(bank_path),
            "no_eval_leakage": True,
        },
    )
    print(f"[FORMAL EGMAP] split manifest -> {split_manifest_path}", flush=True)

    if args.skip_optimize:
        if not prompt_path.exists() or not handoff_path.exists():
            raise FileNotFoundError(f"Missing cached prompts: {prompt_path} or {handoff_path}")
        prompt_map = json.loads(prompt_path.read_text(encoding="utf-8"))
        handoff_map = json.loads(handoff_path.read_text(encoding="utf-8"))
        prompt_map = {int(k): v for k, v in prompt_map.items()}
        print(f"[FORMAL EGMAP] loaded prompts from {prompt_path}", flush=True)
    else:
        use_handoff = True
        mas = MAS(
            graph_type,
            task_type,
            Nr=args.nr,
            Na=args.na,
            use_handoff=use_handoff,
            use_disagreement_handoff=True,
        )
        seed_map = {
            i: mas.agents[i].template
            for i in range(len(mas.agents))
            if mas.agents[i].type != AgentType.AGGREGATOR and mas.agents[i].template
        }
        seed_handoff_map = mas.get_handoff_map()
        image_lookup = None
        if task_type in (TaskType.VQA_OPEN, TaskType.VQA_CHOICE):
            full = load_test_data(dataset)
            image_lookup = {it["problem"]: it["image"] for it in full if it.get("image")}

        optimizer = MAPromptOptimizer(
            mas,
            train_problems,
            seed_map,
            create_evaluator_client(),
            seed_handoff_map=seed_handoff_map,
            use_handoff=use_handoff,
            use_structured_meta_prompt=True,
            image_lookup=image_lookup,
            use_disagreement_handoff=True,
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
        optimizer.best_prompt = prompt_map.copy()
        handoff_map, handoff_stats = await optimizer.optimize_all_handoffs(
            requirement=requirement,
            max_rounds=args.handoff_rounds,
        )
        prompt_path.write_text(json.dumps(prompt_map, ensure_ascii=False, indent=2), encoding="utf-8")
        handoff_path.write_text(json.dumps(handoff_map, ensure_ascii=False, indent=2), encoding="utf-8")
        Path(f"stats/{tag}_stats.json").write_text(
            json.dumps({"prompt_statistics": prompt_stats, "handoff_statistics": handoff_stats}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if bank_path.exists():
        bank_path.unlink()
    build_bank = ExperienceMemoryBank(str(bank_path), top_k=args.top_k)
    build_file = f"result/{tag}_stage1_opt_memory_build.json"
    await arun_test_suite(
        opt_items,
        task_type=task_type,
        graph_types=[graph_type],
        sample_size=None,
        seed=args.seed,
        split_info={"stage": "memory_build_from_opt_split", "seed": args.seed, "opt_size": args.opt_size, "source": "opt/train only", "no_eval_leakage": True},
        output_file=build_file,
        max_concurrent=args.max_concurrent,
        prompt_map=prompt_map,
        handoff_map=handoff_map,
        use_handoff=True,
        use_judge=use_judge,
        nr=args.nr,
        use_disagreement_handoff=True,
        use_residual_selector=True,
        experience_bank=build_bank,
        experience_top_k=args.top_k,
        write_experience=True,
    )
    kept = finalize_experience_bank(bank_path, args.bank_size)
    verify_bank_from_opt_only(str(bank_path), opt_ids)
    print(f"[FORMAL EGMAP] built bank entries={kept} -> {bank_path}", flush=True)

    eval_bank = ExperienceMemoryBank(str(bank_path), top_k=args.top_k)
    eval_file = f"result/{tag}.json"
    await arun_test_suite(
        eval_run_items,
        task_type=task_type,
        graph_types=[graph_type],
        sample_size=None,
        seed=args.seed,
        split_info={
            "stage": "frozen_bank_eval",
            "seed": args.seed,
            "opt_size": args.opt_size,
            "eval_pool_size": len(eval_pool),
            "eval_run_size": len(eval_run_items),
            "split_manifest": str(split_manifest_path),
            "bank": str(bank_path),
            "bank_size_cap": args.bank_size,
            "top_k": args.top_k,
            "write_experience": False,
            "no_eval_leakage": True,
        },
        output_file=eval_file,
        max_concurrent=args.max_concurrent,
        prompt_map=prompt_map,
        handoff_map=handoff_map,
        use_handoff=True,
        use_judge=use_judge,
        nr=args.nr,
        use_disagreement_handoff=True,
        use_residual_selector=True,
        experience_bank=eval_bank,
        experience_top_k=args.top_k,
        write_experience=False,
    )
    print(f"[FORMAL EGMAP] saved eval -> {eval_file}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
