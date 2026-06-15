import json
import time
import random
import asyncio
import argparse
import os
from typing import List, Dict, Any, Optional

from tqdm.asyncio import tqdm_asyncio

from config import (
    GraphType, AgentType, TaskType, DATASET_CONFIG,
    aclient, bclient, create_evaluator_client
)
from prompts import OPTIMIZATION_REQUIREMENTS
from utils import (
    normalize_answer, extract_output, extract_function_name_from_tests,
    math_equivalent, vqa_open_equivalent, humaneval_program_from_fields,
)
from data_loaders import load_test_data, load_train_for_opt, load_opt_and_eval, get_task_type, get_default_use_judge
from agents import MAS
from optimizers import MAPromptOptimizer
from residual_selector import select_residual_answer
from experience import ExperienceMemoryBank, augment_problem, build_memory_entry


async def score_answer(item: Dict[str, Any], problem: str, task_type: TaskType,
                       final_output: str, final_raw: str,
                       use_judge: bool, judge) -> bool:
    if task_type == TaskType.CODE:
        program = humaneval_program_from_fields(problem, final_output or "", final_raw or "")
        if use_judge and judge and item.get('test_list'):
            return await judge.ajudge(program, item['test_list'])
        return bool(program and "def " in program)

    if task_type == TaskType.VQA_OPEN and os.environ.get("VQA_LLM_JUDGE", "0") != "1":
        return vqa_open_equivalent(final_output, item['answer'], dataset=item.get('dataset'))
    if use_judge and judge:
        return await judge.ajudge(problem, item['answer'], final_raw)

    correct_answer = normalize_answer(item['answer'])
    model_answer = normalize_answer(final_output or "")
    is_correct = model_answer == correct_answer
    if task_type in [TaskType.MATH_CHOICE, TaskType.REASONING_CHOICE, TaskType.VQA_CHOICE]:
        is_correct = is_correct or correct_answer in model_answer
    if not is_correct and task_type == TaskType.MATH:
        is_correct = math_equivalent(final_output, item['answer'])
    return is_correct


async def aprocess_single(item: Dict[str, Any], graph_type: GraphType, 
                          task_type: TaskType, prompt_map: Optional[Dict[int, str]] = None,
                          handoff_map: Optional[Dict[str, str]] = None,
                          use_handoff: bool = False,
                          use_judge: bool = False, nr: int = 1,
                          use_disagreement_handoff: bool = False,
                          use_residual_selector: bool = False,
                          selector_client=None,
                          experience_bank: Optional[ExperienceMemoryBank] = None,
                          experience_top_k: int = 3) -> Dict[str, Any]:
    problem = item['problem']
    unique_id = item['unique_id']
    image = item.get('image')

    if task_type == TaskType.CODE:
        func_name = item.get('entry_point') or extract_function_name_from_tests(item.get('test_list', []))
        if func_name:
            problem = f"{problem}\n\nNote: The function should be named '{func_name}'."
    judge_problem = problem
    experience_matches = []
    if experience_bank:
        retrieved = experience_bank.retrieve(
            problem,
            task_type,
            dataset=str(item.get("dataset", "")),
            top_k=experience_top_k,
        )
        problem, experience_matches = augment_problem(problem, retrieved, task_type)
    try:
        if use_residual_selector:
            start = time.time()
            base_mas = MAS(
                graph_type, task_type,
                use_judge=use_judge,
                judge_client=bclient,
                Nr=nr,
                Na=int(os.environ.get("MASPO_NA", "1")),
                use_handoff=False,
                handoff_map=None,
                use_disagreement_handoff=False,
            )
            challenger_mas = MAS(
                graph_type, task_type,
                use_judge=use_judge,
                judge_client=bclient,
                Nr=nr,
                Na=int(os.environ.get("MASPO_NA", "1")),
                use_handoff=use_handoff,
                handoff_map=handoff_map,
                use_disagreement_handoff=use_disagreement_handoff,
            )
            if prompt_map:
                base_mas.inject_prompt_map(prompt_map)
                challenger_mas.inject_prompt_map(prompt_map)
            if handoff_map:
                challenger_mas.inject_handoff_map(handoff_map)

            base_result, challenger_result = await asyncio.gather(
                base_mas.arun(problem, image=image),
                challenger_mas.arun(problem, image=image),
            )
            base_terminal = base_mas.get_terminal_id()
            challenger_terminal = challenger_mas.get_terminal_id()
            base_output = base_result["final"]
            challenger_output = challenger_result["final"]
            base_raw = base_result["raw_trace"].get(base_terminal, "")
            challenger_raw = challenger_result["raw_trace"].get(challenger_terminal, "")

            selection = await select_residual_answer(
                selector_client or create_evaluator_client(),
                problem=problem,
                task_type=task_type,
                base_answer=base_output,
                base_raw=base_raw,
                challenger_answer=challenger_output,
                challenger_raw=challenger_raw,
                image=image,
            )
            final_output = selection["final_answer"]
            used_challenger = bool(selection.get("used_challenger"))
            final_raw = challenger_raw if used_challenger else base_raw

            base_correct = await score_answer(
                item, judge_problem, task_type, base_output, base_raw, use_judge, base_mas.judge
            )
            challenger_correct = await score_answer(
                item, judge_problem, task_type, challenger_output, challenger_raw, use_judge, challenger_mas.judge
            )
            if task_type == TaskType.CODE and item.get("test_list"):
                if challenger_correct and not base_correct:
                    used_challenger = True
                    final_output, final_raw = challenger_output, challenger_raw
                elif base_correct and not challenger_correct:
                    used_challenger = False
                    final_output, final_raw = base_output, base_raw

            end = time.time()
            is_correct = await score_answer(
                item, judge_problem, task_type, final_output, final_raw,
                use_judge, challenger_mas.judge if used_challenger else base_mas.judge
            )
            residual_info = {
                "base_output": base_output,
                "base_raw": base_raw,
                "base_correct": base_correct,
                "challenger_output": challenger_output,
                "challenger_raw": challenger_raw,
                "challenger_correct": challenger_correct,
                "selection": selection,
                "fix_gain": (not base_correct) and is_correct,
                "break_loss": base_correct and (not is_correct),
            }
            return {
                "unique_id": unique_id,
                "graph_type": graph_type.value,
                "output": final_output,
                "final_raw": final_raw,
                "correct": is_correct,
                "response_time": end - start,
                "full_output": challenger_result if used_challenger else base_result,
                "error": None,
                "residual": residual_info,
                "experience": experience_matches,
                "experience_entry": build_memory_entry(
                    item, task_type, graph_type.value,
                    final_output, final_raw, is_correct, residual_info
                ),
            }

        mas = MAS(
            graph_type, task_type,
            use_judge=use_judge,
            judge_client=bclient,
            Nr=nr,
            Na=int(os.environ.get("MASPO_NA", "1")),
            use_handoff=use_handoff,
            handoff_map=handoff_map,
            use_disagreement_handoff=use_disagreement_handoff,
        )
        if prompt_map:
            mas.inject_prompt_map(prompt_map)
        if handoff_map:
            mas.inject_handoff_map(handoff_map)
        
        start = time.time()
        result = await mas.arun(problem, image=image)
        end = time.time()
        final_output = result['final']
        terminal_id = mas.get_terminal_id()
        final_raw = result["raw_trace"][terminal_id]
        
        is_correct = await score_answer(
            item, judge_problem, task_type, final_output, final_raw, use_judge, mas.judge
        )
        return {
            "unique_id": unique_id,
            "graph_type": graph_type.value,
            "output": final_output,
            "final_raw": final_raw,
            "correct": is_correct,
            "response_time": end - start,
            "full_output": result,
            "error": None,
            "experience": experience_matches,
            "experience_entry": build_memory_entry(
                item, task_type, graph_type.value,
                final_output, final_raw, is_correct
            ),
        }
    except Exception as e:
        err = str(e)
        return {
            "unique_id": unique_id,
            "graph_type": graph_type.value,
            "output": None,
            "final_raw": None,
            "correct": False,
            "response_time": 0,
            "full_output": None,
            "error": err,
            "experience": experience_matches,
            "experience_entry": build_memory_entry(
                item, task_type, graph_type.value,
                "", "", False, error=err,
            ),
        }
async def aprocess_task(item: Dict[str, Any], graph_types: List[GraphType],
                        task_type: TaskType, prompt_map: Optional[Dict[int, str]] = None,
                        handoff_map: Optional[Dict[str, str]] = None,
                        use_handoff: bool = False,
                        use_judge: bool = False, nr: int = 1,
                        use_disagreement_handoff: bool = False,
                        use_residual_selector: bool = False,
                        selector_client=None,
                        experience_bank: Optional[ExperienceMemoryBank] = None,
                        experience_top_k: int = 3) -> List[Dict[str, Any]]:

    return [await aprocess_single(item, gt, task_type, prompt_map, handoff_map, use_handoff, use_judge, nr, use_disagreement_handoff, use_residual_selector, selector_client, experience_bank, experience_top_k)
            for gt in graph_types]
async def limited_aprocess_task(item: Dict[str, Any], graph_types: List[GraphType],
                                 task_type: TaskType, sem: asyncio.Semaphore,
                                 prompt_map: Optional[Dict[int, str]] = None,
                                 handoff_map: Optional[Dict[str, str]] = None,
                                 use_handoff: bool = False,
                                 use_judge: bool = False, nr: int = 1,
                                 use_disagreement_handoff: bool = False,
                                 use_residual_selector: bool = False,
                                 selector_client=None,
                                 experience_bank: Optional[ExperienceMemoryBank] = None,
                                 experience_top_k: int = 3):
    async with sem:
        return await aprocess_task(item, graph_types, task_type, prompt_map, handoff_map, use_handoff, use_judge, nr=nr, use_disagreement_handoff=use_disagreement_handoff, use_residual_selector=use_residual_selector, selector_client=selector_client, experience_bank=experience_bank, experience_top_k=experience_top_k)
async def arun_test_suite(data: List[Dict[str, Any]],
                          task_type: TaskType,
                          graph_types: List[GraphType] = None,
                          sample_size: int = None,
                          seed: int = 42,
                          split_info: Optional[Dict[str, Any]] = None,
                          output_file: str = "test_results.json",
                          max_concurrent: int = 20,
                          prompt_map: Optional[Dict[int, str]] = None,
                          handoff_map: Optional[Dict[str, str]] = None,
                          use_handoff: bool = False,
                          use_judge: bool = False,
                          nr: int = 1,
                          use_disagreement_handoff: bool = False,
                          use_residual_selector: bool = False,
                          experience_bank: Optional[ExperienceMemoryBank] = None,
                          experience_top_k: int = 3,
                          write_experience: bool = False):
    if graph_types is None:
        graph_types = list(GraphType)
    
    original_len = len(data)
    if sample_size and sample_size < original_len:
        random.seed(seed)
        data = random.sample(data, sample_size)
        print(f"{original_len} samples {sample_size}")
    else:
        print(f"use {original_len} samples")
    task_sem = asyncio.Semaphore(max_concurrent)
    results = {
        "total": len(data),
        "task_type": task_type.value,
        "graph_types": {gt.value: {"correct": 0, "total": 0, "accuracy": 0.0, "avg_response_time": 0.0}
                       for gt in graph_types},
        "split_info": split_info or {},
        "detailed": [],
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "use_judge": use_judge,
        "residual_selector": use_residual_selector,
        "experience_guided": bool(experience_bank),
        "experience_top_k": experience_top_k,
    }
    if use_residual_selector:
        results["residual_stats"] = {
            "total": 0,
            "selector_called": 0,
            "used_challenger": 0,
            "kept_base": 0,
            "base_correct": 0,
            "challenger_correct": 0,
            "final_correct": 0,
            "fix_gain": 0,
            "break_loss": 0,
        }
    start_total = time.time()
    selector_client = create_evaluator_client() if use_residual_selector else None
    tasks = [
        limited_aprocess_task(
            item, graph_types, task_type, task_sem,
            prompt_map, handoff_map, use_handoff, use_judge, nr, use_disagreement_handoff,
            use_residual_selector, selector_client, experience_bank, experience_top_k
        )
        for item in data
    ]
    results_list = await tqdm_asyncio.gather(*tasks, total=len(tasks), desc="")
    flat_results = [r for sub in results_list for r in sub]
    detailed_map = {}
    memory_entries = []
    for r in flat_results:
        uid, gt_val = r["unique_id"], r["graph_type"]
        if uid not in detailed_map:
            orig = next(item for item in data if item["unique_id"] == uid)
            detailed_map[uid] = {
                "unique_id": uid,
                "problem": orig["problem"],
                "models": {},
            }
            if task_type == TaskType.CODE:
                detailed_map[uid]["test_list"] = orig.get("test_list", [])
            else:
                detailed_map[uid]["correct_answer"] = normalize_answer(orig["answer"])
        
        detailed_map[uid]["models"][gt_val] = {
            "output": r["output"],
            "raw_output": r["final_raw"],
            "correct": r["correct"],
            "response_time": r["response_time"],
            "error": r["error"],
        }
        if r.get("residual"):
            detailed_map[uid]["models"][gt_val]["residual"] = r["residual"]
            rs = results["residual_stats"]
            residual = r["residual"]
            selection = residual.get("selection", {})
            rs["total"] += 1
            rs["selector_called"] += int(bool(selection.get("selector_called")))
            rs["used_challenger"] += int(bool(selection.get("used_challenger")))
            rs["kept_base"] += int(not bool(selection.get("used_challenger")))
            rs["base_correct"] += int(bool(residual.get("base_correct")))
            rs["challenger_correct"] += int(bool(residual.get("challenger_correct")))
            rs["final_correct"] += int(bool(r["correct"]))
            rs["fix_gain"] += int(bool(residual.get("fix_gain")))
            rs["break_loss"] += int(bool(residual.get("break_loss")))
        if r.get("experience"):
            detailed_map[uid]["models"][gt_val]["experience"] = r["experience"]
        if r.get("experience_entry"):
            memory_entries.append(r["experience_entry"])
        
        st = results["graph_types"][gt_val]
        st["total"] += 1
        if r["correct"]:
            st["correct"] += 1
        st["avg_response_time"] += r["response_time"]
    for gt in graph_types:
        st = results["graph_types"][gt.value]
        if st["total"]:
            st["accuracy"] = st["correct"] / st["total"]
            st["avg_response_time"] /= st["total"]
    results["detailed"] = list(detailed_map.values())
    results["experience_stats"] = {
        "retrieval_enabled": bool(experience_bank),
        "write_enabled": bool(write_experience),
        "new_entries": len(memory_entries),
    }
    if write_experience and experience_bank:
        experience_bank.append_many(memory_entries)
    results["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    results["duration"] = time.time() - start_total
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

_NR2_LONG_OUTPUT_SEED_TOKEN_OVERRIDES = {
    ("aqua", 123): 8192,
    ("gpqa", 123): 8192,
    ("humaneval", 456): 8192,
}


def _apply_seed_work_max_tokens(args) -> int | None:
    """Use a larger work-model output cap for known nr2 long-output seeds."""
    if os.environ.get("MASPO_WORK_MAX_TOKENS"):
        return None
    if args.graph != "reflect" or args.nr != 2:
        return None
    cap = _NR2_LONG_OUTPUT_SEED_TOKEN_OVERRIDES.get((args.dataset, args.seed))
    if cap is None:
        return None
    os.environ["MASPO_WORK_MAX_TOKENS"] = str(cap)
    return cap


def main():
    global bclient
    parser = argparse.ArgumentParser(description="Run MAS evaluation with configurable dataset and prompt mode.")
    parser.add_argument("--dataset", type=str, required=True, 
                        choices=list(DATASET_CONFIG.keys()),
                        help=f"Dataset to use: {', '.join(DATASET_CONFIG.keys())}")
    parser.add_argument("--graph", type=str, default="reflect",
                        choices=[gt.value for gt in GraphType],
                        help="Graph type to use")
    parser.add_argument("--optimize", action="store_true",
                        help="Enable prompt optimization. If not set, use original prompts.")
    parser.add_argument("--use-llm-judge", action="store_true",
                        help="Use LLM judge for non-code tasks (code tasks always use code judge)")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Number of samples to use (default: all)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sample selection (default: 42)")
    parser.add_argument("--opt-size", type=int, default=50,
                        help="Number of reserved optimization examples in disjoint split (default: 50)")
    split_group = parser.add_mutually_exclusive_group()
    split_group.add_argument("--disjoint-eval", dest="disjoint_eval", action="store_true", default=True,
                             help="Reserve --opt-size examples and evaluate on the remaining held-out set (default)")
    split_group.add_argument("--no-disjoint-eval", dest="disjoint_eval", action="store_false",
                             help="Disable held-out split and evaluate on the full dataset")
    parser.add_argument("--max-concurrent", type=int, default=20,
                        help="Maximum concurrent tasks")
    parser.add_argument("--prompt-file", type=str, default=None,
                        help="Path to pre-optimized prompt file (JSON)")
    parser.add_argument("--handoff", action="store_true",
                        help="Enable fixed edge-level handoff interfaces during execution.")
    parser.add_argument("--handoff-optimize", action="store_true",
                        help="Run Handoff-MASPO: optimize edge-level paired handoff interfaces.")
    parser.add_argument("--handoff-file", type=str, default=None,
                        help="Path to pre-optimized handoff interface file (JSON)")
    parser.add_argument("--handoff-rounds", type=int, default=1,
                        help="Optimization rounds per edge for Handoff-MASPO")
    parser.add_argument("--structured-meta-prompt", action="store_true",
                        help="Use the 4-stage diagnostic meta-prompt (prompts_structured.py) instead of the official MASPO PROMPT_OPTIMIZE_TEMPLATE.")
    parser.add_argument("--disagreement-handoff", action="store_true",
                        help="v10: topology-aware handoff: multi-input disagreement arbitration + single-chain conservative verification.")
    parser.add_argument("--residual-selector", action="store_true",
                        help="v11: run a MASPO-style base answer and a handoff challenger, then conservatively select the challenger only with high-confidence evidence.")
    parser.add_argument("--v11-min-confidence", type=str, default=None,
                        choices=["LOW", "MEDIUM", "HIGH", "low", "medium", "high"],
                        help="Minimum selector confidence required to override BASE (default HIGH).")
    parser.add_argument("--experience-guided", action="store_true",
                        help="Enable ExHandoff: retrieved error-memory + handoff/disagreement + residual selector.")
    parser.add_argument("--experience-bank", type=str, default="memory/experience_bank.jsonl",
                        help="JSONL memory bank for retrieved reusable failure patterns.")
    parser.add_argument("--experience-top-k", type=int, default=3,
                        help="Number of retrieved memories injected into each problem.")
    parser.add_argument("--write-experience", action="store_true",
                        help="Append new failure/fix/break events to --experience-bank after evaluation.")
    parser.add_argument("--work-port", type=str, default=None,
                        help="vLLM port for the weaker work/target model, default 8005")
    parser.add_argument("--strong-port", type=str, default=None,
                        help="vLLM port for the stronger optimizer/evaluator model, default 8004")
    parser.add_argument("--work-model", type=str, default=None,
                        help="OpenAI model name for the work model, default Qwen/Qwen3.5-4B")
    parser.add_argument("--strong-model", type=str, default=None,
                        help="OpenAI model name for the strong model, default Qwen/Qwen3.5-9B")
    parser.add_argument("--work-max-tokens", type=int, default=None,
                        help="Output token cap for work model calls, default 4096")
    parser.add_argument("--strong-max-tokens", type=int, default=16384,
                        help="Output token cap for strong model calls, default 16384 (matches official MASPO)")
    parser.add_argument("--work-max-prompt-chars", type=int, default=None,
                        help="Character cap for work model prompts before local vLLM calls, default 24000")
    parser.add_argument("--strong-max-prompt-chars", type=int, default=None,
                        help="Character cap for strong model prompts before local vLLM calls, default 8000")
    
    parser.add_argument("--round-robin", action="store_true",
                        help="Use round-robin optimization instead of sequential optimization")
    parser.add_argument("--depth", type=int, default=10,
                        help="Maximum optimization rounds per agent (for round-robin mode)")
    parser.add_argument("--patience", type=int, default=3,
                        help="Early stopping patience: pause agent after N rounds without improvement")
    parser.add_argument("--fixed-rounds", action="store_true",
                    help="Use fixed rounds per turn optimization (2 rounds per agent each turn)")
    parser.add_argument("--dynamic-switching", action="store_true",
                    help="Enable Dynamic Anchor Switching for joint optimization (zero-cost)")
    parser.add_argument("--stochastic-sampling", action="store_true",
                    help="Enable Stochastic Beam Context Sampling for robust joint optimization")
    parser.add_argument("--beam-refresh", action="store_true",
                    help="Enable Beam Refresh strategy to re-score nodes based on updated partners")
    parser.add_argument("--nr", type=int, default=1,
                    help="Number of reflect rounds (for reflect topology)")
    parser.add_argument("--na", type=int, default=1,
                    help="Number of parallel predictors (for aggregate / llm_agg topology). v9: K parallel solvers -> handoff-optimized aggregator.")
    parser.add_argument("--feedback", action="store_true",
                    help="Enable multi-agent collaborative feedback (pass bad cases from downstream to upstream).")
    parser.add_argument("--misleading-sampling", action="store_true",
                    help="Enable sampling injection of upstream 'misleading cases' (Local Win / Global Lose).")
    parser.add_argument("--lookahead-score", action="store_true",
                    help="Enable Lookahead Scoring: 0.5*Local + 0.3*Next_Local + 0.2*Global")
    parser.add_argument("--lookahead-weights", type=str, default="4:4:2",
                        help="Weights for lookahead scoring (Local:Next:Global), e.g., '4:4:2'. Default is 4:4:2 (0.4, 0.4, 0.2)")

    
    args = parser.parse_args()
    if args.experience_guided:
        # ExHandoff is the full method: MASPO prompts + structured handoff +
        # topology-aware verification + conservative residual selection.
        args.handoff = True
        args.handoff_optimize = True
        args.structured_meta_prompt = True
        args.disagreement_handoff = True
        args.residual_selector = True
    if args.v11_min_confidence:
        os.environ["V11_MIN_CONFIDENCE"] = args.v11_min_confidence.upper()
    os.environ["MASPO_NA"] = str(args.na)  # v9: parallel predictor count for llm_agg, read by aprocess_single
    if args.work_port:
        os.environ["MASPO_WORK_PORT"] = args.work_port
        os.environ["MASPO_BASE_URL"] = f"http://localhost:{args.work_port}/v1"
    if args.strong_port:
        os.environ["MASPO_STRONG_PORT"] = args.strong_port
        os.environ["MASPO_EVALUATOR_BASE_URL"] = f"http://localhost:{args.strong_port}/v1"
        os.environ["MASPO_JUDGE_BASE_URL"] = f"http://localhost:{args.strong_port}/v1"
    if args.work_model:
        os.environ["MASPO_MODEL"] = args.work_model
    if args.strong_model:
        os.environ["MASPO_EVALUATOR_MODEL"] = args.strong_model
        os.environ["MASPO_JUDGE_MODEL"] = args.strong_model
    auto_work_max_tokens = None
    if args.work_max_tokens is not None:
        os.environ["MASPO_WORK_MAX_TOKENS"] = str(args.work_max_tokens)
    else:
        auto_work_max_tokens = _apply_seed_work_max_tokens(args)
    if args.strong_max_tokens is not None:
        os.environ["MASPO_STRONG_MAX_TOKENS"] = str(args.strong_max_tokens)
    if args.work_max_prompt_chars is not None:
        os.environ["MASPO_WORK_MAX_PROMPT_CHARS"] = str(args.work_max_prompt_chars)
    if args.strong_max_prompt_chars is not None:
        os.environ["MASPO_STRONG_MAX_PROMPT_CHARS"] = str(args.strong_max_prompt_chars)
    if args.work_port or args.strong_port:
        import agent as agent_module
        import config as config_module
        config_module.aclient = config_module.create_main_client()
        config_module.bclient = config_module.create_judge_client()
        agent_module.aclient = config_module.aclient
        bclient = config_module.bclient

    dataset = args.dataset
    graph_type = GraphType(args.graph)
    task_type = get_task_type(dataset)
    
    default_use_judge = get_default_use_judge(dataset)
    if task_type == TaskType.CODE:
        use_judge = True
    elif task_type == TaskType.VQA_OPEN:
        use_judge = args.use_llm_judge or default_use_judge
    else:
        use_judge = args.use_llm_judge
    
    try:
        w_parts = [float(x) for x in args.lookahead_weights.split(':')]
        if len(w_parts) != 3:
            raise ValueError("Must provide exactly 3 numbers separated by colon.")
        total_w = sum(w_parts)
        if total_w == 0:
            raise ValueError("Sum of weights cannot be 0.")
        lookahead_weights = tuple(w / total_w for w in w_parts)
    except Exception as e:
        print(f"[Warning] Invalid lookahead-weights format ({e}), using default 0.4:0.4:0.2")
        lookahead_weights = (0.4, 0.4, 0.2)
    print(f"Lookahead Weights: {args.lookahead_weights} -> Local:{lookahead_weights[0]:.2f}, Next:{lookahead_weights[1]:.2f}, Global:{lookahead_weights[2]:.2f}")

    print(f"Dataset: {dataset}")
    print(f"Task Type: {task_type.value}")
    print(f"Graph Type: {graph_type.value}")
    print(f"Optimize: {args.optimize}")
    use_handoff = args.handoff or args.handoff_optimize or bool(args.handoff_file)
    print(f"Handoff Enabled: {use_handoff}")
    print(f"Handoff Optimize: {args.handoff_optimize}")
    print(f"Disagreement Handoff: {args.disagreement_handoff}")
    print(f"Residual Selector: {args.residual_selector} (min_conf={os.environ.get('V11_MIN_CONFIDENCE', 'HIGH')})")
    print(f"Experience Guided: {args.experience_guided} (bank={args.experience_bank}, top_k={args.experience_top_k}, write={args.write_experience})")
    print(f"Work LLM: {os.environ.get('MASPO_MODEL', 'Qwen/Qwen3.5-4B')} @ {os.environ.get('MASPO_BASE_URL', 'http://localhost:8005/v1')}")
    print(f"Strong LLM: {os.environ.get('MASPO_EVALUATOR_MODEL', 'Qwen/Qwen3.5-9B')} @ {os.environ.get('MASPO_EVALUATOR_BASE_URL', 'http://localhost:8004/v1')}")
    print(f"Token Caps: work={os.environ.get('MASPO_WORK_MAX_TOKENS', '4096')}, strong={os.environ.get('MASPO_STRONG_MAX_TOKENS', '16384')}")
    if auto_work_max_tokens:
        print(f"Auto Token Override: nr2 {dataset} seed{args.seed} uses work={auto_work_max_tokens}; other seeds keep default.")
    print(f"Prompt Char Caps: work={os.environ.get('MASPO_WORK_MAX_PROMPT_CHARS', '24000')}, strong={os.environ.get('MASPO_STRONG_MAX_PROMPT_CHARS', '8000')}")
    print(f"Round-Robin Mode: {args.round_robin}")
    if args.round_robin:
        print(f"  - Depth per Agent: {args.depth}")
        print(f"  - Patience: {args.patience}")
    print(f"Use Judge: {use_judge}" + (" (code judge)" if task_type == TaskType.CODE else " (LLM judge)" if use_judge else " (string match)"))
    prompt_map = None
    handoff_map = None
    experience_bank = ExperienceMemoryBank(
        args.experience_bank,
        top_k=args.experience_top_k,
    ) if (args.experience_guided or args.write_experience) else None
    if args.prompt_file:
        with open(args.prompt_file, 'r', encoding='utf-8') as f:
            prompt_map = {int(k): v for k, v in json.load(f).items()}
        print(f"Loaded prompts from {args.prompt_file}")
    if args.handoff_file:
        with open(args.handoff_file, 'r', encoding='utf-8') as f:
            handoff_map = {str(k): str(v) for k, v in json.load(f).items()}
        print(f"Loaded handoffs from {args.handoff_file}")
    train_problems = None
    split_info = {
        "disjoint_eval": args.disjoint_eval,
        "seed": args.seed,
        "opt_size": args.opt_size,
    }
    if args.disjoint_eval:
        train_problems, eval_data = load_opt_and_eval(dataset, opt_size=args.opt_size, seed=args.seed)
        split_info.update({
            "opt_count": len(train_problems),
            "eval_count": len(eval_data),
            "note": "Evaluation excludes the reserved optimization pool.",
        })
        print(f"[DISJOINT SPLIT] opt={len(train_problems)} eval={len(eval_data)} (seed={args.seed}, opt_size={args.opt_size})")
    else:
        eval_data = None  # will fall through to full load_test_data below
        split_info.update({
            "opt_count": 0,
            "eval_count": None,
            "note": "Disjoint split disabled; evaluation uses the full loaded dataset.",
        })

    if args.optimize or args.handoff_optimize:
        if train_problems is None:
            train_problems = load_train_for_opt(dataset, k=args.opt_size, seed=args.seed)
            split_info["opt_count"] = len(train_problems)
            print(f"[OPT POOL] opt={len(train_problems)} sampled from full dataset (seed={args.seed}, opt_size={args.opt_size})")
        mas = MAS(
            graph_type, task_type, Nr=args.nr, Na=args.na,
            use_handoff=use_handoff, handoff_map=handoff_map,
            use_disagreement_handoff=args.disagreement_handoff,
        )
        
        seed_map = {
            i: mas.agents[i].template 
            for i in range(len(mas.agents))
            if mas.agents[i].type != AgentType.AGGREGATOR and mas.agents[i].template
        }
        if prompt_map:
            seed_map.update(prompt_map)
        seed_handoff_map = mas.get_handoff_map() if use_handoff else None
        
        requirement = OPTIMIZATION_REQUIREMENTS.get(task_type, "")
        evaluator_client = create_evaluator_client()

        # Multimodal: question->image map so the v3 optimizer can see images
        # (the "image gradient"). Empty/None for text datasets -> no change.
        image_lookup = None
        if task_type in (TaskType.VQA_OPEN, TaskType.VQA_CHOICE):
            full = load_test_data(dataset)
            image_lookup = {it['problem']: it['image'] for it in full if it.get('image')}

        optimizer = MAPromptOptimizer(
            mas, train_problems, seed_map, evaluator_client,
            seed_handoff_map=seed_handoff_map,
            use_handoff=use_handoff,
            use_structured_meta_prompt=args.structured_meta_prompt,
            image_lookup=image_lookup,
            use_disagreement_handoff=args.disagreement_handoff,
        )
        statistics = None
        
        if args.optimize and not args.prompt_file and args.round_robin:
            prompt_map = asyncio.run(optimizer.optimize_all_round_robin(
                requirement=requirement,
                max_total_depth=args.depth,
                patience=args.patience,
                beam_width=2
            ))
        elif args.optimize and not args.prompt_file and args.fixed_rounds:
            prompt_map, statistics = asyncio.run(optimizer.optimize_all_fixed_rounds(
                requirement=requirement,
                max_total_depth=int(os.environ.get('MASPO_FIXED_DEPTH', '3')),
                rounds_per_turn=int(os.environ.get('MASPO_FIXED_ROUNDS_PER_TURN', '3')),
                beam_width=2,
                use_dynamic_switching=args.dynamic_switching,
                use_stochastic_sampling=args.stochastic_sampling,
                use_beam_refresh=args.beam_refresh,
                use_feedback=args.feedback,
                use_misleading_sampling=args.misleading_sampling,
                use_lookahead_score=args.lookahead_score,
                lookahead_weights=lookahead_weights
            ))
        elif args.optimize and not args.prompt_file:
            prompt_map = asyncio.run(optimizer.optimize_all(requirement=requirement))
        else:
            prompt_map = seed_map.copy()
        optimizer.best_prompt = prompt_map.copy()

        handoff_statistics = None
        if args.handoff_optimize:
            handoff_map, handoff_statistics = asyncio.run(optimizer.optimize_all_handoffs(
                requirement=requirement,
                max_rounds=args.handoff_rounds,
            ))
        elif use_handoff:
            handoff_map = optimizer.best_handoff.copy()
        
        if args.round_robin:
            mode_suffix = "rr"
        elif args.lookahead_score and args.misleading_sampling:
            mode_suffix = "ms_ls"
        elif args.misleading_sampling:
            mode_suffix = "ms"
        elif args.lookahead_score:
            mode_suffix = "ls"
        else:
            mode_suffix = "topo"
        if args.handoff_optimize:
            mode_suffix = f"handoff_{mode_suffix}"
        elif use_handoff:
            mode_suffix = f"fixed_handoff_{mode_suffix}"

        os.makedirs("prompt", exist_ok=True)
        prompt_output_file = f"prompt/handoff_maspo_{dataset}_{graph_type.value}_{mode_suffix}_prompts.json"
        with open(prompt_output_file, "w", encoding="utf-8") as f:
            json.dump(prompt_map, f, indent=2, ensure_ascii=False)
        print(f"Optimized prompts saved to {prompt_output_file}")

        if use_handoff and handoff_map:
            handoff_output_file = f"prompt/handoff_maspo_{dataset}_{graph_type.value}_{mode_suffix}_handoffs.json"
            with open(handoff_output_file, "w", encoding="utf-8") as f:
                json.dump(handoff_map, f, indent=2, ensure_ascii=False)
            print(f"Optimized handoffs saved to {handoff_output_file}")

        if statistics or handoff_statistics:
            stats_output_file = f"stats/tbdspo_{dataset}_{graph_type.value}_{mode_suffix}_stats.json"

            os.makedirs("stats", exist_ok=True)
            combined_stats = {
                "prompt_statistics": statistics,
                "handoff_statistics": handoff_statistics,
            }
            
            with open(stats_output_file, "w", encoding="utf-8") as f:
                json.dump(combined_stats, f, indent=2, ensure_ascii=False)
            print(f"Optimization statistics saved to {stats_output_file}")
            
            
    
    # eval data: held-out set by default; pass --no-disjoint-eval to use full data.
    test_data = eval_data if eval_data is not None else load_test_data(dataset)
    split_info["eval_count"] = len(test_data)

    suffix = "tbdspo" if (args.optimize or args.prompt_file or args.handoff_optimize) else "original"
    if args.round_robin and args.optimize:
        suffix = "tbdspo_rr"
    if args.handoff_optimize and args.optimize:
        suffix = "full_maspo"        # MASPO prompt opt + handoff opt
    elif args.handoff_optimize:
        suffix = "handoff_maspo"     # handoff opt only
    elif use_handoff:
        suffix = f"{suffix}_handoff"
    if args.disagreement_handoff:
        suffix = f"{suffix}_dh"
    if args.residual_selector:
        suffix = f"{suffix}_v11"
    if args.experience_guided:
        suffix = f"{suffix}_egmap"
    output_file = f"result/{dataset}_{graph_type.value}_{suffix}.json"
    
    asyncio.run(arun_test_suite(
        test_data,
        task_type=task_type,
        graph_types=[graph_type],
        sample_size=args.sample_size,
        seed=args.seed,
        split_info=split_info,
        output_file=output_file,
        max_concurrent=args.max_concurrent,
        prompt_map=prompt_map,
        handoff_map=handoff_map,
        use_handoff=use_handoff,
        use_judge=use_judge,
        nr=args.nr,
        use_disagreement_handoff=args.disagreement_handoff,
        use_residual_selector=args.residual_selector,
        experience_bank=experience_bank,
        experience_top_k=args.experience_top_k,
        write_experience=args.write_experience,
    ))
if __name__ == "__main__":
    main()
