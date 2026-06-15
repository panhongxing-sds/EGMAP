
import json
import os
import re
import ast
import base64
import random
from typing import List, Dict, Any

from config import DATASET_CONFIG, TaskType


def _letter_from_choices(question: str, gold: str) -> str:
    """For multiple-choice VQA where options are embedded in the question as
    'A) A:text B) B:text ...' and the gold answer is the option *text*, map the
    gold text to its option letter so judging is a deterministic letter match.
    Falls back to the original gold if no option matches."""
    opts = re.findall(r'([A-Z])\)\s*(?:[A-Z]:\s*)?(.+?)(?=\s+[A-Z]\)|$)', question)
    g = (gold or '').strip().lower()
    for letter, content in opts:
        if content.strip().lower() == g:
            return letter
    return gold

def load_test_data(dataset: str) -> List[Dict[str, Any]]:
    config = DATASET_CONFIG.get(dataset)
    if not config:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    file_path = config["path"]
    task_type = config["task_type"]
    problem_key = config["problem_key"]
    answer_key = config["answer_key"]
    
    image_key = config.get("image_key")
    image_dir = config.get("image_dir")

    # Multimodal datasets ship as a JSON array plus a separate image dir;
    # text datasets keep the original JSONL line-by-line path (unchanged).
    is_array = bool(image_key) or file_path.endswith(".json")

    def _build_entry(item, idx):
        entry = {
            'problem': item[problem_key],
            'unique_id': item.get('unique_id', item.get('task_id', str(idx))),
            'task_type': task_type,
            'dataset': dataset,
        }
        if task_type == TaskType.CODE:
            entry['answer'] = ''
            entry_point = item.get('entry_point')
            entry['entry_point'] = entry_point
            if item.get('test_list'):
                entry['test_list'] = item['test_list']
            elif item.get('test') and entry_point:
                # HumanEval JSONL stores harness in ``test`` with a ``check(candidate)`` entry point.
                entry['test_list'] = [item['test'], f"check({entry_point})"]
            else:
                entry['test_list'] = []
        else:
            entry['answer'] = item.get(answer_key, '')
            if config.get("answer_is_list"):
                try:
                    _p = ast.literal_eval(entry['answer'])
                    if isinstance(_p, list) and _p:
                        entry['answer'] = str(_p[0])
                except Exception:
                    pass
            if config.get("choice_answer_from_text"):
                entry['answer'] = _letter_from_choices(item[problem_key], entry['answer'])
        if image_key and item.get(image_key):
            img_path = os.path.join(image_dir, item[image_key]) if image_dir else item[image_key]
            with open(img_path, 'rb') as fp:
                entry['image'] = base64.b64encode(fp.read()).decode('ascii')
            entry['image_name'] = item[image_key]
            entry['unique_id'] = f"{item[image_key]}#{idx}"
        return entry

    row_filter = config.get("row_filter")  # optional (field, value), e.g. keep only English rows
    data = []
    if is_array:
        with open(file_path, 'r', encoding='utf-8') as f:
            items = json.load(f)
        for idx, item in enumerate(items):
            if row_filter and item.get(row_filter[0]) != row_filter[1]:
                continue
            data.append(_build_entry(item, idx))
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                data.append(_build_entry(item, len(data)))

    return data

def split_opt_eval_items(
    dataset: str, opt_size: int = 50, seed: int = 42
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Disjoint opt/eval split by ``unique_id`` (single source of truth for formal runs)."""
    data = load_test_data(dataset)
    opt_size = min(opt_size, len(data))
    random.seed(seed)
    opt_items = random.sample(data, opt_size)
    opt_ids = {item["unique_id"] for item in opt_items}
    eval_items = [item for item in data if item["unique_id"] not in opt_ids]
    verify_disjoint(opt_items, eval_items, dataset=dataset, seed=seed)
    return opt_items, eval_items


def select_eval_subset(
    eval_items: List[Dict[str, Any]], sample_size: int | None = None, seed: int = 42
) -> List[Dict[str, Any]]:
    """Deterministic held-out subsample; call once and reuse the list for EGMAP + MASPO."""
    items = list(eval_items)
    if sample_size and sample_size < len(items):
        random.seed(seed)
        return random.sample(items, sample_size)
    return items


def verify_disjoint(
    opt_items: List[Dict[str, Any]],
    eval_items: List[Dict[str, Any]],
    *,
    dataset: str = "",
    seed: int | None = None,
) -> None:
    opt_ids = {item["unique_id"] for item in opt_items}
    eval_ids = {item["unique_id"] for item in eval_items}
    overlap = opt_ids & eval_ids
    if overlap:
        ctx = f"dataset={dataset} seed={seed}" if dataset else ""
        sample = sorted(overlap)[:5]
        raise ValueError(
            f"opt/eval leakage: {len(overlap)} shared unique_id(s) {sample} {ctx}".strip()
        )


def verify_bank_from_opt_only(bank_path: str, opt_ids: set[str]) -> None:
    """Ensure experience bank rows (if tagged) only reference opt-pool problems."""
    path = bank_path if isinstance(bank_path, str) else str(bank_path)
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            uid = (obj.get("metadata") or {}).get("unique_id")
            if uid and uid not in opt_ids:
                raise ValueError(
                    f"experience bank leakage at {path}:{line_no}: "
                    f"unique_id {uid!r} not in opt pool ({len(opt_ids)} ids)"
                )


def save_formal_split_manifest(manifest_path: str, manifest: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def load_formal_split_manifest(manifest_path: str) -> Dict[str, Any]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_train_for_opt(dataset: str, k: int = 50, seed: int = 42) -> List[str]:
    opt_items, _ = split_opt_eval_items(dataset, opt_size=k, seed=seed)
    return [item["problem"] for item in opt_items]


def load_opt_and_eval(dataset: str, opt_size: int = 50, seed: int = 42):
    """Disjoint split: opt set (for optimization) and eval set (held out).

    Returns: (opt_problems: List[str], eval_data: List[Dict])
    """
    opt_items, eval_items = split_opt_eval_items(dataset, opt_size=opt_size, seed=seed)
    opt_problems = [item["problem"] for item in opt_items]
    return opt_problems, eval_items

def get_task_type(dataset: str) -> TaskType:
    config = DATASET_CONFIG.get(dataset)
    if not config:
        raise ValueError(f"Unknown dataset: {dataset}")
    return config["task_type"]

def get_default_use_judge(dataset: str) -> bool:
    config = DATASET_CONFIG.get(dataset)
    if not config:
        return False
    return config.get("default_use_judge", False)
