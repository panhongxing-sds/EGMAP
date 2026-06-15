import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from config import DATASET_CONFIG, TaskType
from utils import extract_output


_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "with", "is", "are",
    "was", "were", "be", "by", "on", "as", "at", "from", "that", "this", "it",
    "what", "which", "how", "why", "when", "where", "answer", "question",
}


def _clip(text: Optional[str], limit: int = 900) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + " ..."


def _tokens(text: str) -> set[str]:
    toks = set(re.findall(r"[a-zA-Z0-9_]{3,}", (text or "").lower()))
    return {t for t in toks if t not in _STOPWORDS}


def _safe_task_value(task_type: Any) -> str:
    return task_type.value if isinstance(task_type, TaskType) else str(task_type)


def _extracted_choice_letter(output: str, raw: str) -> str:
    text = (output or raw or "").strip()
    m = re.search(r"<answer>\s*([A-Ga-g])\s*</answer>", text, re.I)
    if m:
        return m.group(1).upper()
    m = re.search(r"(?:^|\n)\s*(?:final\s+)?(?:answer|option|choice)\s*[:：]?\s*([A-Ga-g])\s*(?:$|\n)", text, re.I)
    if m:
        return m.group(1).upper()
    stripped = text.strip()
    if len(stripped) == 1 and stripped.isalpha():
        return stripped.upper()
    return ""


def _is_truncated_math_output(output: str, raw: str) -> bool:
    out = (output or "").strip()
    if not out:
        return bool((raw or "").strip())
    if "\\end{answer}" in out:
        return True
    if out.count("(") > out.count(")"):
        return True
    if re.search(
        r"^(?:Let's re-evaluate|Sum: \$|is such that|minant \$|ne \(which|es on the last|scribed in)",
        out,
        re.I,
    ):
        return True
    if len(out) > 40 and re.search(r"^[A-Za-z][a-z]+ ", out) and not re.search(
        r"\\boxed\{|\$[^$]+\$|^\\begin\{",
        out,
    ):
        return True
    return False


def _is_bankable_model_answer(
    model_answer: str,
    *,
    task_type: TaskType,
    error: Optional[str] = None,
) -> bool:
    if error:
        return True
    ans = (model_answer or "").strip()
    if not ans:
        return False
    if task_type in (TaskType.MATH_CHOICE, TaskType.REASONING_CHOICE, TaskType.VQA_CHOICE):
        return len(ans) == 1 and ans.isalpha()
    if task_type == TaskType.CODE:
        return "def " in ans or "class " in ans
    if task_type == TaskType.MATH:
        if "\\end{answer}" in ans:
            return False
        if ans.count("(") > ans.count(")"):
            return False
        if _is_truncated_math_output(ans, ""):
            return False
    return True


def classify_error(
    task_type: TaskType,
    problem: str,
    output: str,
    raw: str,
    *,
    error: Optional[str] = None,
) -> str:
    """Return a coarse, label-free error signature for reusable experience."""
    err = (error or "").lower()
    if err:
        if "timed out" in err or "timeout" in err:
            return "runtime_timeout"
        return "runtime_error"

    q = (problem or "").lower()
    out = (output or "").strip()
    trace = (raw or "").lower()

    if task_type == TaskType.CODE:
        if "def " not in out and "def " not in trace:
            return "code_format_or_missing_function"
        if "index" in trace or "empty" in trace or "edge" in trace:
            return "code_edge_case"
        return "code_semantic_bug"

    if task_type in (TaskType.MATH_CHOICE, TaskType.REASONING_CHOICE):
        letter = _extracted_choice_letter(output, raw)
        if not letter:
            return "choice_not_extractable"
        if "except" in q or "not " in q or "least" in q:
            return "choice_negation_or_trap"
        return "choice_reasoning_error"

    if task_type == TaskType.MATH:
        if _is_truncated_math_output(output, raw):
            return "output_truncated"
        if re.search(r"\d", q) and re.search(r"\d", trace):
            return "calculation_or_algebra_error"
        return "math_reasoning_error"

    if task_type in (TaskType.VQA_OPEN, TaskType.VQA_CHOICE):
        if re.search(r"\b(chart|axis|bar|legend|value|percent|percentage)\b", q):
            return "visual_chart_or_numeric_grounding"
        if re.search(r"\b(text|written|word|label|sign|ocr)\b", q):
            return "visual_ocr_grounding"
        if re.search(r"\bct|xray|mri|lesion|organ|lung|heart|medical|radiograph\b", q):
            return "medical_visual_grounding"
        return "visual_evidence_grounding"
    return "general_reasoning_error"


def advice_for_error(error_type: str, task_type: TaskType) -> str:
    """Convert an error signature into a promptable, label-free correction rule."""
    advice = {
        "choice_not_extractable": "End with exactly one option letter in <answer>...</answer>; do not leave the final choice implicit.",
        "choice_negation_or_trap": "Re-read negations, exception words, and option traps before changing the selected option.",
        "choice_reasoning_error": "Verify the decisive option evidence from first principles before trusting another agent.",
        "output_truncated": "Finish the decisive calculation and state the complete final answer (use \\boxed{} or <answer>) before stopping.",
        "calculation_or_algebra_error": "Re-check the decisive arithmetic or algebra step; state the complete simplified final answer.",
        "math_reasoning_error": "Identify the exact logical step that determines the final answer, then compute it fully.",
        "runtime_timeout": "Keep reasoning concise; prioritize reaching a complete final answer over long exploratory text.",
        "runtime_error": "Avoid overly long chains; verify the final answer format before finishing.",
        "code_format_or_missing_function": "Return executable Python with the required function signature and no prose-only answer.",
        "code_edge_case": "Mentally execute the code on empty, boundary, and example cases before finalizing.",
        "code_semantic_bug": "Trace the examples through the code and repair the specific violated constraint.",
        "visual_chart_or_numeric_grounding": "Ground the answer in chart axis/legend/OCR values; normalize numbers, commas, units, and percentages.",
        "visual_ocr_grounding": "Use explicit visible text/OCR as evidence and avoid speculative object/color guesses.",
        "medical_visual_grounding": "Name the visible anatomical/pathological finding and revise only when the image clearly contradicts it.",
        "visual_evidence_grounding": "State the visual evidence that supports the answer and revise only for clear visual contradictions.",
    }.get(error_type)
    if advice:
        return advice
    if task_type in (TaskType.VQA_OPEN, TaskType.VQA_CHOICE):
        return "Use concrete visual evidence; give a complete supported final answer."
    return "Complete the decisive reasoning step and give a fully formatted final answer."


@dataclass
class Experience:
    dataset: str
    task_type: str
    problem: str
    error_type: str
    advice: str
    source: str = "unknown"
    correct: Optional[bool] = None
    model_answer: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, obj: Dict[str, Any]) -> "Experience":
        return cls(
            dataset=str(obj.get("dataset", "")),
            task_type=str(obj.get("task_type", "")),
            problem=str(obj.get("problem", "")),
            error_type=str(obj.get("error_type", "general_reasoning_error")),
            advice=str(obj.get("advice", "")),
            source=str(obj.get("source", "unknown")),
            correct=obj.get("correct"),
            model_answer=str(obj.get("model_answer", "")),
            metadata=dict(obj.get("metadata", {}) or {}),
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "task_type": self.task_type,
            "problem": self.problem,
            "error_type": self.error_type,
            "advice": self.advice,
            "source": self.source,
            "correct": self.correct,
            "model_answer": self.model_answer,
            "metadata": self.metadata,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }


class ExperienceMemoryBank:
    """Small JSONL memory bank used at inference time.

    The retrieved context is deliberately label-free by default: it exposes
    reusable error patterns and correction rules, not gold answers.
    """

    def __init__(self, path: Optional[str], top_k: int = 3):
        self.path = Path(path) if path else None
        self.top_k = top_k
        self.memories: List[Experience] = []
        if self.path and self.path.exists():
            self.memories = self._load(self.path)

    @staticmethod
    def _load(path: Path) -> List[Experience]:
        memories: List[Experience] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    memories.append(Experience.from_json(json.loads(line)))
                except Exception:
                    continue
        return memories

    def retrieve(self, problem: str, task_type: TaskType, dataset: str = "", top_k: Optional[int] = None) -> List[Tuple[Experience, float]]:
        if not self.memories:
            return []
        query_tokens = _tokens(problem)
        task_value = _safe_task_value(task_type)
        scored: List[Tuple[Experience, float]] = []
        for mem in self.memories:
            mem_tokens = _tokens(mem.problem)
            overlap = len(query_tokens & mem_tokens)
            union = max(1, len(query_tokens | mem_tokens))
            score = overlap / union
            if dataset and mem.dataset == dataset:
                score += 0.25
            if mem.task_type == task_value:
                score += 0.15
            if mem.correct is False:
                score += 0.05
            if score > 0:
                scored.append((mem, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        k = top_k if top_k is not None else self.top_k
        deduped: List[Tuple[Experience, float]] = []
        seen: set[Tuple[str, str]] = set()
        for mem, score in scored:
            key = (mem.error_type, mem.advice)
            if key in seen:
                continue
            seen.add(key)
            deduped.append((mem, score))
            if len(deduped) >= k:
                break
        return deduped

    def append_many(self, entries: Sequence[Dict[str, Any]]) -> None:
        if not self.path or not entries:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self.memories.extend(Experience.from_json(e) for e in entries)


def format_experience_context(matches: Iterable[Tuple[Experience, float]], task_type: TaskType) -> str:
    rows = list(matches)
    if not rows:
        return ""
    lines = [
        "[EXPERIENCE-GUIDED HANDOFF MEMORY]",
        "Use these retrieved failure patterns as reusable guidance. They are not gold answers for the current problem.",
    ]
    seen = set()
    for idx, (mem, score) in enumerate(rows, 1):
        key = (mem.error_type, mem.advice)
        if key in seen:
            continue
        seen.add(key)
        lines.append(
            f"{idx}. pattern={mem.error_type}; similarity={score:.2f}; "
            f"rule={_clip(mem.advice, 260)}"
        )
        if os.environ.get("EGMAP_SHOW_MEMORY_EXAMPLES", "0") == "1":
            lines.append(f"   prior_problem={_clip(mem.problem, 320)}")
    if task_type in (TaskType.MATH_CHOICE, TaskType.REASONING_CHOICE, TaskType.VQA_CHOICE):
        lines.append("Final-format guard: output exactly one option letter when the task is multiple-choice.")
    elif task_type == TaskType.CODE:
        lines.append("Final-format guard: output executable code with the required function name.")
    elif task_type in (TaskType.VQA_OPEN, TaskType.VQA_CHOICE):
        lines.append("Visual guard: keep visually supported answers; correct only with concrete image/OCR/chart evidence.")
    lines.append("[/EXPERIENCE-GUIDED HANDOFF MEMORY]")
    return "\n".join(lines)


def augment_problem(problem: str, matches: Iterable[Tuple[Experience, float]], task_type: TaskType) -> Tuple[str, List[Dict[str, Any]]]:
    rows = list(matches)
    block = format_experience_context(rows, task_type)
    if not block:
        return problem, []
    metadata = [
        {
            "dataset": mem.dataset,
            "task_type": mem.task_type,
            "error_type": mem.error_type,
            "score": score,
            "advice": mem.advice,
            "source": mem.source,
        }
        for mem, score in rows
    ]
    return f"{problem}\n\n{block}", metadata


def build_memory_entry(item: Dict[str, Any], task_type: TaskType, graph_type: str,
                       output: str, raw: str, correct: bool,
                       residual: Optional[Dict[str, Any]] = None,
                       error: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Create a reusable experience from a wrong answer or runtime failure.

    The bank is failure-only: correct answers are not stored, even when
    residual fix_gain is true. Runtime failures (timeouts) are always stored.
    """
    residual = residual or {}
    runtime_failed = bool(error)
    if correct and not runtime_failed:
        return None

    problem = item.get("problem", "")
    model_answer = extract_output(output or raw or "", task_type)
    if not _is_bankable_model_answer(model_answer, task_type=task_type, error=error):
        if runtime_failed:
            model_answer = ""
        else:
            return None

    error_type = classify_error(task_type, problem, output or "", raw or "", error=error)
    entry = Experience(
        dataset=str(item.get("dataset", "")),
        task_type=_safe_task_value(task_type),
        problem=problem,
        error_type=error_type,
        advice=advice_for_error(error_type, task_type),
        source=f"{graph_type}:{'opt' if runtime_failed else 'eval'}",
        correct=False,
        model_answer=model_answer,
        metadata={
            "unique_id": item.get("unique_id"),
            "residual_fix_gain": bool(residual.get("fix_gain")),
            "residual_break_loss": bool(residual.get("break_loss")),
            "runtime_error": error or "",
        },
    )
    return entry.to_json()


def finalize_experience_bank(path: Path, limit: int) -> int:
    """Filter, dedupe, and cap a JSONL bank after stage-1 opt build.

    - Keeps failure-only rows (``correct`` must not be true)
    - Drops unscoreable ``model_answer`` fragments
    - Dedupes by ``unique_id`` (first occurrence wins)
    - Caps with round-robin over ``error_type`` for diversity
    """
    if not path.exists() or limit <= 0:
        return 0
    entries: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entries.append(json.loads(line))
        except Exception:
            continue

    task_lookup = {name: cfg["task_type"] for name, cfg in DATASET_CONFIG.items()}
    filtered: List[Dict[str, Any]] = []
    seen_uid: set[str] = set()
    for obj in entries:
        if obj.get("correct") is True:
            continue
        uid = str((obj.get("metadata") or {}).get("unique_id") or "")
        if uid and uid in seen_uid:
            continue
        ds = str(obj.get("dataset") or "")
        tt = task_lookup.get(ds, TaskType.MATH)
        err = (obj.get("metadata") or {}).get("runtime_error") or None
        if not _is_bankable_model_answer(
            str(obj.get("model_answer") or ""),
            task_type=tt,
            error=err or None,
        ):
            if not err:
                continue
        if uid:
            seen_uid.add(uid)
        filtered.append(obj)

    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for obj in filtered:
        by_type.setdefault(str(obj.get("error_type") or "general"), []).append(obj)

    kept: List[Dict[str, Any]] = []
    types = sorted(by_type.keys())
    idx = {t: 0 for t in types}
    while len(kept) < limit and types:
        progressed = False
        for t in types:
            i = idx[t]
            if i < len(by_type[t]):
                kept.append(by_type[t][i])
                idx[t] = i + 1
                progressed = True
                if len(kept) >= limit:
                    break
        if not progressed:
            break

    path.write_text("".join(json.dumps(obj, ensure_ascii=False) + "\n" for obj in kept), encoding="utf-8")
    return len(kept)
