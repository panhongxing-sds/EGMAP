import json
import os
import re
from typing import Any, Dict, Optional

from config import TaskType
from utils import async_call_llm, extract_output, normalize_answer, humaneval_program_from_fields


_CONF_RANK = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
_CHOICE_TASKS = {TaskType.MATH_CHOICE, TaskType.REASONING_CHOICE}
_VQA_EVIDENCE_WORDS = {
    "visual", "visible", "image", "shown", "shows", "seen", "chart", "bar",
    "axis", "label", "ocr", "text", "written", "number", "value", "percent",
    "percentage", "legend", "color", "lesion", "mass", "calcification",
    "calcified", "ventricle", "organ", "anatomy", "opacity", "density",
}
_COLORS_AND_MATERIALS = {
    "black", "white", "gray", "grey", "red", "blue", "green", "yellow",
    "orange", "purple", "brown", "pink", "silver", "gold", "bronze",
}


def _clip(text: Optional[str], limit: int = 1200) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + " ..."


def _norm(answer: Optional[str], task_type: TaskType) -> str:
    text = (answer or "").strip()
    if not text:
        return ""
    text = _strip_answer_label(text)
    if task_type in (TaskType.VQA_OPEN, TaskType.VQA_CHOICE):
        text = re.sub(r"</?answer>", "", text.lower())
        text = re.sub(r"[^a-z0-9一-鿿 ]", " ", text)
        return re.sub(r"\s+", " ", text).strip()
    if task_type == TaskType.CODE:
        return re.sub(r"\s+", "", text)[:240]
    return normalize_answer(text)


def _strip_answer_label(text: str) -> str:
    text = re.sub(r"</?answer>", "", text, flags=re.I).strip()
    patterns = [
        r"^(?:therefore|thus|so)[,:\s]+(?:the\s+)?(?:final\s+)?answer\s*(?:is|:|=|-)?\s*",
        r"^(?:the\s+)?(?:final\s+)?answer\s*(?:is|:|=|-)?\s*",
        r"^(?:choice|option)\s*(?:is|:|=|-)?\s*",
    ]
    for pat in patterns:
        text = re.sub(pat, "", text, flags=re.I).strip()
    return text.strip(" .。,:;")


def _vqa_mode() -> str:
    return (os.environ.get("V11_1_VQA_MODE") or os.environ.get("V11_VQA_MODE") or "generic").strip().lower()


def _task_policy(task_type: TaskType, problem: str = "") -> str:
    if task_type in (TaskType.VQA_OPEN, TaskType.VQA_CHOICE):
        mode = _vqa_mode()
        base = (
            "Use visual evidence as the primary criterion. Prefer CHALLENGER only if it identifies a concrete "
            "visual/OCR/chart/medical-entity error in BASE and its answer is better supported by the image. "
            "If the image evidence is ambiguous, KEEP_BASE."
        )
        if mode in {"vqarad", "slake"}:
            return (
                base + " For medical VQA, use selective strong correction: override a weak or generic BASE answer "
                "when CHALLENGER names a visible anatomical/pathological finding, modality cue, or localization "
                "that directly answers the question. Do not be conservative merely because the answer is yes/no; "
                "be conservative only when both answers are similarly specific and visually plausible."
            )
        if mode == "chartqa":
            return (
                base + " For ChartQA, actively use direct OCR/legend/axis/bar evidence and explicit arithmetic. "
                "First infer whether the question asks for an entity/category or a numeric value, then choose the "
                "answer whose type and extracted chart evidence match the question."
            )
        if mode == "textvqa":
            return (
                base + " For TextVQA, actively prefer CHALLENGER when it reads explicit text/OCR that BASE missed, "
                "or when BASE is generic/incomplete. Keep BASE for speculative color/material/object guesses or "
                "minor one-character OCR variants unless CHALLENGER cites unambiguous visible text."
            )
        return base
    if task_type == TaskType.CODE:
        return (
            "Use function signature, examples, edge cases, and constraints as the primary criteria. Prefer CHALLENGER "
            "only if BASE has a concrete bug or violated constraint and CHALLENGER fixes it. If uncertain, KEEP_BASE."
        )
    if task_type in (TaskType.MATH_CHOICE, TaskType.REASONING_CHOICE):
        return (
            "Use option evidence and the decisive reasoning step. Prefer CHALLENGER only if BASE makes a concrete "
            "logical/option-selection error and CHALLENGER provides stronger evidence. If uncertain, KEEP_BASE."
        )
    return (
        "Use the decisive calculation/reasoning step. Prefer CHALLENGER only if BASE contains a concrete arithmetic, "
        "algebraic, or logical error and CHALLENGER fixes it. If uncertain, KEEP_BASE."
    )


def _selector_prompt(problem: str, task_type: TaskType,
                     base_answer: str, base_raw: str,
                     challenger_answer: str, challenger_raw: str) -> str:
    return f"""
You are a risk-controlled residual selector for a multi-agent system.

Goal: preserve a strong BASE answer by default, and override it only when the CHALLENGER has clear evidence that BASE is wrong.

Task policy: {_task_policy(task_type, problem)}

Question:
{_clip(problem, 2400)}

BASE answer:
{_clip(base_answer, 500)}

BASE reasoning/evidence snippet:
{_clip(base_raw, 1600)}

CHALLENGER answer:
{_clip(challenger_answer, 500)}

CHALLENGER reasoning/evidence snippet:
{_clip(challenger_raw, 1600)}

Decision rules:
1. Default to KEEP_BASE.
2. Choose USE_CHALLENGER only if there is a concrete, named error in BASE and the CHALLENGER answer is better supported.
3. Do not choose CHALLENGER for style, verbosity, or speculative reasoning.
4. Do not invent a third answer; only choose between BASE and CHALLENGER.
5. Confidence must be HIGH only when the evidence is decisive; otherwise MEDIUM or LOW.

Return strict JSON only:
{{"choice":"KEEP_BASE|USE_CHALLENGER", "confidence":"LOW|MEDIUM|HIGH", "reason":"one short sentence"}}
""".strip()


def _parse_selector_json(text: str) -> Dict[str, str]:
    raw = (text or "").strip()
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            return {
                "choice": str(obj.get("choice", "KEEP_BASE")).upper(),
                "confidence": str(obj.get("confidence", "LOW")).upper(),
                "reason": str(obj.get("reason", ""))[:500],
            }
        except Exception:
            pass
    choice = "USE_CHALLENGER" if "USE_CHALLENGER" in raw.upper() else "KEEP_BASE"
    confidence = "HIGH" if "HIGH" in raw.upper() else "MEDIUM" if "MEDIUM" in raw.upper() else "LOW"
    return {"choice": choice, "confidence": confidence, "reason": _clip(raw, 300)}


def _min_confidence_rank() -> int:
    val = os.environ.get("V11_MIN_CONFIDENCE", "HIGH").strip().upper()
    return _CONF_RANK.get(val, _CONF_RANK["HIGH"])


def _choice_verifier_enabled() -> bool:
    return os.environ.get("V11_CHOICE_VERIFIER", "1").strip().lower() not in {"0", "false", "no", "off"}


def _norm_choice(value: Optional[str]) -> Optional[str]:
    match = re.fullmatch(r"\(?\s*([A-Da-d])\s*\)?[.:]?\s*", str(value or "").strip())
    return match.group(1).upper() if match else None


def _extract_choice(answer: Optional[str], raw: Optional[str]) -> Optional[str]:
    """Extract a deployable A-D answer without using labels/gold answers."""
    choice = _norm_choice(answer)
    if choice:
        return choice

    text = raw or ""
    for content in reversed(re.findall(r"<answer[^>]*>(.*?)</answer>", text, flags=re.I | re.S)):
        choice = _norm_choice(content)
        if choice:
            return choice
    for content in reversed(re.findall(r"\\boxed\s*\{([^{}]{0,80})\}", text, flags=re.S)):
        choice = _norm_choice(content)
        if choice:
            return choice

    tail = text[-500:]
    patterns = [
        r"(?:final\s+answer|correct\s+answer|answer|option|choice)\s*(?:is|:)?\s*[*\s]*\(?\s*([A-D])\s*\)?\b",
        r"corresponds\s+to\s+option\s*\(?\s*([A-D])\s*\)?\b",
    ]
    hits = []
    for pattern in patterns:
        hits += re.findall(pattern, tail, flags=re.I)
    return hits[-1].upper() if hits else None


def _choice_verifier(task_type: TaskType, base_answer: str, base_raw: str,
                     challenger_answer: str, challenger_raw: str) -> Optional[Dict[str, Any]]:
    """Deterministic residual gate for AQuA/GPQA-style choice tasks.

    It only uses answer well-formedness available at inference time. It does not
    inspect gold labels or saved correctness flags.
    """
    if task_type not in _CHOICE_TASKS or not _choice_verifier_enabled():
        return None

    base_choice = _extract_choice(base_answer, base_raw)
    challenger_choice = _extract_choice(challenger_answer, challenger_raw)

    if not base_choice and challenger_choice:
        return {
            "choice": "USE_CHALLENGER",
            "confidence": "HIGH",
            "reason": "Choice verifier: BASE has no extractable final option while CHALLENGER provides one.",
            "final_answer": challenger_answer,
            "used_challenger": True,
            "selector_called": False,
            "deterministic_choice_verifier": True,
            "base_choice": None,
            "challenger_choice": challenger_choice,
        }
    if base_choice and not challenger_choice:
        return {
            "choice": "KEEP_BASE",
            "confidence": "HIGH",
            "reason": "Choice verifier: BASE provides an extractable final option while CHALLENGER does not.",
            "final_answer": base_answer,
            "used_challenger": False,
            "selector_called": False,
            "deterministic_choice_verifier": True,
            "base_choice": base_choice,
            "challenger_choice": None,
        }
    return None


def _compact(text: Optional[str]) -> str:
    return re.sub(r"[^a-z0-9一-鿿]+", "", (text or "").lower())


def _is_numeric_only(text: Optional[str]) -> bool:
    s = _strip_answer_label(text or "").lower().strip()
    if not s:
        return False
    return bool(re.fullmatch(r"[\d\s,./:%+\-]+", s)) and bool(re.search(r"\d", s))


def _is_yes_no(text: Optional[str]) -> bool:
    return _norm(text, TaskType.VQA_OPEN) in {"yes", "no"}


def _is_generic_bad_answer(text: Optional[str]) -> bool:
    n = _norm(text, TaskType.VQA_OPEN)
    return n in {
        "", "unknown", "unk", "none", "n a", "na", "not sure", "cannot determine",
        "can not determine", "unclear", "insufficient data", "not visible",
    }


def _wants_entity(problem: str) -> bool:
    q = problem.lower()
    if re.search(r"\b(which|who|where)\b", q):
        return True
    return bool(re.search(r"\b(category|group|country|year|name|label|item|object|organ|plane|view)\b", q))


def _wants_numeric(problem: str) -> bool:
    q = problem.lower()
    if re.search(r"\bwhich\b", q) and re.search(r"\b(category|group|country|year|has|is)\b", q):
        return False
    return bool(re.search(r"\b(value|number|percentage|percent|how many|count|amount|total|ratio|rate)\b", q))


def _asks_text(problem: str) -> bool:
    return bool(re.search(r"\b(text|written|word|name|label|sign|poster|advertisement|monitor|screen|title|brand)\b", problem.lower()))


def _asks_color_or_material(problem: str) -> bool:
    return bool(re.search(r"\b(color|colour|metal|material)\b", problem.lower()))


def _edit_distance_at_most_one(a: str, b: str) -> bool:
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) > len(b):
        a, b = b, a
    i = j = edits = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1
            j += 1
            continue
        edits += 1
        if edits > 1:
            return False
        if len(a) == len(b):
            i += 1
            j += 1
        else:
            j += 1
    return True


def _vqa_hard_reject(problem: str, mode: str, base_answer: str,
                     challenger_answer: str, parsed: Dict[str, str]) -> Optional[str]:
    """Cheap safeguards for VQA break-loss before accepting a challenger.

    These are answer-shape and task-shape checks, not label-aware checks.
    """
    if parsed.get("choice") != "USE_CHALLENGER":
        return None
    q = (problem or "").lower()
    b_norm = _norm(base_answer, TaskType.VQA_OPEN)
    c_norm = _norm(challenger_answer, TaskType.VQA_OPEN)
    reason = (parsed.get("reason") or "").lower()

    if _is_generic_bad_answer(challenger_answer) and not _is_generic_bad_answer(base_answer):
        return "challenger answer is generic/empty while base is specific"

    if mode in {"vqarad", "slake"}:
        if "plane" in q and c_norm in {"ap", "pa", "anteroposterior", "posteroanterior"}:
            return "projection/view answer does not match a plane question"
        if re.search(r"\b(which two|two)\b", q) and not re.search(r"\b(and|both|,|/|&|\\+)\b", c_norm):
            return "question asks for two findings but challenger gives a single finding"

    if mode == "chartqa":
        if _wants_entity(problem) and _is_numeric_only(challenger_answer):
            return "chart question asks for an entity/category, but challenger gives only a number"
        if _wants_numeric(problem) and not _is_numeric_only(challenger_answer) and _is_numeric_only(base_answer):
            return "chart question asks for a value, but challenger changes it to a non-numeric answer"
        if _is_numeric_only(base_answer) and _is_numeric_only(challenger_answer) and _compact(base_answer) != _compact(challenger_answer):
            if not _is_generic_bad_answer(base_answer) and "calculation" not in reason and "sum" not in reason:
                return "numeric-to-numeric chart override is not backed by an explicit calculation"

    if mode == "textvqa":
        if _asks_color_or_material(problem):
            b = b_norm.split()
            c = c_norm.split()
            if b and c and b[-1] in _COLORS_AND_MATERIALS and c[-1] in _COLORS_AND_MATERIALS and b[-1] != c[-1]:
                return "color/material override is too speculative for TextVQA"
        b_comp, c_comp = _compact(base_answer), _compact(challenger_answer)
        if len(b_comp) >= 5 and len(c_comp) >= 5 and _edit_distance_at_most_one(b_comp, c_comp):
            return "one-character OCR variant is not reliable enough to override base"
        if _asks_text(problem) and b_comp and c_comp and len(c_comp) + 3 < len(b_comp) and c_comp not in b_comp:
            return "text answer becomes shorter/less specific without clear containment"

    return None


def _vqa_safety_prompt(problem: str, mode: str, base_answer: str, base_raw: str,
                       challenger_answer: str, challenger_raw: str,
                       selector_reason: str) -> str:
    if mode in {"vqarad", "slake"}:
        policy = (
            "Medical images are ambiguous. ALLOW only if BASE is clearly impossible or misses a named visible "
            "finding; REJECT if BASE is also plausible, if the change is just a different view/plane/normality "
            "interpretation, or if the evidence is subtle."
        )
    elif mode == "chartqa":
        policy = (
            "ChartQA needs answer-type fidelity. ALLOW only for direct legend/axis/OCR/bar evidence or an explicit "
            "calculation, and only when the challenger answer type matches what the question asks."
        )
    else:
        policy = (
            "ALLOW only when the challenger cites direct image/text evidence that makes BASE wrong or incomplete. "
            "REJECT speculative visual guesses."
        )
    return f"""
You are a second-pass safety verifier for VQA residual selection.

Dataset mode: {mode}
Safety policy: {policy}

Question:
{_clip(problem, 1800)}

BASE answer:
{_clip(base_answer, 400)}
BASE evidence:
{_clip(base_raw, 1200)}

CHALLENGER answer:
{_clip(challenger_answer, 400)}
CHALLENGER evidence:
{_clip(challenger_raw, 1200)}

Initial selector reason:
{_clip(selector_reason, 500)}

Return ALLOW only if replacing BASE is safer than keeping it. If both answers are plausible, return REJECT.
Return strict JSON only:
{{"verdict":"ALLOW|REJECT", "confidence":"LOW|MEDIUM|HIGH", "reason":"one short sentence"}}
""".strip()


async def _vqa_second_pass(client, problem: str, mode: str,
                           base_answer: str, base_raw: str,
                           challenger_answer: str, challenger_raw: str,
                           selector_reason: str, image: Optional[str]) -> Dict[str, str]:
    prompt = _vqa_safety_prompt(problem, mode, base_answer, base_raw, challenger_answer, challenger_raw, selector_reason)
    raw = await async_call_llm(
        client,
        prompt,
        temperature=0.0,
        max_tokens=384,
        use_ds_api=True,
        images=[image] if image else None,
    )
    parsed = _parse_selector_json(raw.replace('"verdict"', '"choice"').replace("ALLOW", "USE_CHALLENGER").replace("REJECT", "KEEP_BASE"))
    parsed["verdict"] = "ALLOW" if parsed.get("choice") == "USE_CHALLENGER" else "REJECT"
    parsed["safety_raw"] = raw
    return parsed


async def select_residual_answer(client, problem: str, task_type: TaskType,
                                 base_answer: str, base_raw: str,
                                 challenger_answer: str, challenger_raw: str,
                                 image: Optional[str] = None) -> Dict[str, Any]:
    """Select final answer with a selective residual gate.

    BASE is preserved by default, but VQA allows strong evidence corrections
    instead of the over-conservative v11.1 guard.
    """
    base_norm = _norm(base_answer, task_type)
    challenger_norm = _norm(challenger_answer, task_type)
    if base_norm and base_norm == challenger_norm:
        return {
            "choice": "SAME_ANSWER",
            "confidence": "HIGH",
            "reason": "Base and challenger have the same normalized answer.",
            "final_answer": base_answer or challenger_answer,
            "used_challenger": False,
            "selector_called": False,
        }

    verified = _choice_verifier(task_type, base_answer, base_raw, challenger_answer, challenger_raw)
    if verified is not None:
        return verified

    prompt = _selector_prompt(problem, task_type, base_answer, base_raw, challenger_answer, challenger_raw)
    raw = await async_call_llm(
        client,
        prompt,
        temperature=0.0,
        max_tokens=512,
        use_ds_api=True,
        images=[image] if image and task_type in (TaskType.VQA_OPEN, TaskType.VQA_CHOICE) else None,
    )
    parsed = _parse_selector_json(raw)
    choice = parsed["choice"]
    confidence = parsed["confidence"]
    is_vqa = task_type in (TaskType.VQA_OPEN, TaskType.VQA_CHOICE)
    mode = _vqa_mode() if is_vqa else ""
    guard_reason = _vqa_hard_reject(problem, mode, base_answer, challenger_answer, parsed) if is_vqa else None
    if guard_reason:
        parsed = {
            **parsed,
            "choice": "KEEP_BASE",
            "confidence": "LOW",
            "reason": f"v11.1 VQA guard kept BASE: {guard_reason}.",
            "guard_reason": guard_reason,
        }
        choice = parsed["choice"]
        confidence = parsed["confidence"]

    use_challenger = (
        choice == "USE_CHALLENGER" and
        _CONF_RANK.get(confidence, 0) >= _min_confidence_rank()
    )
    if use_challenger and is_vqa and os.environ.get("V11_1_VQA_VERIFY", "0") == "1" and mode in {"vqarad", "slake"}:
        safety = await _vqa_second_pass(
            client, problem, mode,
            base_answer, base_raw, challenger_answer, challenger_raw,
            parsed.get("reason", ""), image,
        )
        parsed["v11_1_safety"] = safety
        if safety.get("verdict") != "ALLOW" or _CONF_RANK.get(safety.get("confidence", "LOW"), 0) < _CONF_RANK["HIGH"]:
            parsed = {
                **parsed,
                "choice": "KEEP_BASE",
                "confidence": safety.get("confidence", "LOW"),
                "reason": f"v11.1 safety verifier kept BASE: {safety.get('reason', '')}",
            }
            use_challenger = False

    final_answer = challenger_answer if use_challenger else base_answer
    if task_type == TaskType.CODE:
        selected_raw = challenger_raw if use_challenger else base_raw
        selected_prog = humaneval_program_from_fields(problem, final_answer or "", selected_raw or "")
        challenger_prog = humaneval_program_from_fields(problem, challenger_answer or "", challenger_raw or "")
        base_prog = humaneval_program_from_fields(problem, base_answer or "", base_raw or "")
        if not selected_prog and challenger_prog:
            use_challenger = True
            final_answer = challenger_answer or extract_output(challenger_raw or "", TaskType.CODE)
        elif not selected_prog and base_prog:
            use_challenger = False
            final_answer = base_answer or extract_output(base_raw or "", TaskType.CODE)
        elif not use_challenger and base_prog and not challenger_prog:
            final_answer = base_answer or extract_output(base_raw or "", TaskType.CODE)
        elif use_challenger:
            final_answer = challenger_answer or extract_output(challenger_raw or "", TaskType.CODE)
        else:
            final_answer = base_answer or extract_output(base_raw or "", TaskType.CODE)

    return {
        **parsed,
        "selector_raw": raw,
        "final_answer": final_answer,
        "used_challenger": use_challenger,
        "selector_called": True,
    }
