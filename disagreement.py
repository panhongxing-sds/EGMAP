import os
import re
from collections import defaultdict
from typing import Dict, Iterable, List

from config import TaskType
from utils import extract_output, normalize_answer


def _norm_answer(answer: str, task_type: TaskType) -> str:
    text = (answer or "").strip()
    if not text:
        return ""
    text = _strip_answer_label(text)
    if task_type in (TaskType.VQA_OPEN, TaskType.VQA_CHOICE):
        text = re.sub(r"</?answer>", "", text.lower())
        text = re.sub(r"[^a-z0-9一-鿿 ]", " ", text)
        return re.sub(r"\s+", " ", text).strip()
    if task_type == TaskType.CODE:
        return re.sub(r"\s+", "", text)[:200]
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


def _clip(text: str, limit: int = 700) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + " ..."


def _task_receiver_rule(task_type: TaskType) -> str:
    if task_type in (TaskType.VQA_OPEN, TaskType.VQA_CHOICE):
        return (
            "Receiver rule: resolve disagreement by checking visual evidence first. "
            "Prefer the candidate whose stated evidence is actually visible in the image; "
            "do not majority-vote if the minority answer has stronger visual grounding."
        )
    if task_type == TaskType.CODE:
        return (
            "Receiver rule: resolve disagreement by mentally executing the candidates on the examples/tests. "
            "Prefer code that satisfies the observed constraints; do not choose by verbosity."
        )
    if task_type in (TaskType.MATH_CHOICE, TaskType.REASONING_CHOICE):
        return (
            "Receiver rule: resolve disagreement by re-checking the decisive reasoning step or option evidence. "
            "Prefer the answer with an explicit, non-contradictory derivation."
        )
    return (
        "Receiver rule: resolve disagreement by verifying the decisive calculation/reasoning step. "
        "Prefer the candidate with explicit support, not simply the most verbose response."
    )


def _task_sequential_rule(task_type: TaskType) -> str:
    if task_type in (TaskType.VQA_OPEN, TaskType.VQA_CHOICE):
        return (
            "Verification rule: check the predecessor's answer against the image first. "
            "Keep the predecessor answer if the visual evidence supports it; correct it only when you can identify "
            "a concrete visual contradiction or missing visual evidence."
        )
    if task_type == TaskType.CODE:
        return (
            "Verification rule: mentally execute the predecessor solution on the examples/tests and constraints. "
            "Keep it if no concrete failing case is found; correct it only when a specific bug or violated constraint is identified."
        )
    if task_type in (TaskType.MATH_CHOICE, TaskType.REASONING_CHOICE):
        return (
            "Verification rule: re-check the decisive reasoning step and option evidence. "
            "Keep the predecessor answer if no explicit contradiction is found; correct it only with a concrete reason."
        )
    return (
        "Verification rule: re-check the decisive calculation/reasoning step. "
        "Keep the predecessor answer if no concrete error is found; correct it only with a specific arithmetic or logic error."
    )


def _force_report() -> bool:
    return os.environ.get("V10_ALWAYS_DISAGREE_REPORT", "").strip().lower() in {
        "1", "true", "yes", "on"
    }


def _enable_sequential_verify() -> bool:
    return os.environ.get("V10_SEQUENTIAL_VERIFY", "1").strip().lower() not in {
        "0", "false", "no", "off"
    }


def format_disagreement_context(agent_id: int, predecessors: Iterable[int],
                                upstream_outputs: Dict[int, str],
                                task_type: TaskType) -> str:
    """Build an adaptive disagreement report for multi-input receivers.

    v10 keeps the topology unchanged. It injects extra context only when
    upstream agents produce conflicting extracted answers, avoiding consensus
    cases where an extra arbitration block just adds noise.
    """
    rows: List[Dict[str, str]] = []
    buckets: Dict[str, List[int]] = defaultdict(list)
    for pred_id in predecessors:
        raw = upstream_outputs.get(pred_id, "")
        if not raw:
            continue
        answer = extract_output(raw, task_type) or raw
        norm = _norm_answer(answer, task_type)
        rows.append({
            "pred_id": str(pred_id),
            "answer": _clip(answer, 180),
            "norm": norm,
            "snippet": _clip(raw, 520),
        })
        if norm:
            buckets[norm].append(pred_id)

    if len(rows) < 2:
        return ""

    nonempty = [key for key in buckets if key]
    disagreement = len(nonempty) > 1
    if not disagreement and not _force_report():
        return ""

    status = "DISAGREEMENT" if disagreement else "CONSENSUS"
    bucket_text = "; ".join(
        f"{repr(key)} from agents {ids}" for key, ids in sorted(buckets.items(), key=lambda kv: (-len(kv[1]), kv[0])) if key
    ) or "no extractable final answers"

    lines = [
        f"[ADAPTIVE DISAGREEMENT HANDOFF for Agent-{agent_id}]",
        f"Status: {status}",
        f"Candidate answer groups: {bucket_text}",
        _task_receiver_rule(task_type),
        "Upstream candidate evidence:",
    ]
    for row in rows:
        lines.append(
            f"- Agent-{row['pred_id']}: candidate_answer={row['answer']!r}; "
            f"evidence_or_reasoning_snippet={row['snippet']!r}"
        )
    lines.append(
        "Decision instruction: explicitly compare the candidates above, identify the decisive evidence or error, "
        "then output one final answer in the task's required format."
    )
    lines.append("[/ADAPTIVE DISAGREEMENT HANDOFF]")
    return "\n".join(lines)


def format_sequential_verification_context(agent_id: int, pred_id: int,
                                           upstream_outputs: Dict[int, str],
                                           task_type: TaskType) -> str:
    """Build a compact trust-but-verify handoff for single-chain receivers.

    This makes v10 useful for reflect/sequential topologies too. The block is
    conservative by design: it tells the receiver to preserve the predecessor's
    answer unless it can point to a concrete visual, logical, arithmetic, or
    test-based error.
    """
    if not _enable_sequential_verify():
        return ""

    raw = upstream_outputs.get(pred_id, "")
    if not raw:
        return ""

    answer = extract_output(raw, task_type) or raw
    lines = [
        f"[ADAPTIVE SEQUENTIAL VERIFICATION for Agent-{agent_id}]",
        f"Predecessor: Agent-{pred_id}",
        f"Predecessor candidate answer: {_clip(answer, 180)!r}",
        _task_sequential_rule(task_type),
        f"Predecessor evidence/reasoning snippet: {_clip(raw, 620)!r}",
        "Decision instruction: first verify the predecessor's candidate. Preserve it when it is supported; "
        "revise only after locating a concrete error or stronger evidence. Then output the final answer in the required format.",
        "[/ADAPTIVE SEQUENTIAL VERIFICATION]",
    ]
    return "\n".join(lines)
