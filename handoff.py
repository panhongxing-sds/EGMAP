from typing import Any, Dict, Iterable, List, Tuple

from config import AgentType, TaskType


def edge_key(src: int, dst: int) -> str:
    return f"{src}->{dst}"


def parse_edge_key(key: str) -> Tuple[int, int]:
    left, right = key.split("->", 1)
    return int(left), int(right)


def _agent_name(agent: Any) -> str:
    agent_type = getattr(agent, "type", None)
    if isinstance(agent_type, AgentType):
        return agent_type.value
    return str(agent_type or "agent")


def _task_fields(task_type: TaskType) -> List[str]:
    if task_type == TaskType.CODE:
        return [
            "key_result_or_code_intent",
            "evidence_or_tests_considered",
            "confidence",
        ]
    if task_type in (TaskType.MATH_CHOICE, TaskType.REASONING_CHOICE):
        return [
            "key_result",
            "selected_option_if_any",
            "confidence",
        ]
    if task_type == TaskType.VQA_OPEN:
        return [
            "key_result",
            "visual_evidence",          # what was seen in the image: relevant regions, modality, structures, colors, counts
            "confidence",               # high/medium/low — downstream preserves high-confidence answers
        ]
    if task_type == TaskType.VQA_CHOICE:
        return [
            "key_result",
            "selected_option_if_any",
            "visual_evidence",          # image regions, modality, structures, colors, counts supporting the choice
            "image_regions_examined",
            "confidence",
            "missing_information",
        ]
    return [
        "key_result",
        "supporting_steps",
        "confidence",
    ]


def default_handoff_for_edge(src: int, dst: int, src_agent: Any, dst_agent: Any,
                             task_type: TaskType) -> str:
    fields = ", ".join(_task_fields(task_type))
    if task_type in (TaskType.VQA_OPEN, TaskType.VQA_CHOICE):
        return (
            f"Edge Agent-{src}({_agent_name(src_agent)}) -> Agent-{dst}({_agent_name(dst_agent)}).\n"
            "Sender rule: hand off a compact block — the answer (key_result), the specific visual_evidence "
            "actually seen in the image, and your confidence (high/medium/low).\n"
            "Receiver rule: the upstream answer is usually correct. If sender confidence is high and the "
            "visual_evidence is consistent with what the image shows, PRESERVE the answer unchanged. Only revise "
            "when the image clearly and unambiguously contradicts it; never flip a yes/no answer without clear "
            "visual proof.\n"
            f"Required fields: {fields}."
        )
    # v4 task-specific receiver rules (MASPO-style per-task design):
    # CODE / REASONING_CHOICE need active re-derivation → independent verification
    # MATH / MATH_CHOICE: baseline already high + deterministic answer → trust high confidence
    if task_type == TaskType.CODE:
        return (
            f"Edge Agent-{src}({_agent_name(src_agent)}) -> Agent-{dst}({_agent_name(dst_agent)}).\n"
            "Sender rule: hand off a compact block — the key_result_or_code_intent (your final Python code), "
            "the evidence_or_tests_considered (which docstring examples you mentally verified), and your "
            "confidence (high/medium/low).\n"
            "Receiver rule: You MUST independently trace each example in the docstring through the upstream "
            "code. For each example, simulate inputs through the code and check whether the output matches "
            "expected. ONLY preserve the code if ALL examples pass. If any example fails, identify the specific "
            "bug (off-by-one, wrong operator, missing case, incorrect data structure) and provide a corrected "
            "version. Do NOT blindly trust the upstream confidence — it is unreliable on code.\n"
            f"Required fields: {fields}."
        )
    if task_type == TaskType.REASONING_CHOICE:
        return (
            f"Edge Agent-{src}({_agent_name(src_agent)}) -> Agent-{dst}({_agent_name(dst_agent)}).\n"
            "Sender rule: hand off a compact block — the key_result (the selected option letter), the "
            "selected_option_if_any, and your confidence (high/medium/low).\n"
            "Receiver rule: You MUST independently solve the problem from first principles BEFORE looking at "
            "the upstream selected_option. Apply the relevant scientific principles step by step to derive "
            "your own answer. ONLY preserve the upstream answer if your independent derivation arrives at the "
            "same option. If you arrive at a different option, identify which scientific principle is being "
            "misapplied and revise. Do NOT blindly trust the upstream confidence — it is unreliable on multi-step reasoning.\n"
            f"Required fields: {fields}."
        )
    # MATH / MATH_CHOICE: trust high-confidence (avoid over-correction on already-strong baseline)
    return (
        f"Edge Agent-{src}({_agent_name(src_agent)}) -> Agent-{dst}({_agent_name(dst_agent)}).\n"
        "Sender rule: hand off a compact block — the answer (key_result or selected_option_if_any), the core "
        "supporting reasoning/evidence (mathematical steps), and your confidence (high/medium/low).\n"
        "Receiver rule: the upstream answer is usually correct. If sender confidence is high and the reasoning "
        "aligns, PRESERVE the answer unchanged. Only revise when there is a clear logical or computational error "
        "in the reasoning; for multiple-choice, do NOT change the selected option without identifying a specific "
        "error.\n"
        f"Required fields: {fields}."
    )


def build_default_handoff_map(edges: Dict[int, List[int]], agents: List[Any],
                              task_type: TaskType) -> Dict[str, str]:
    handoffs: Dict[str, str] = {}
    for src, dsts in edges.items():
        for dst in dsts:
            if src < len(agents) and dst < len(agents):
                handoffs[edge_key(src, dst)] = default_handoff_for_edge(
                    src, dst, agents[src], agents[dst], task_type
                )
    return handoffs


def normalize_handoff_map(raw: Dict[Any, Any]) -> Dict[str, str]:
    normalized: Dict[str, str] = {}
    for key, value in (raw or {}).items():
        normalized[str(key)] = str(value)
    return normalized


def format_sender_guidance(agent_id: int, successors: Iterable[int],
                           handoff_map: Dict[str, str]) -> str:
    blocks = []
    for dst in successors:
        instruction = handoff_map.get(edge_key(agent_id, dst))
        if instruction:
            blocks.append(f"[Sender handoff for Agent-{agent_id} -> Agent-{dst}]\n{instruction}")
    if not blocks:
        return ""
    return (
        "[Inter-Agent Handoff Requirements]\n"
        "Your answer will be consumed by downstream agents. Follow these sender rules exactly:\n"
        + "\n\n".join(blocks)
    )


def format_receiver_context(src: int, dst: int, message: str,
                            handoff_map: Dict[str, str]) -> str:
    instruction = handoff_map.get(edge_key(src, dst), "")
    return (
        f"[HANDOFF Agent-{src} -> Agent-{dst}]\n"
        f"Interface contract:\n{instruction or '(no explicit contract)'}\n\n"
        f"Upstream message:\n{message or '(empty)'}\n"
        "[/HANDOFF]"
    )


def sanitize_handoff(text: str, fallback: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return fallback
    if len(cleaned.split()) < 12:
        return fallback
    return cleaned[:2400]


HANDOFF_OPTIMIZE_TEMPLATE = """
You are optimizing an inter-agent handoff interface in an LLM-based multi-agent system.
This is NOT a node prompt rewrite. Optimize the edge contract: what the sender must pass and how the receiver must consume or reject it.

Edge: Agent-{src_id} ({src_type}) -> Agent-{dst_id} ({dst_type})

[NODE-AWARE CONTEXT] The handoff must be coherent with the actual prompts that the sender and receiver are currently using. Do not introduce fields the agents cannot produce/consume given these prompts.

Current sender (Agent-{src_id}) prompt:
```
{src_prompt}
```

Current receiver (Agent-{dst_id}) prompt:
```
{dst_prompt}
```

Current handoff:
```
{handoff}
```

Sample execution traces for this edge:
```
{samples}
```

Repair the handoff using paired instructions:
1. Sender rule: required fields the upstream agent must provide. These fields must be naturally producible by the sender prompt above.
2. Receiver rule: how the downstream agent should use, verify, or reject the handoff. The verification logic must be executable given the receiver prompt above.
3. Keep it compact and task-agnostic. Do not add new agents or change topology.
4. Prefer actionable fields such as evidence, confidence, missing_information, assumptions, and final_result.
5. Co-design with prompts: if the sender/receiver prompts already cover certain behaviors, the handoff should leverage them rather than duplicate. If the prompts miss something critical (e.g., format), the handoff should enforce it.

Return exactly:
<analyse>Briefly diagnose the handoff weakness and how it should align with the current sender/receiver prompts.</analyse>
<handoff>The complete optimized handoff interface.</handoff>
"""
