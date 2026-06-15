"""
Structured meta-prompt templates (4-stage diagnostic design).

These templates strengthen the official MASPO PROMPT_OPTIMIZE_TEMPLATE by enforcing
a structured diagnose-then-fix workflow:
    1. Group failed vs passed traces
    2. Categorize observed failures into a taxonomy
    3. Trace each failure to a specific deficiency in the reference prompt
    4. Propose a minimal targeted edit with anti-regression check + confidence

The placeholders are identical to PROMPT_OPTIMIZE_TEMPLATE
({agent_type}, {role_description}, {samples}, {requirements}, {prompt}) so this
template can be swapped in via .format(**same_kwargs).

Activate via --structured-meta-prompt CLI flag in run_maspo.py.
"""

from config import TaskType


def _build(task_domain: str, visual: bool = False) -> str:
    visual_note = ""
    if visual:
        visual_note = (
            "[VISUAL CONTEXT] You are ALSO shown the actual image(s) for the sample(s) below — for BOTH the "
            "failed and the passed traces. Use them in two directions:\n"
            "(a) Diagnose failures by a three-way comparison — what the image actually shows vs. what the agent "
            "answered vs. the requirement/gold: decide whether the agent missed the relevant region, misread it "
            "(wrong imaging modality, structure, color, count, or spatial relation), or saw it correctly but "
            "reasoned wrong. Tie each failure to a concrete visual cause.\n"
            "(b) Protect what already works (anti-regression): ALSO look at the images of the PASSED traces and "
            "make sure your proposed edit does NOT break the agent's correct visual reading on those. Keep the "
            "prompt's visual guidance general — do NOT overfit it to the failed cases or hard-code answers.\n"
            "Then steer the optimized prompt to attend to the right visual evidence the question depends on, "
            "grounded in what the images actually show, not just the text.\n\n"
        )
    return f"""You are optimizing a prompt for a specific agent in a multi-agent {task_domain} system.
CRITICAL: The agent's core role and responsibilities MUST be preserved in the optimized prompt.

{visual_note}Agent Type: {{agent_type}}
Current System Role: {{role_description}}

[STAGE 1 — STRUCTURED EVIDENCE]
Sample Execution Traces (Question + Context + Agent Output):
```
{{samples}}
```

Requirements:
```
{{requirements}}
```

Reference prompt:
```
{{prompt}}
```

[STAGE 2 — DIAGNOSE]
First, partition the traces above:
- failed_traces  = samples where the Agent Output does NOT satisfy the Requirements
- passed_traces  = samples where it DOES
Then induce 3 to 5 error categories that best describe the failed_traces. You may
draw from common patterns such as:
    format_violation     — output format / extraction tag missing or malformed
    reasoning_gap        — wrong intermediate steps, skipped sub-problems
    output_extraction    — final answer present but not in expected location/style
    boundary             — edge cases, unit conversions, sign/scale errors
    coordination         — does not fit downstream agent's expected interface
    hallucination        — confidently asserted but unsupported facts/values
    visual_grounding     — misreading or misaligning visual content (for VQA)
However, you should INVENT task-specific categories when the failures genuinely
do not match the above (e.g., "spatial_relation_error" for vision tasks,
"library_misuse" for code tasks, "tool_selection_error" for agentic tasks).
Avoid a generic "other" bucket — name the actual pattern.
Report the most frequent error category.

[STAGE 3 — ROOT CAUSE]
For the most frequent failure category, identify the SPECIFIC deficiency in the
reference prompt that allowed this error class to slip through. Quote (or
paraphrase) the offending clause, OR state explicitly what is MISSING.

[STAGE 4 — MINIMAL TARGETED EDIT]
Propose a TARGETED change that addresses the root cause:
- Choose ONE edit axis: clarity | format_specificity | reasoning_structure | boundary_handling | coordination
- Preserve at least 70 percent of the reference prompt's wording
- Do NOT rewrite from scratch — surgical, minimal change
- Do NOT add new agents, new roles, or alter topology

[STAGE 5 — ANTI-REGRESSION CHECK]
Mentally re-run the passed_traces against your proposed prompt. If any previously
successful sample is now at risk of failing, revise the edit so the success path
is preserved. Briefly state which passed samples you re-validated.

[STAGE 6 — CONFIDENCE]
Rate your confidence that this edit improves over the reference (0.0 - 1.0).
If confidence < 0.4, return the reference prompt UNCHANGED inside <prompt>.

Return exactly the following XML structure:
<analyse>
<failed_summary>Top failed traces and their error tags (one short line each).</failed_summary>
<error_taxonomy>Frequency-ordered list of observed error categories.</error_taxonomy>
<root_cause>Specific clause or omission in the reference prompt causing the top failure category.</root_cause>
<regression_check>Brief note on which passed samples remain safe under the new prompt.</regression_check>
</analyse>
<edit_axis>One of: clarity | format_specificity | reasoning_structure | boundary_handling | coordination</edit_axis>
<modification>One sentence summarizing the targeted change.</modification>
<confidence>Float between 0.0 and 1.0</confidence>
<prompt>The complete optimized prompt. Keep at least 70 percent overlap with the reference. Preserve the core role description verbatim or near-verbatim.</prompt>
"""


PROMPT_OPTIMIZE_TEMPLATE_STRUCTURED = {
    TaskType.MATH: _build("mathematical reasoning"),
    TaskType.CODE: _build("code generation"),
    TaskType.MATH_CHOICE: _build("multiple-choice mathematical reasoning"),
    TaskType.REASONING_CHOICE: _build("multiple-choice reasoning"),
    TaskType.VQA_OPEN: _build("visual question answering", visual=True),
    TaskType.VQA_CHOICE: _build("visual multiple-choice question answering", visual=True),
}
