import json
import re
import html
import asyncio
import subprocess
import tempfile
import os
from typing import List, Dict, Any, Optional
from collections import Counter

from config import TaskType

DS_API_CONCURRENCY_LIMIT = 60 
ds_semaphore = asyncio.Semaphore(DS_API_CONCURRENCY_LIMIT)


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _truncate_prompt_for_vllm(prompt: str, use_ds_api: bool) -> str:
    env_name = "MASPO_STRONG_MAX_PROMPT_CHARS" if use_ds_api else "MASPO_WORK_MAX_PROMPT_CHARS"
    # Default 0 = no truncation (matches official MASPO behavior)
    limit = _int_env(env_name, 0)
    if limit <= 0 or len(prompt) <= limit:
        return prompt

    marker = "\n\n[... middle truncated to fit local vLLM context window ...]\n\n"
    budget = max(0, limit - len(marker))
    head = int(budget * 0.6)
    tail = budget - head
    return (
        prompt[:head]
        + marker
        + prompt[-tail:]
    )


def async_retry(tries: int = 5, delay: float = 0.5, max_delay: float = 30):
    def deco(func):
        async def wrapper(*args, **kwargs):
            d = delay
            for i in range(1, tries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception:
                    if i == tries:
                        raise
                    await asyncio.sleep(min(max_delay, d))
                    d *= 2
            return None
        return wrapper
    return deco

@async_retry()
async def async_call_llm(client, prompt: str, temperature: float = 0.0,
                         max_tokens: int = 4096, use_ds_api: bool = False,
                         images: Optional[List[str]] = None) -> str:
    model = os.environ.get(
        "MASPO_EVALUATOR_MODEL" if use_ds_api else "MASPO_MODEL",
        "Qwen/Qwen3.5-9B" if use_ds_api else "Qwen/Qwen3.5-4B",
    )
    token_env = "MASPO_STRONG_MAX_TOKENS" if use_ds_api else "MASPO_WORK_MAX_TOKENS"
    token_cap = _int_env(token_env, 10000 if use_ds_api else 4096)
    if token_cap > 0:
        if use_ds_api:
            max_tokens = min(max_tokens, token_cap)
        else:
            # Work-model calls use the env value as the default generation budget.
            max_tokens = token_cap if token_env in os.environ else min(max_tokens, token_cap)
    prompt = _truncate_prompt_for_vllm(prompt, use_ds_api)

    # Multimodal auto-switch: if images are provided, build an OpenAI vision
    # content list; otherwise keep the exact text-only message (byte-for-byte
    # identical to the original behaviour for all text datasets).
    if images:
        content = [{"type": "text", "text": prompt}]
        for b64 in images:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })
        messages = [{"role": "user", "content": content}]
    else:
        messages = [{"role": "user", "content": prompt}]

    request_kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "extra_body": {"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}},
    }

    try:
        if use_ds_api:
            async with ds_semaphore:
                resp = await client.chat.completions.create(** request_kwargs)
        else:
            resp = await client.chat.completions.create(** request_kwargs)
    except Exception as e:
        # 400 BadRequestError (context overflow etc.): skip and return empty string
        msg = str(e)
        if "BadRequestError" in type(e).__name__ or "400" in msg or "context length" in msg.lower():
            print(f"[WARN] LLM 400 error, skipping this call: {msg[:150]}")
            return ""
        raise

    first_choice = resp.choices[0]

    text = first_choice.message.content or ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def _norm_choice_letter(text: str) -> Optional[str]:
    match = re.fullmatch(r"\(?\s*([A-Da-d])\s*\)?[.:]?\s*", str(text or "").strip())
    return match.group(1).upper() if match else None


def _extract_handoff_json_result(raw: str) -> Optional[str]:
    """Parse handoff-style JSON blocks for final_result (GPQA / choice tasks)."""
    for block in re.findall(r"```json\s*(\{.*?\})\s*```", raw, flags=re.I | re.S):
        try:
            obj = json.loads(block)
            val = obj.get("final_result") or obj.get("selected_option_if_any") or obj.get("key_result")
            if val:
                return str(val).strip()
        except Exception:
            pass

    for mobj in re.finditer(r"\{[^{}]*\}", raw, flags=re.S):
        try:
            obj = json.loads(mobj.group(0))
            val = obj.get("final_result") or obj.get("selected_option_if_any")
            if val:
                return str(val).strip()
        except Exception:
            continue
    return None


def _extract_option_from_text(raw: str) -> Optional[str]:
    """Extract GPQA-style Option X / final-answer patterns from model tail."""
    tail = raw[-800:]
    patterns = [
        r"(?:final\s+answer|correct\s+answer|answer|option|choice)\s*(?:is|:)?\s*[*\s]*\(?\s*([A-D])\s*\)?\b",
        r"corresponds\s+to\s+option\s*\(?\s*([A-D])\s*\)?\b",
        r"\bOption\s+([A-D])\b",
        r"\"final_result\"\s*:\s*\"Option\s+([A-D])\"",
        r"\"selected_option_if_any\"\s*:\s*\"([A-D])\"",
    ]
    hits = []
    for pattern in patterns:
        hits.extend(re.findall(pattern, tail, flags=re.I))
    if hits:
        return hits[-1].upper()
    return None


def extract_answer(raw: str) -> str:
    raw = raw.strip()
    for _ in range(3):
        new_raw = html.unescape(raw)
        if new_raw == raw:
            break
        raw = new_raw

    matches = re.findall(r"<answer>(.*?)</answer>", raw, re.S)
    if matches:
        return matches[-1].strip()

    # Handle unclosed <answer> tag (output truncated before </answer>)
    m = re.search(r"<answer>([^<]{0,300})$", raw.strip(), re.S)
    if m:
        val = m.group(1).strip()
        if val and val not in ("UNVERIFIED", "UNDEFINED"):
            return val

    box_pat = re.compile(r"\\boxed\s*\{((?:[^{}]|\{[^{}]*\})*)\}", re.S)
    m = box_pat.search(raw)
    if m:
        boxed = m.group(1).strip()
        choice = _norm_choice_letter(boxed)
        if choice:
            return choice
        return boxed

    handoff = _extract_handoff_json_result(raw)
    if handoff:
        choice = _norm_choice_letter(handoff)
        if choice:
            return choice
        opt = _extract_option_from_text(handoff)
        if opt:
            return opt
        return handoff

    option = _extract_option_from_text(raw)
    if option:
        return option

    sentences = re.split(r'[。\n;]+', raw)
    last = sentences[-1].strip()
    return last[-30:] if len(last) > 30 else last

def extract_code(raw: str) -> str:
    raw = raw.strip()

    candidates = []
    seen = set()

    python_pattern = r'```\s*python\s*(.*?)\s*```'
    for m in re.findall(python_pattern, raw, flags=re.DOTALL | re.IGNORECASE):
        code = m.strip()
        if code and code not in seen:
            seen.add(code)
            candidates.append(code)

    general_pattern = r'```\s*(.*?)\s*```'
    for m in re.findall(general_pattern, raw, flags=re.DOTALL):
        code = m.strip()
        if code and code not in seen:
            code = re.sub(r'^python\s*', '', code, flags=re.IGNORECASE).strip()
            seen.add(code)
            candidates.append(code)

    code_tag_pattern = r'<code>(.*?)</code>'
    for m in re.findall(code_tag_pattern, raw, flags=re.DOTALL | re.IGNORECASE):
        code = m.strip()
        if code and code not in seen:
            seen.add(code)
            candidates.append(code)

    for code in reversed(candidates):
        if 'def ' in code:
            return code

    if 'def ' in raw:
        def_pattern = r'def\s+[\w_]+\(.*?\):\s*\n?(.|\n)*?(?=\n\s*\n|```|\Z)'
        def_match = re.search(def_pattern, raw, flags=re.IGNORECASE)
        if def_match:
            return def_match.group(0).strip()

    return ''

def extract_output(raw: str, task_type: TaskType) -> str:
    if task_type == TaskType.CODE:
        return extract_code(raw)
    else:
        return extract_answer(raw)


def assemble_humaneval_program(prompt: str, model_text: str) -> str:
    """Standard HumanEval scoring: prompt stub + model completion."""
    prompt = (prompt or "").rstrip()
    text = (model_text or "").strip()
    if not text:
        return ""
    code = extract_code(text)
    if not code:
        if re.search(r"^\s*def \w+", text, re.M):
            return prompt + text
        # Body-only completion without fences (standard HumanEval protocol).
        if text and not re.search(r"(^|\n)\s*#|STAGE|^\*\*|^```", text, re.I | re.M):
            return prompt + text
        return ""
    if re.search(r"(?:^|\n)(?:import |from )", code):
        return code
    if code.lstrip().startswith("def "):
        imports = "\n".join(
            line for line in prompt.splitlines() if line.startswith(("import ", "from "))
        )
        return f"{imports}\n\n{code}" if imports else code
    return prompt + code


def humaneval_program_from_fields(prompt: str, answer: str = "", raw: str = "") -> str:
    """Prefer the full model trace over the compressed short answer."""
    for text in (raw, answer):
        program = assemble_humaneval_program(prompt, text)
        if program and "def " in program:
            return program
    return ""

def normalize_answer(answer: str) -> str:
    answer = answer.strip()
    answer = re.sub(r'\$(.*?)\$', r'\1', answer)
    answer = re.sub(r'\\\[(.*?)\\\]', r'\1', answer, flags=re.S)
    answer = re.sub(r'\\text\{([^}]*)\}', r'\1', answer)
    answer = re.sub(r'\\boxed\s*{((?:[^{}]|{[^}]*})*?)}', r'\1', answer)
    answer = re.sub(r'\\\((.*?)\\\)', r'\1', answer)
    answer = re.sub(r'\\?°', '', answer)
    answer = re.sub(r'\^?\\?circ', '', answer)
    answer = re.sub(r'\s+', '', answer)
    answer = re.sub(r'\\sqrt\s*{([^}]*)}', r'sqrt(\1)', answer)
    answer = re.sub(r'√(\d+)', r'sqrt(\1)', answer)
    answer = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'\1/\2', answer)
    answer = re.sub(r'\\dfrac\{([^}]+)\}\{([^}]+)\}', r'\1/\2', answer)
    answer = re.sub(r'\\pi', 'π', answer)
    answer = re.sub(r'(?<![a-z])pi(?![a-z])', 'π', answer, flags=re.I)
    answer = re.sub(r'\\left|\\right', '', answer)
    answer = answer.replace('[', '(').replace(']', ')')
    if re.fullmatch(r'[,\s\-0-9]+', answer):
        nums = [int(x) for x in re.findall(r'-?\d+', answer)]
        return ','.join(map(str, sorted(nums)))
    return answer.lower()


# ===== Math-aware lenient equivalence (for MATH task type) =====
# Handles fraction <-> decimal, LaTeX matrix <-> comma list, unit stripping.
from fractions import Fraction as _Fraction


def _lenient_strip(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = re.sub(r"\\begin\{(pmatrix|bmatrix|matrix|array)\}|\\end\{(pmatrix|bmatrix|matrix|array)\}", "", s)
    s = re.sub(r"\\boxed\s*\{([^{}]*(?:\{[^{}]*\})*[^{}]*)\}", r"\1", s)
    s = re.sub(r"\\dfrac|\\frac|\\tfrac", "", s)
    s = re.sub(r"\\left|\\right", "", s)
    s = re.sub(r"\\,|\\ |\\;|\\!", "", s)
    s = re.sub(r"\\\\", ",", s)
    s = s.replace("$", "")
    # crude unit stripping: numbers followed by alphabetic units
    s = re.sub(r"(\d(?:\.\d+)?)\s*(?:cm|mm|km|kg|mg|sec|hr|min|degrees?|deg|%|°)\b",
               r"\1", s, flags=re.IGNORECASE)
    s = re.sub(r"[\s\{\}\(\)\[\]]+", "", s)
    return s.lower()


def _to_fraction(part: str):
    if not part:
        return None
    try:
        if "/" in part:
            num, den = part.split("/", 1)
            return _Fraction(num.strip()) / _Fraction(den.strip())
        if "." in part:
            return _Fraction(part).limit_denominator(10**6)
        return _Fraction(part)
    except Exception:
        return None


def math_equivalent(model_output: str, correct_answer: str) -> bool:
    """Lenient equivalence for math answers. Used for MATH (free-form) task type
    to handle fraction <-> decimal, LaTeX matrix <-> comma list, unit stripping.
    Always called AFTER strict string compare fails — additional ways to
    accept a mathematically-equivalent answer."""
    if model_output is None or correct_answer is None:
        return False
    a = _lenient_strip(model_output)
    b = _lenient_strip(correct_answer)
    if not a or not b:
        return False
    if a == b:
        return True
    # (15,-29) vs 15/-29 style coordinate pairs
    slash_m = re.fullmatch(r"(-?\d+)/(-?\d+)", a)
    if slash_m:
        a_slash = f"({slash_m.group(1)},{slash_m.group(2)})"
        if normalize_answer(a_slash) == normalize_answer(b):
            return True
    a_parts = [p for p in re.split(r"[,;]+", a) if p]
    b_parts = [p for p in re.split(r"[,;]+", b) if p]
    if len(a_parts) != len(b_parts):
        return False
    for pa, pb in zip(a_parts, b_parts):
        fa = _to_fraction(pa)
        fb = _to_fraction(pb)
        if fa is None or fb is None:
            if pa != pb:
                return False
            continue
        if abs(float(fa) - float(fb)) > 1e-4:
            return False
    return True

def vqa_open_equivalent(model_output: str, correct_answer: str, dataset: str = None) -> bool:
    """Deterministic open-ended VQA scoring shared by all methods.

    Base rule is normalized exact/containment. For numeric/OCR-heavy VQA we add
    format-invariant matching so answers such as "2,287,881" and "2287881" or
    "3,781 million GBP" and "3781" are judged consistently for every method.
    Dataset-specific guards avoid unsafe unit-insensitive matches on medical VQA.
    """
    dataset = (dataset or "").lower()

    def _norm(s: str) -> str:
        s = (s or "").strip().lower()
        s = re.sub(r"</?answer>", "", s)
        s = s.replace("&", "and")
        s = re.sub(r"[^a-z0-9一-鿿 ]", " ", s)
        return re.sub(r"\s+", " ", s).strip()

    def _compact(s: str) -> str:
        return re.sub(r"[^a-z0-9一-鿿]+", "", (s or "").lower().replace("&", "and"))

    pred, gold = _norm(model_output), _norm(correct_answer)
    if not pred or not gold:
        return False
    if pred == gold or gold in pred.split() or gold in pred or (len(gold) > 2 and pred in gold):
        return True

    num_words = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    }

    def _number_tokens(s: str):
        s = re.sub(r"</?answer>", " ", (s or "").lower())
        vals = []
        for m in re.finditer(r"\b(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)\b", s):
            a, b = float(m.group(1)), float(m.group(2))
            if b:
                vals.append(f"{a / b:.10g}")
        for m in re.finditer(r"[-+]?\d[\d,]*(?:\.\d+)?", s):
            vals.append(m.group(0).replace(",", ""))
        for m in re.finditer(r"\b\d{1,3}(?:\s+\d{3})+(?:\.\d+)?\b", s):
            vals.append(re.sub(r"\s+", "", m.group(0)))
        out = []
        for v in vals:
            if v not in out:
                out.append(v)
        return out

    def _close_num(a: str, b: str) -> bool:
        try:
            fa, fb = float(a), float(b)
            return abs(fa - fb) <= max(1e-4, abs(fb) * 1e-4)
        except Exception:
            return False

    def _numeric_match() -> bool:
        gt = _number_tokens(correct_answer)
        pt = _number_tokens(model_output)
        if not gt or not pt:
            return False
        return all(any(p == g or _close_num(p, g) for p in pt) for g in gt)

    def _number_word_match() -> bool:
        cp, cg = _compact(model_output), _compact(correct_answer)
        for w, d in num_words.items():
            if (cg == w and cp == d) or (cg == d and cp == w):
                return True
        return False

    cp, cg = _compact(model_output), _compact(correct_answer)
    if dataset == "chartqa":
        return _numeric_match()
    if dataset == "textvqa":
        return _numeric_match() or _number_word_match() or (len(cg) >= 3 and len(cp) >= 3 and (cg in cp or cp in cg))
    if dataset in {"slake", "vqarad"}:
        # Conservative: allow number-word equivalence and punctuation/OCR variants,
        # but do not ignore units (e.g., 5mm is not 5cm).
        return _number_word_match() or (len(cg) >= 3 and len(cp) >= 3 and (cg in cp or cp in cg))
    return False

def execute_code_with_tests(code: str, test_list: List[str], timeout: int = 5) -> Dict[str, Any]:
    
    required_imports = set()
    for test in test_list:
        if 'math.' in test:
            required_imports.add('import math')
    
    existing_imports = set()
    for imp in required_imports:
        if imp in code:
            existing_imports.add(imp)
    
    missing_imports = required_imports - existing_imports
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
        temp_file = f.name
        
        for imp in sorted(missing_imports):
            f.write(imp + '\n')
        
        if missing_imports:
            f.write('\n')
        
        f.write(code + '\n\n')
        
        for test in test_list:
            f.write(test + '\n')
    
    try:
        result = subprocess.run(
            ['python', temp_file],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        os.unlink(temp_file)
        
        if result.returncode == 0:
            return {"success": True, "passed": len(test_list), "total": len(test_list), "errors": []}
        else:
            return {"success": False, "passed": 0, "total": len(test_list), "errors": [result.stderr]}
    except subprocess.TimeoutExpired:
        if os.path.exists(temp_file):
            os.unlink(temp_file)
        return {"success": False, "passed": 0, "total": len(test_list), "errors": ["Execution timeout"]}
    except Exception as e:
        if os.path.exists(temp_file):
            os.unlink(temp_file)
        return {"success": False, "passed": 0, "total": len(test_list), "errors": [str(e)]}

def extract_function_name_from_tests(test_list: List[str]) -> Optional[str]:
    BUILTIN_SKIP = {
        'assert', 'math', 'set', 'list', 'dict', 'tuple', 'str', 'int', 'float',
        'len', 'range', 'enumerate', 'zip', 'map', 'filter', 'sorted', 'sum',
        'max', 'min', 'abs', 'round', 'all', 'any', 'isinstance', 'type',
        'print', 'input', 'open', 'isclose', 'sqrt', 'pow', 'ceil', 'floor'
    }
    
    function_candidates = []
    
    for test in test_list:
        matches = re.findall(r'\b([a-zA-Z_]\w*)\s*\(', test)
        for match in matches:
            if match not in BUILTIN_SKIP:
                function_candidates.append(match)
    
    if not function_candidates:
        return None
    
    most_common = Counter(function_candidates).most_common(1)
    return most_common[0][0] if most_common else None

def majority_vote(answers: List[str]) -> str:
    from collections import defaultdict
    
    if not answers:
        return ""
    
    counts = defaultdict(int)
    for a in answers:
        if a:
            normalized = normalize_answer(a)
            counts[normalized] += 1
    
    if not counts:
        return answers[0] if answers else ""
    
    return max(counts.items(), key=lambda x: x[1])[0]

def code_vote(codes: List[str]) -> str:
    def normalize_code(code: str) -> str:
        lines = code.split('\n')
        normalized = []
        for line in lines:
            if '#' in line:
                line = line[:line.index('#')]
            line = line.strip()
            if line:
                normalized.append(line)
        return '\n'.join(normalized)
    
    if not codes:
        return ""
    
    normalized_codes = [normalize_code(c) for c in codes if c]
    
    if not normalized_codes:
        return codes[0] if codes else ""
    
    code_hashes = [hash(nc) for nc in normalized_codes]
    most_common_hash = Counter(code_hashes).most_common(1)[0][0]
    
    for i, h in enumerate(code_hashes):
        if h == most_common_hash:
            return codes[i]
    
    return codes[0]

def parse_comparison_result(raw: str, default: bool = True) -> bool:
    if "<choose>" in raw and "</choose>" in raw:
        try:
            choice = raw.split("<choose>")[1].split("</choose>")[0]
            return "A" in choice
        except:
            pass
    for c in reversed(raw.strip().upper()):
        if c.isalpha():
            return c == 'A'
    return default
