import os
from enum import Enum
from openai import AsyncOpenAI

class GraphType(Enum):
    SUMMARIZE = "summarize"
    AGGREGATE = "aggregate"
    REFLECT = "reflect"
    DEBATE = "debate"
    LLM_AGG = "llm_agg"
    DEBATE_LLM_AGG = "debate_llm_agg"

    SELF_REFINE = "self_refine"
    HIERARCHICAL = "hierarchical"

class AgentType(Enum):
    PREDICTOR = "predictor"
    SUMMARIZER = "summarizer"
    AGGREGATOR = "aggregator"
    REFLECTOR = "reflector"
    DEBATOR = "debator"
    LLM_AGG = "llm_agg"
    MATH_EXPERT = "math_expert"
    MATH_ANALYST = "math_analyst"
    PROGRAMMING_EXPERT = "programming_expert"
    PROJECT_MANAGER = "project_manager"
    PROGRAMMER = "programmer"
    TEST_ANALYST = "test_analyst"
    BUG_FIXER = "bug_fixer"
    KNOWLEDGEABLE_EXPERT = "knowledgeable_expert"
    CRITIC = "critic"
    PSYCHOLOGIST = "psychologist"
    HISTORIAN = "historian"
    REFINER = "refiner"  
    MATH_AGENT = "math_agent"
    SCIENCE_AGENT = "science_agent"
    CODE_AGENT = "code_agent"
    TASK_SUMMARIZER = "task_summarizer"

class TaskType(Enum):
    MATH = "math"               
    MATH_CHOICE = "math_choice"    
    REASONING_CHOICE = "reasoning_choice"
    VQA_OPEN = "vqa_open"
    VQA_CHOICE = "vqa_choice"
    CODE = "code"              

DATASET_ROOT = os.environ.get(
    "HANDOFF_DATASET_ROOT",
    "/public2/TangXiaoying/agentv5/datasets",
)


def dataset_path(*parts: str) -> str:
    return os.path.join(DATASET_ROOT, *parts)


def _clear_proxy_env():
    # Local vLLM calls should go directly to localhost instead of through proxies.
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        os.environ.pop(key, None)


DATASET_CONFIG = {
    "math": {
        "path": dataset_path("math500", "math500.jsonl"),
        "task_type": TaskType.MATH,
        "problem_key": "problem",
        "answer_key": "answer",
        "default_use_judge": False, 
    },
    "math500": {
        "path": dataset_path("math500", "math500.jsonl"),
        "task_type": TaskType.MATH,
        "problem_key": "problem",
        "answer_key": "answer",
        "default_use_judge": False, 
    },
    "aime": {
        "path": dataset_path("aime2025", "aime2025.jsonl"),
        "task_type": TaskType.MATH,
        "problem_key": "problem",
        "answer_key": "answer",
        "default_use_judge": False,
    },
    "aime2025": {
        "path": dataset_path("aime2025", "aime2025.jsonl"),
        "task_type": TaskType.MATH,
        "problem_key": "problem",
        "answer_key": "answer",
        "default_use_judge": False,
    },
    "agi": {
        "path": dataset_path("agieval_math", "test.jsonl"),
        "task_type": TaskType.MATH,
        "problem_key": "problem",
        "answer_key": "answer",
        "default_use_judge": False,
    },
    "agieval": {
        "path": dataset_path("agieval_math", "test.jsonl"),
        "task_type": TaskType.MATH,
        "problem_key": "problem",
        "answer_key": "answer",
        "default_use_judge": False,
    },
    "aqua": {
        "path": dataset_path("aqua", "aqua.jsonl"),
        "task_type": TaskType.MATH_CHOICE,
        "problem_key": "problem",
        "answer_key": "answer",
        "default_use_judge": False,
    },
    "gpqa": {
        "path": dataset_path("gpqa", "gpqa_diamond.jsonl"),
        "task_type": TaskType.REASONING_CHOICE,
        "problem_key": "question",
        "answer_key": "answer",
        "default_use_judge": False,
    },
    "mbpp": {
        "path": dataset_path("mbpp", "sanitized-mbpp.jsonl"),
        "task_type": TaskType.CODE,
        "problem_key": "prompt",
        "answer_key": None,
        "default_use_judge": True,
    },
    "humaneval": {
        "path": dataset_path("humaneval", "humaneval.jsonl"),
        "task_type": TaskType.CODE,
        "problem_key": "prompt",
        "answer_key": None,
        "default_use_judge": True,
    },
    "vqarad": {
        "path": dataset_path("vqa", "vqarad", "test.json"),
        "task_type": TaskType.VQA_OPEN,
        "problem_key": "question",
        "answer_key": "answer",
        "image_key": "image_name",
        "image_dir": dataset_path("vqa", "vqarad", "images"),
        "default_use_judge": True,
    },
    "pmcvqa": {
        "path": dataset_path("vqa", "pmcvqa", "test.json"),
        "task_type": TaskType.VQA_CHOICE,
        "problem_key": "question",
        "answer_key": "answer",
        "image_key": "image_name",
        "image_dir": dataset_path("vqa", "pmcvqa", "images"),
        "default_use_judge": False,
        "choice_answer_from_text": True,
    },
    "slake": {
        "path": dataset_path("vqa", "slake", "test.json"),
        "task_type": TaskType.VQA_OPEN,
        "problem_key": "question",
        "answer_key": "answer",
        "image_key": "img_name",
        "image_dir": dataset_path("vqa", "slake", "images"),
        "default_use_judge": True,
        "row_filter": ("q_lang", "en"),
    },
    "chartqa": {
        "path": dataset_path("chartqa", "test", "metadata.json"),
        "task_type": TaskType.VQA_OPEN,
        "problem_key": "question",
        "answer_key": "answer",
        "image_key": "image",
        "default_use_judge": True,
        "answer_is_list": True,
    },
    "textvqa": {
        "path": dataset_path("textvqa", "test", "metadata.json"),
        "task_type": TaskType.VQA_OPEN,
        "problem_key": "question",
        "answer_key": "answer",
        "image_key": "image",
        "default_use_judge": True,
    },
}

def _api_key() -> str:
    return os.environ.get("OPENAI_API_KEY", "dummy")


def _vllm_url(env_name: str, default_port: str) -> str:
    return os.environ.get(env_name) or f"http://localhost:{default_port}/v1"


def create_main_client():
    _clear_proxy_env()
    return AsyncOpenAI(
        api_key=_api_key(),
        base_url=_vllm_url("MASPO_BASE_URL", os.environ.get("MASPO_WORK_PORT", "8005")),
        default_headers={}
    )

def create_judge_client():
    _clear_proxy_env()
    return AsyncOpenAI(
        api_key=_api_key(),
        base_url=_vllm_url("MASPO_JUDGE_BASE_URL", os.environ.get("MASPO_STRONG_PORT", "8004")),
        default_headers={}
    )

def create_evaluator_client():
    _clear_proxy_env()
    return AsyncOpenAI(
        api_key=_api_key(),
        base_url=_vllm_url("MASPO_EVALUATOR_BASE_URL", os.environ.get("MASPO_STRONG_PORT", "8004")),
        default_headers={}
    )

aclient = create_main_client()
bclient = create_judge_client()


def get_default_use_judge(dataset: str) -> bool:
    config = DATASET_CONFIG.get(dataset)
    if not config:
        return False
    return config.get("default_use_judge", False)
