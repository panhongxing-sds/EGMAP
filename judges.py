
import asyncio
from typing import List

from utils import async_call_llm, execute_code_with_tests

class LLMJudgeAgent:
    
    _TEMPLATE = (
        "Below is a {problem_type}, its standard answer, and a candidate solution. "
        "Please determine whether the candidate solution produced the correct answer as its final output. "
        "Only output 'True' or 'False', nothing else.\n\n"
        "Problem: {question}\n"
        "Standard answer: {gold}\n"
        "Candidate solution: {raw_output}\n\n"
        "Correct:"
    )

    def __init__(self, client, problem_type: str = "math problem"):
        self.client = client
        self.problem_type = problem_type

    async def ajudge(self, question: str, gold: str, raw_output: str) -> bool:
        prompt = self._TEMPLATE.format(
            problem_type=self.problem_type,
            question=question, 
            gold=gold, 
            raw_output=raw_output
        )
        resp = await async_call_llm(self.client, prompt, temperature=0.0)
        return resp.strip().lower() == "true"


class CodeJudgeAgent:
    
    def __init__(self):
        pass
    
    async def ajudge(self, code: str, test_list: List[str]) -> bool:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, execute_code_with_tests, code, test_list)
        return result["success"]
