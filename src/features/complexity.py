from src.llm_integration import LLMClient
from typing import List

async def estimate_complexity(intention: str, answers: List[str], llm_client: LLMClient) -> str:

    """

    Estimates the complexity of the task.

    """

    details = " ".join(answers)

    context = f"Intention: {intention}\nDetails: {details}"

    prompt = (

        f"Analyze the following software task: '{context}'. "

        "Estimate the complexity (Low, Medium, High) and explain why in one sentence."

    )

    

    messages = [{"role": "user", "content": prompt}]
    res = await llm_client.agenerate_completion(messages)
    return res or "Medium"
