from src.llm_integration import LLMClient

async def suggest_persona(intention: str, llm_client: LLMClient) -> str:
    """
    Suggests a persona for the LLM based on the intention.
    """
    prompt = (
        f"Suggest a single professional persona for an AI assistant to help with this task: '{intention}'. "
        "Return ONLY the persona name and a brief description (e.g., 'Senior Python Backend Developer: Expert in API design...')."
    )
    
    messages = [{"role": "user", "content": prompt}]
    res = await llm_client.agenerate_completion(messages)
    return res or "Senior Software Engineer"
