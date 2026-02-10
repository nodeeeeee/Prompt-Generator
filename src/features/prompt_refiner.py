from typing import List, Dict
from src.llm_integration import LLMClient

class PromptRefiner:
    """
    Allows targeted, interactive refinement of generated prompts.
    """
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    async def refine_prompt(
        self, 
        current_prompt: str, 
        instruction: str,
        context_insights: str = ""
    ) -> str:
        """
        Refines the current prompt based on a specific user instruction.
        """
        system_instruction = """You are an expert Prompt Engineer. 
You are given a high-quality technical prompt and a specific refinement instruction from a researcher.
Your goal is to modify the prompt to incorporate the instruction while maintaining the professional, high-density tone.

STRICT RULES:
1. Keep the structural headers (e.g., # MISSION, # ARCHITECTURE).
2. ONLY change the parts relevant to the instruction.
3. If the instruction is vague, interpret it in a way that maximizes technical rigor.
"""

        user_content = f"""
CURRENT PROMPT:
---
{current_prompt}
---

ARCHITECTURAL INSIGHTS (for context):
{context_insights}

REFINEMENT INSTRUCTION: {instruction}

Provide the updated prompt in full.
"""
        
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_content}
        ]

        return await self.llm_client.agenerate_completion(messages, temperature=0.5)
