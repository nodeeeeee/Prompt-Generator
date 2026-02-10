import logging
from typing import List, Optional
from src.llm_integration import LLMClient

logger = logging.getLogger(__name__)

class PromptOptimizer:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    async def optimize_prompt(
        self, 
        raw_prompt: str, 
        mode: str, 
        intention: str
    ) -> str:
        """
        Uses an LLM to refine and expand the mechanically generated prompt into a 
        high-quality, long-form prompt designed for deep reasoning.
        """
        
        system_instructions = """You are an expert Prompt Engineer and Senior Software Architect. 
Your goal is to transform a technical prompt into an elite-level directive that is highly detailed but remains focused and avoids unnecessary wordiness.

CRITICAL GUIDELINES:
1. **MANDATORY STANDARDIZED STRUCTURE**: Every single prompt MUST include these exact 5 top-level Markdown headers. Do not rename, merge, or prune them:
   - # MISSION: High-level implementation goal.
   - # ROLE: Technical persona and expertise.
   - # SPECIFICATIONS: Project-specific technical constraints and synthetic Q&A.
   - # IMPLEMENTATION PROTOCOL: Step-by-step rules and iterative cycle.
   - # EVALUATION: Mandatory benchmarking and verification strategy.
2. **TECHNICAL DENSITY**: Prune all generic filler. Focus on specific libraries, algorithms, and technical edge cases.
3. **BALANCED LENGTH**: Aim for high impact per word. The total length should be substantial (ideally 4000-7000 chars) but avoid repeating concepts.
4. **FORCE DEEP THINKING**: Retain the mandatory <thinking> process but keep it focused on the immediate implementation path.
"""

        user_content = f"""
I have a mechanically generated prompt for the task: "{intention}".
The current mode is: {mode}.

MECHANICAL PROMPT:
---
{raw_prompt}
---

Please rewrite this prompt to be significantly longer, more creative, and more detailed. 
Incorporate the specific development method requirements:
1) write unit tests for the program.
2) write necessary comments.
3) For every module built, do a pilot run, get report/feedback, and refine.

Make the prompt feel like a high-level research directive from a lead scientist.
"""

        messages = [
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": user_content}
        ]

        try:
            import asyncio
            optimized_prompt = await asyncio.wait_for(
                self.llm_client.agenerate_completion(messages, temperature=0.8),
                timeout=60.0
            )
            if not optimized_prompt or not optimized_prompt.strip():
                return raw_prompt
            return optimized_prompt
        except (asyncio.TimeoutError, Exception) as e:
            # Fallback to raw if LLM optimization fails or times out
            logger.warning(f"Optimization phase soft-failed or timed out: {e}")
            return f"/* Optimization Failed or Timed Out: {e} */\n\n{raw_prompt}"
