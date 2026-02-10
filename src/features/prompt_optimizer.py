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
Your goal is to take a structured technical prompt and transform it into an elite-level directive that forces an AI agent to produce fine-grained, research-quality results.

CRITICAL GUIDELINES:
1. **EXPAND AND ENRICH**: Do not just summarize. Blow up the prompt into a detailed, multi-section masterpiece.
2. **FORCE DEEP THINKING**: Explicitly mandate a massive <thinking> process. Tell the agent to be pedantic and self-critical.
3. **RESEARCH RIGOR**: For CS researchers, use academic-grade terminology. Focus on reproducibility, modularity, and scalability.
4. **STYLE**: Use a professional, authoritative, and mission-critical tone.
5. **STRUCTURE**: Use Markdown headers, bold text, and clear protocols.

MODE-SPECIFIC ENHANCEMENT:
- If mode is 'one-shot': Focus on architectural perfection in a single pass.
- If mode is 'iterative': Emphasize the 'Evolving Architecture' and the 'Pilot-Run-Refine' loop. Force the agent to describe how it will supplement its plan mid-implementation.
- If mode is 'chain-of-thought': Mandate exhaustive research and complex system modeling before code.
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
