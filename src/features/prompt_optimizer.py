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
Your goal is to transform a structured technical prompt into an elite-level directive that forces a coding agent to produce high-density, research-quality results.

CRITICAL GUIDELINES:
1. **PRUNE REDUNDANCY**: DO NOT include environment setup, README summaries, or generic "how to run" instructions. The coding agent already has access to the codebase.
2. **EXPAND TECHNICAL DEPTH**: Blow up the architectural directives into detailed, multi-section requirements. Focus on data flows, edge cases, and high-concurrency constraints.
3. **FORCE DEEP THINKING**: Explicitly mandate a massive, pedantic <thinking> process. Command the agent to be self-critical and analyze trade-offs (e.g., Space-Time complexity, CAP theorem implications).
4. **RESEARCH RIGOR**: Use formal academic terminology. Focus on modularity, scalability, and formal verification goals.
5. **ACTIONABLE PROTOCOLS**: Replace generic instructions with strict implementation protocols.

MODE-SPECIFIC ENHANCEMENT:
- If mode is 'one-shot': Focus on architectural perfection and "correct-by-construction" code in a single pass.
- If mode is 'iterative': Emphasize the 'Atomic Cycle' (Implement -> Pilot -> Refine).
- If mode is 'chain-of-thought': Mandate exhaustive systemic modeling and research before a single line of code is written.
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
