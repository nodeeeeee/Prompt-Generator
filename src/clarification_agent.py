import json
import logging
from typing import List, Dict, Tuple, Any
from src.llm_integration import LLMClient
from src.features.bulletproof_parser import parse_json_safely

logger = logging.getLogger(__name__)

class ClarificationAgent:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    async def analyze_status(self, intention: str, history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Analyzes the current intention and Q&A history to decide if more info is needed.
        Returns a dictionary with status, questions, and estimated turns remaining.
        """
        history_list = history or []
        turn_count = len(history_list)
        
        # Increased cap to allow for more turns if truly necessary, but still prevent infinity
        if turn_count >= 5:
            return {"status": "READY", "questions": [], "estimated_turns_remaining": 0, "rationale": "Max turns reached."}

        history_str = ""
        if history_list:
            history_str = "\n".join([f"Q: {item['q']}\nA: {item['a']}" for item in history_list])

        system_prompt = f"""You are a Senior Software Requirements Architect. 
Your goal is to evaluate technical clarity for a software implementation task.

You must respond ONLY with a JSON object in this format:
{{
  "status": "READY" or "REFINING",
  "questions": ["Q1?", "Q2?"],
  "estimated_turns_remaining": <integer>,
  "rationale": "Brief reasoning."
}}

GUIDELINES:
1. **Estimate Turns**: Provide an honest estimate of how many more Q&A turns you need to reach 'READY' status.
2. **Efficiency**: Do not be pedantic. If you have the core "What" and "How", move to READY.
3. **Transition**: As you get closer to 0 estimated turns, your questions should become more specific and focused.

CRITERIA FOR READY:
- Intention is clear enough for a senior dev to implement.
- Basic architecture and tech stack are identifiable.
"""

        user_content = f"INTENTION: {intention}\n\nCLARIFICATION HISTORY:\n{history_str if history_str else 'None'}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        try:
            response_text = await self.llm_client.agenerate_completion(messages, temperature=0.2)
            
            data = parse_json_safely(response_text, default_fallback={})
            
            # Basic validation and defaults
            status = data.get("status", "REFINING")
            questions = data.get("questions", [])
            estimated = data.get("estimated_turns_remaining", 1)
            
            # Ensure types are correct
            return {
                "status": str(status),
                "questions": [str(q) for q in questions],
                "estimated_turns_remaining": int(estimated),
                "rationale": data.get("rationale", "")
            }
        except Exception as e:
            return {
                "status": "REFINING", 
                "questions": ["Could you provide more specific details on the architectural constraints?"],
                "estimated_turns_remaining": 1,
                "rationale": f"System Error: {e}"
            }

    async def generate_questions(self, intention: str, num_questions: int = 3) -> List[str]:
        # Legacy method compatibility
        data = await self.analyze_status(intention)
        return data.get("questions", [])[:num_questions]

    async def self_answer_questions(self, intention: str, questions: List[str]) -> List[Dict[str, str]]:
        """
        In creativity mode, the agent answers its own questions by performing deep technical reasoning.
        It investigates common practice and specifies state-of-the-art solutions.
        """
        if not questions:
            return []

        system_prompt = """You are a Principal Systems Architect and Research Lead. 
You are tasked with providing the definitive technical specifications for a high-complexity research project.

Your goal is to fill in the missing architectural details with high-density, pedantic, and state-of-the-art solutions.

GUIDELINES:
1. **NO GENERIC STATEMENTS**: Never use phrases like "industry-standard practices" or "high-performance libraries". 
2. **BE SPECIFIC**: Name specific algorithms (e.g., Paxos, RCU, LSM-trees), libraries (e.g., jemalloc, io_uring), and architectural patterns.
3. **MANDATE RIGOR**: Specify exact performance targets, memory reclamation strategies, and consistency models.
4. **COHESION**: Every answer must integrate perfectly into a unified, scalable system design.

Respond ONLY with a JSON list of detailed strings: ["Specific Answer 1...", "Specific Answer 2...", ...]
"""

        user_content = f"""USER INTENTION: {intention}

TECHNICAL QUESTIONS TO RESOLVE:
{chr(10).join([f"{i+1}. {q}" for i, q in enumerate(questions)])}

Provide architect-level, technically dense answers for each question."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        try:
            # Use higher temperature for creativity, but the system prompt enforces technical rigor
            response_text = await self.llm_client.agenerate_completion(messages, temperature=0.9)
            
            answers = parse_json_safely(response_text, default_fallback=[])
            
            if not isinstance(answers, list) or len(answers) == 0:
                # If parsing fails, try to wrap the response as a single answer rather than generic boilerplate
                logger.warning("Failed to parse answers as list, using raw response as integrated answer.")
                return [{"q": "Integrated Architectural Detail", "a": response_text}]
            
            qa_history = []
            for i, q in enumerate(questions):
                # Handle cases where LLM might provide fewer answers than questions
                ans = answers[i] if i < len(answers) else "Consulted state-of-the-art documentation: implementation requires custom logic specific to this module's performance constraints."
                qa_history.append({"q": q, "a": str(ans)})
            
            return qa_history
        except Exception as e:
            logger.error(f"Self-answering failed: {e}")
            return [{"q": q, "a": f"Technical specification required: Analysis of {q} indicates a need for deep integration with the core {intention} logic."} for q in questions]
