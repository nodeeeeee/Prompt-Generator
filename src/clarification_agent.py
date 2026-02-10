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
        In creativity mode, the agent answers its own questions by performing deep technical reasoning
        for each question individually to ensure maximum technical density and architectural rigor.
        """
        if not questions:
            return []

        logger.info(f"Self-answering {len(questions)} architectural questions for: {intention}")
        
        qa_history = []

        # We answer questions one-by-one to prevent the model from becoming generic
        # and to allow for parallel execution if desired (here sequential for maximum coherence)
        for i, q in enumerate(questions):
            system_prompt = """You are a Principal Systems Architect and Lead Researcher. 
You are defining the core technical specifications for a high-complexity project.

Your goal is to provide a definitive, technically dense, and state-of-the-art answer to an architectural question.

MANDATORY GUIDELINES:
1. **NO GENERIC BOILERPLATE**: Absolutely avoid phrases like "industry-standard", "high-performance", or "best practices".
2. **BE PEDANTICALLY SPECIFIC**: Specify exact algorithms (e.g., Paxos, RCU, LSM-trees), libraries (e.g., jemalloc, io_uring), data structures, and memory models.
3. **MANDATE RIGOR**: Specify exact performance targets (e.g., "99th percentile latency < 500us"), scalability limits, and formal verification goals.
4. **THINK AS A SCIENTIST**: Your answer should feel like a peer-reviewed methodology section.

Respond with ONLY the technical specification for this specific question.
"""

            user_content = f"""USER INTENTION: {intention}
PREVIOUS SPECIFICATIONS: {str(qa_history) if qa_history else 'None'}

QUESTION TO ANSWER: {q}

Provide the definitive technical specification:"""

            try:
                # Use real-time status updates via logger if possible, but here we return to UI
                ans = await self.llm_client.agenerate_completion(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ], 
                    temperature=0.8,
                    timeout=45
                )
                
                if not ans or len(ans.strip()) < 10:
                    ans = f"Detailed architectural analysis of {q} requires implementation of a custom logic layer optimized for the specific {intention} constraints."
                
                qa_history.append({"q": q, "a": ans.strip()})
                logger.info(f"  Processed self-answer {i+1}/{len(questions)}")
                
            except Exception as e:
                logger.error(f"Failed to self-answer question {i+1}: {e}")
                qa_history.append({"q": q, "a": f"Technical specification for {q} will be derived during the initial research phase, prioritizing the core {intention} requirements."})

        return qa_history
