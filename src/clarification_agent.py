from typing import List, Dict, Tuple, Any
from src.llm_integration import LLMClient
import json

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
            
            # Robust JSON extraction
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            
            if start == -1 or end == 0:
                raise ValueError("No JSON object found in response")
                
            data = json.loads(response_text[start:end])
            
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
                "rationale": f"Parsing Error: {e}"
            }

    async def generate_questions(self, intention: str, num_questions: int = 3) -> List[str]:
        # Legacy method compatibility
        data = await self.analyze_status(intention)
        return data.get("questions", [])[:num_questions]
