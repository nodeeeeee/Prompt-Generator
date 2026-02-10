import asyncio
import os
import sys
import logging
from src.llm_integration import LLMClient
from src.clarification_agent import ClarificationAgent
from src.prompt_builder import PromptBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CreativityTest")

async def test_creativity_flow():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("API Key missing.")
        return

    client = LLMClient(default_model="o3-mini", timeout=120)
    clarifier = ClarificationAgent(client)
    builder = PromptBuilder(client)

    intention = "Implement a wait-free, linearizable multi-producer multi-consumer queue using hazard pointers for memory reclamation in C++23."
    logger.info(f"Testing Creativity Mode for intention: {intention}")

    # 1. Analyze Status (Get questions)
    res = await clarifier.analyze_status(intention)
    logger.info(f"Generated {len(res['questions'])} questions.")

    # 2. Self-Answer
    logger.info("Self-answering questions...")
    qa_history = await clarifier.self_answer_questions(intention, res['questions'])
    for item in qa_history:
        logger.info(f"Q: {item['q']}\nA: {item['a']}\n")

    # 3. Build Prompt with synthetic QA
    logger.info("Building final prompt with synthetic QA...")
    questions = [i['q'] for i in qa_history]
    answers = [i['a'] for i in qa_history]
    
    final_prompt, _ = await builder.build_prompt(intention, answers, questions, mode="iterative")
    
    logger.info(f"Final Prompt Length: {len(final_prompt)}")
    logger.info("--- PREVIEW OF GENERATED PROMPT ---")
    logger.info(final_prompt[:1000] + "...")

if __name__ == "__main__":
    asyncio.run(test_creativity_flow())