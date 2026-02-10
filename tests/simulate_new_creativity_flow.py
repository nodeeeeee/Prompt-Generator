import asyncio
import os
import sys
import logging
from src.llm_integration import LLMClient
from src.clarification_agent import ClarificationAgent
from src.prompt_builder import PromptBuilder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger("CreativityTest")

async def test_creativity_flow():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("API Key missing.")
        return

    client = LLMClient(default_model="o3-mini", timeout=120)
    clarifier = ClarificationAgent(client)
    builder = PromptBuilder(client)

    # Use the user's specific case
    intention = "An agent that can automatically scan github issues and fix bugs."
    logger.info(f"Testing Deep Architectural Self-Answering for: {intention}")

    # 1. Analyze Status (Get questions)
    logger.info("Scoping project requirements...")
    res = await clarifier.analyze_status(intention)
    logger.info(f"Generated {len(res['questions'])} architectural questions.")

    # 2. Self-Answer (Deep Reasoning)
    logger.info("Performing sequential deep reasoning for each question...")
    qa_history = await clarifier.self_answer_questions(intention, res['questions'][:3]) # Test with first 3 for speed
    
    for item in qa_history:
        logger.info(f"---")
        logger.info(f"Q: {item['q']}")
        logger.info(f"A: {item['a']}")

    # 3. Build Prompt with synthetic QA
    logger.info("\nBuilding final architectural directive...")
    questions = [i['q'] for i in qa_history]
    answers = [i['a'] for i in qa_history]
    
    final_prompt, _ = await builder.build_prompt(intention, answers, questions, mode="chain-of-thought")
    
    logger.info(f"Final Prompt Length: {len(final_prompt)}")
    
    # Check for keywords that indicate "thinking" instead of "boilerplate"
    generic_keywords = ["industry-standard", "high-performance", "best practices"]
    found_generic = [kw for kw in generic_keywords if kw in final_prompt.lower()]
    
    if found_generic:
        logger.warning(f"⚠️ Found generic keywords in prompt: {found_generic}")
    else:
        logger.info("✅ No generic boilerplate detected in final prompt.")

if __name__ == "__main__":
    asyncio.run(test_creativity_flow())
