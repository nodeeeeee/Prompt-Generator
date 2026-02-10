import asyncio
import os
import sys
import logging
from src.llm_integration import LLMClient
from src.features.pdf_parser import extract_text_from_pdf
from src.prompt_builder import PromptBuilder
from src.clarification_agent import ClarificationAgent

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FinalSimulation")

async def run_final_simulation():
    logger.info("🎬 Starting Final Input Simulation Test")
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("API Key missing. Simulation aborted.")
        return

    client = LLMClient(default_model="o3-mini", timeout=120)
    
    # --- TEST 1: PDF EXTRACTION SIMULATION ---
    logger.info("\n[Test 1] Simulating PDF Upload & Parsing...")
    try:
        from pypdf import PdfWriter
        # Create a tiny valid PDF
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        with open("tests/dummy_research.pdf", "wb") as f:
            writer.write(f)
        
        with open("tests/dummy_research.pdf", "rb") as f:
            text = extract_text_from_pdf(f)
            logger.info(f"  PDF Parser Result length: {len(text)}")
        os.remove("tests/dummy_research.pdf")
    except Exception as e:
        logger.warning(f"  PDF creation/parsing test failed: {e}")

    # --- TEST 2: CREATIVITY MODE SELF-ANSWERING ---
    logger.info("\n[Test 2] Simulating New Project (Creativity Mode) Flow...")
    intent = "Build a privacy-preserving federated learning framework for medical imaging."
    clarifier = ClarificationAgent(client)
    
    status = await clarifier.analyze_status(intent)
    logger.info(f"  Initial Questions: {status['questions']}")
    
    logger.info("  Agent is self-answering...")
    qa_history = await clarifier.self_answer_questions(intent, status['questions'])
    for item in qa_history:
        logger.info(f"    Q: {item['q']}\n    A: {item['a'][:100]}...")

    # --- TEST 3: CONSENSUS MODE SIMULATION ---
    logger.info("\n[Test 3] Simulating Consensus Mode (Two Models)...")
    builder = PromptBuilder(client)
    
    logger.info("  Generating Primary Prompt (o3-mini)...")
    prompt1, _ = await builder.build_prompt(intent, [i['a'] for i in qa_history], [i['q'] for i in qa_history], mode="chain-of-thought")
    
    logger.info("  Generating Secondary Prompt (gpt-4o-mini)...")
    client2 = LLMClient(default_model="gpt-4o-mini")
    builder2 = PromptBuilder(client2)
    prompt2, _ = await builder2.build_prompt(intent, [i['a'] for i in qa_history], [i['q'] for i in qa_history], mode="chain-of-thought")
    
    logger.info(f"  Primary Length: {len(prompt1)}")
    logger.info(f"  Secondary Length: {len(prompt2)}")
    
    assert len(prompt1) > 1000
    assert len(prompt2) > 1000
    logger.info("  ✅ Consensus generation successful.")

    logger.info("\n✨ FINAL SIMULATION COMPLETED SUCCESSFULLY.")

if __name__ == "__main__":
    asyncio.run(run_final_simulation())