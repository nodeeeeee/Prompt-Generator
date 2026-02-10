import asyncio
import os
import sys
import logging
import json
import time
from typing import List, Dict, Any
from src.llm_integration import LLMClient
from src.clarification_agent import ClarificationAgent
from src.prompt_builder import PromptBuilder
from src.features.research_journal import ResearchJournal, ResearchEntry
from src.security_engine import SecurityEngine

# Initialize results dir if missing
os.makedirs("results", exist_ok=True)

# High-fidelity logging
logger = logging.getLogger("MassSimulation")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

SCENARIOS = [
    {"domain": "Systems", "intent": "Implement a lock-free work-stealing scheduler in C++20.", "mode": "chain-of-thought", "creative": True},
    {"domain": "ML", "intent": "Develop a federated learning protocol for non-IID medical images.", "mode": "iterative", "creative": True},
    {"domain": "Security", "intent": "Design an eBPF tool for tracking TCP retransmissions in K8s.", "mode": "one-shot", "creative": True}
]

async def run_scenario(client, clarifier, builder, journal, scenario: Dict[str, Any]):
    domain = scenario["domain"]
    intent = scenario["intent"]
    mode = scenario["mode"]
    creative = scenario["creative"]
    
    logger.info(f"▶️ TESTING: [{domain}] intent='{intent}' mode={mode}")
    
    try:
        # 1. Self-Answer Check (Conciseness)
        logger.info("  [1/3] Verifying Conciseness...")
        status_res = await clarifier.analyze_status(intent)
        qa_history = await clarifier.self_answer_questions(intent, status_res['questions'][:2])
        
        for item in qa_history:
            # Conciseness Check: Should be around 3 sentences.
            sentence_count = item['a'].count('.') + item['a'].count('!') + item['a'].count('?')
            logger.info(f"    Q: {item['q'][:30]}... | Ans Sentences: {sentence_count}")
            if sentence_count > 5:
                logger.warning(f"    ⚠️ Answer might be too long ({sentence_count} sentences)")

        # 2. Build Prompt Check (Consistency)
        logger.info("  [2/3] Verifying Structural Consistency...")
        final_prompt, _ = await builder.build_prompt(intent, [i['a'] for i in qa_history], [i['q'] for i in qa_history], mode=mode)
        
        required_headers = ["# MISSION", "# ROLE", "# SPECIFICATIONS", "# IMPLEMENTATION PROTOCOL", "# EVALUATION"]
        missing = [h for h in required_headers if h not in final_prompt.upper()]
        
        if missing:
            logger.error(f"    ❌ Consistency Failure: Missing headers {missing}")
        else:
            logger.info("    ✅ All standardized headers present.")

        # 3. Length Check
        prompt_len = len(final_prompt)
        logger.info(f"  [3/3] Length Check: {prompt_len} chars")
        if prompt_len > 8000:
            logger.warning(f"    ⚠️ Prompt is very long ({prompt_len} chars)")
        elif prompt_len < 2000:
            logger.warning(f"    ⚠️ Prompt might be too short ({prompt_len} chars)")
        else:
            logger.info("    ✅ Prompt length is balanced.")

        return True if not missing else False
    except Exception as e:
        logger.error(f"❌ CRITICAL SCENARIO FAILURE: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set.")
        return

    client = LLMClient(default_model="o3-mini", timeout=120)
    clarifier = ClarificationAgent(client)
    builder = PromptBuilder(client)
    journal = ResearchJournal("results/thorough_validation_journal.json")
    
    logger.info("🚀 Starting Thorough Final Validation...")
    results = []
    for s in SCENARIOS:
        results.append(await run_scenario(client, clarifier, builder, journal, s))
    
    success_rate = sum(1 for r in results if r) / len(SCENARIOS)
    logger.info(f"\n✨ VALIDATION COMPLETE. Success Rate: {success_rate*100:.1f}%")
    if success_rate < 1.0:
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
