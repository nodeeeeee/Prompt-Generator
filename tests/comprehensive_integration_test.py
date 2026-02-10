import asyncio
import os
import sys
import logging
from datetime import datetime
from src.llm_integration import LLMClient
from src.systems_engine import SystemsEngine, EngineState
from src.clarification_agent import ClarificationAgent
from src.prompt_builder import PromptBuilder
from src.features.discovery_agent import DiscoveryAgent
from src.features.prompt_refiner import PromptRefiner
from src.features.research_journal import ResearchJournal, ResearchEntry
from src.features.academic_exporter import AcademicExporter
from src.security_engine import SecurityEngine, SecurityState

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ComprehensiveTest")

async def run_comprehensive_test():
    logger.info("🚀 Starting Comprehensive Real-World Integration Test")
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not found.")
        return

    client = LLMClient(default_model="o3-mini", timeout=120)
    journal = ResearchJournal("results/integration_test_journal.json")
    
    # --- SCENARIO 1: NEW PROJECT ---
    logger.info("\n[1/5] Scenario: New Project")
    intent1 = "Design a high-performance, lock-free RCU-based hash map in C++."
    builder = PromptBuilder(client)
    final_p1, _ = await builder.build_prompt(intent1, ["C++20", "x86"], ["Arch?", "Platform?"], mode="chain-of-thought")
    journal.add_entry(ResearchEntry(intention=intent1, mode="chain-of-thought", final_prompt=final_p1, tags=["new"]))

    # --- SCENARIO 2: EVOLUTION ---
    logger.info("\n[2/5] Scenario: Evolution")
    intent2 = "Add deep-packet inspection to SecurityEngine."
    discovery = DiscoveryAgent(client)
    insights = await discovery.investigate_and_analyze(os.getcwd(), intent2, "src/", max_files=1)
    final_p2, _ = await builder.build_prompt(intent2, [], [], mode="iterative", project_context=insights)
    journal.add_entry(ResearchEntry(intention=intent2, mode="iterative", insights=insights, final_prompt=final_p2, tags=["evolve"]))

    # --- SCENARIO 3: PAPER TO CODE ---
    logger.info("\n[3/5] Scenario: Paper to Code")
    intent3 = "Implement the 'Efficient RCU' paper algorithm."
    paper_content = "The algorithm uses a global epoch and local generation counters..."
    final_p3, _ = await builder.build_prompt(intent3, [], [], mode="iterative", project_context=f"### Paper\n{paper_content}")
    journal.add_entry(ResearchEntry(intention=intent3, mode="iterative", final_prompt=final_p3, tags=["paper"]))

    # --- SCENARIO 4: SECURITY & PRIVACY ---
    logger.info("\n[4/5] Scenario: Security & Privacy Audit")
    security = SecurityEngine()
    audit_res = await security.process_content("Secret is sk-99999 and email is bob@example.com")
    logger.info(f"PII: {audit_res.pii_detected}")

    # --- SCENARIO 5: ACADEMIC EXPORT ---
    logger.info("\n[5/5] Scenario: Academic Export")
    latex = AcademicExporter.to_latex_methodology(insights, intent2)
    bib = AcademicExporter.get_bibtex()
    logger.info("LaTeX and BibTeX generated.")

    # Final Summary
    summary_md = journal.export_as_markdown()
    with open("results/integration_test_summary.md", "w") as f:
        f.write(summary_md)
    logger.info("\n✨ TEST SUCCESSFUL. Summary in results/integration_test_summary.md")

if __name__ == "__main__":
    asyncio.run(run_comprehensive_test())
