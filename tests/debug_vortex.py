import asyncio
import os
import sys
import logging
import json
from unittest.mock import MagicMock, AsyncMock, patch
from src.features.bulletproof_parser import parse_json_safely
from src.clarification_agent import ClarificationAgent
from src.features.discovery_agent import DiscoveryAgent
from src.llm_integration import LLMClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DebugVortex")

async def run_vortex_stress_test():
    logger.info("🌪 Starting Debug Vortex: Massive Stability Test")
    
    # 1. Test Bulletproof Parser with various garbage inputs
    logger.info("Phase 1: Parser Stress Test")
    garbage_inputs = [
        "Sure, here is the JSON: ```json\n[\"file1.py\"]\n``` and more text.",
        "The files are [\"a.txt\", \"b.txt\"]",
        "{\"status\": \"READY\"} - concluding my analysis.",
        "Conversation filler... { \"key\": \"value\" } trailing text",
        "No json at all here.",
        "['single', 'quotes', 'issue']",
        "```\n[\"unnamed\", \"block\"]\n```"
    ]
    for inp in garbage_inputs:
        res = parse_json_safely(inp)
        logger.info(f"  Input: {inp[:30]}... -> Parsed: {res}")

    # 2. Test Clarification Agent with mixed LLM failures
    logger.info("\nPhase 2: Agent Resilience Test")
    mock_client = MagicMock(spec=LLMClient)
    agent = ClarificationAgent(mock_client)
    
    # Mock return values that might break a normal parser
    mock_client.agenerate_completion.side_effect = [
        "Internal Error 500", # Complete failure
        "Wait, I forgot the JSON. Here it is: {\"status\": \"READY\"}", # Conversational
        "```json\n{\"status\": \"REFINING\", \"questions\": [\"Why?\"]}\n```" # Wrapped
    ]
    
    for i in range(3):
        res = await agent.analyze_status("Testing Vortex")
        logger.info(f"  Turn {i+1} Status: {res['status']}, Questions: {len(res['questions'])}")

    # 3. Test Discovery Agent with parallel failures
    logger.info("\nPhase 3: Parallel Discovery Resilience")
    disco = DiscoveryAgent(mock_client)
    
    # Selection returns 3 files, but analysts fail/hang
    mock_client.agenerate_completion.side_effect = [
        '["f1.py", "f2.py", "f3.py"]', # Selection
        Exception("Timeout"), # Analyst 1
        "Insight 2", # Analyst 2
        Exception("Crashed"), # Analyst 3
        "Final synthesis of what worked" # Synthesis
    ]
    
    # We need to ensure investigate_and_analyze handles the exceptions inside gather
    with patch('src.features.discovery_agent.read_project_file', return_value="code"):
        res = await disco.investigate_and_analyze(".", "intent", "tree")
        logger.info(f"  Discovery Result: {res[:50]}...")

    logger.info("\n✨ DEBUG VORTEX COMPLETED SUCCESSFULLY.")

if __name__ == "__main__":
    asyncio.run(run_vortex_stress_test())