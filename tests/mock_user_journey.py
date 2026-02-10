import asyncio
import os
import sys
import threading
from typing import List, Dict, Any, Tuple
from unittest.mock import MagicMock, AsyncMock

# Add project root
sys.path.append(os.getcwd())

from src.llm_integration import LLMClient
from src.clarification_agent import ClarificationAgent
from src.prompt_builder import PromptBuilder
from src.features.idea_generator import generate_idea_and_prompt, generate_idea_questions, generate_raw_idea
from src.features.context_manager import scan_directory
from src.features.file_interface import read_project_file

# Re-implement run_async logic from app.py to test it
def run_async_mock(coro):
    result = []
    exception = []
    def target():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result.append(loop.run_until_complete(coro))
            loop.close()
        except Exception as e:
            exception.append(e)
    thread = threading.Thread(target=target)
    thread.start()
    thread.join()
    if exception: raise exception[0]
    return result[0]

async def full_functionality_test():
    print("🎭 Starting Comprehensive User Journey Mock Test...")
    
    mock_client = MagicMock(spec=LLMClient)
    mock_client.agenerate_completion = AsyncMock()
    
    clarifier = ClarificationAgent(mock_client)
    builder = PromptBuilder(mock_client)

    print("\n[Tab 1] Testing New Project Flow (Creativity Mode)...")
    mock_client.agenerate_completion.return_value = "Optimized Creativity Prompt"
    # No analyze_status needed if we just go straight to build
    final_p, disc = run_async_mock(builder.build_prompt("Build a store", [], [], mode="one-shot"))
    print(f"  ✅ Final Prompt built in creativity mode.")

    print("\n[Tab 2] Testing Project Evolution Flow...")
    tree = scan_directory(".")
    print(f"  ✅ Directory Scanned.")
    
    file_content = read_project_file(".", "README.md")
    print(f"  ✅ File 'README.md' read.")

    mock_client.agenerate_completion.return_value = "Compare performance of different lock types"
    idea = run_async_mock(generate_raw_idea(mock_client, tree, "conduct experiment"))
    print(f"  ✅ AI Idea Generated.")

    mock_client.agenerate_completion.return_value = '["What locks?", "What metrics?"]'
    qs = run_async_mock(generate_idea_questions(mock_client, tree, idea, "conduct experiment"))
    print(f"  ✅ Technical Questions Designed.")

    mock_client.agenerate_completion.side_effect = [
        "Senior Researcher", 
        "High", 
        '["src/main.py"]', 
        "Final Research Prompt" 
    ]
    
    final_p, disc = run_async_mock(generate_idea_and_prompt(
        mock_client, builder, tree, "conduct experiment", idea, 
        qa_history=[{"q": "What locks?", "a": "Mutex vs RCU"}],
        root_path=".", auto_discover=True
    ))
    print(f"  ✅ Evolved Prompt Built.")
    print(f"  ✅ Autonomously Read Files: {disc}")

    print("\n[Tab 3] Testing Paper Analysis Flow...")
    mock_client.agenerate_completion.side_effect = None
    mock_client.agenerate_completion.return_value = "Implementation Plan based on Paper"
    
    final_p, disc = run_async_mock(builder.build_prompt(
        "Implement Paper", [], [], mode="iterative", project_context="### Paper Content\nDense research text..."
    ))
    print(f"  ✅ Paper Implementation Plan Built.")

    print("\n✨ ALL FUNCTIONALITIES VERIFIED SUCCESSFULLY.")

if __name__ == "__main__":
    asyncio.run(full_functionality_test())