import asyncio
import os
import sys
import logging
from unittest.mock import MagicMock, AsyncMock, patch

# Add project root
sys.path.append(os.getcwd())

from src.llm_integration import LLMClient
from src.clarification_agent import ClarificationAgent
from src.prompt_builder import PromptBuilder
from src.features.idea_generator import generate_idea_and_prompt, generate_idea_questions, generate_raw_idea

# Mock Streamlit session state
class MockSessionState(dict):
    def __getattr__(self, key):
        return self.get(key)
    def __setattr__(self, key, value):
        self[key] = value

async def simulate_ui_journey():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("AppSimulation")
    
    logger.info("🎭 Starting Streamlit App Logic Simulation...")
    
    # 1. Initialize State (Mimic src/ui/app.py)
    state = MockSessionState({
        "intention": "",
        "generated_prompt": "",
        "second_prompt": "",
        "current_questions": [],
        "qa_history": [],
        "clarification_status": "IDLE",
        "estimated_turns": 0,
        "project_context_str": "Project structure...",
        "generated_idea": "",
        "idea_qa_history": [],
        "idea_questions": [],
        "idea_clarification_status": "IDLE",
        "selected_files": {},
        "discovered_files": [],
        "paper_text": ""
    })

    # 2. Setup Services
    mock_client = MagicMock(spec=LLMClient)
    mock_client.agenerate_completion = AsyncMock()
    
    clarifier = ClarificationAgent(mock_client)
    builder = PromptBuilder(mock_client)

    # --- SIMULATE TAB 1: NEW PROJECT (CREATIVITY MODE) ---
    logger.info("\n[UI Action] User enters intent and clicks 'Analyze & Start'")
    user_intent = "Build a high-performance wait-free MPMC queue."
    state.intention = user_intent
    creativity_mode = True
    
    # Mock LLM behavior for architecting
    mock_client.agenerate_completion.side_effect = [
        '{"status": "REFINING", "questions": ["Memory model?"]}', # analyze_status
        '["Sequential consistency"]', # self_answer_questions
        "Insights", # investigate_and_analyze (Persona)
        "Complexity", # investigate_and_analyze (Complexity)
        '["file.py"]', # file selection
        "Analyst insight", # analyst
        "Synthesized context", # synthesizer
        "Complexity Optimization", # optimizer
        "FINAL PROMPT CONTENT" # build_prompt
    ]

    # Mimic logic in tab1
    result = await clarifier.analyze_status(user_intent)
    if creativity_mode and result["status"] == "REFINING":
        self_qa = await clarifier.self_answer_questions(user_intent, result["questions"])
        state.qa_history.extend(self_qa)
        state.clarification_status = "READY"
    
    logger.info(f"  Tab 1 Status: {state.clarification_status}, QA History Size: {len(state.qa_history)}")
    
    # Mimic logic in Build Prompt
    questions = [item['q'] for item in state.qa_history]
    answers = [item['a'] for item in state.qa_history]
    
    # Mock some internal calls to prevent deep LLM call trees during logic test
    with patch('src.features.context_manager.scan_directory', return_value="tree"):
        with patch('src.features.discovery_agent.read_project_file', return_value="content"):
            final_prompt, disc_paths = await builder.build_prompt(state.intention, answers, questions, mode="iterative")
            state.generated_prompt = final_prompt
        
    logger.info(f"  ✅ Tab 1 Prompt Built. Length: {len(state.generated_prompt)}")

    # --- SIMULATE TAB 2: PROJECT EVOLUTION ---
    logger.info("\n[UI Action] User switches to Tab 2: Evolution")
    evolution_mode = "🔬 Experimentation Lab"
    current_choice = "conduct experiment" 
    
    # Mock logic: Brainstorm -> Design -> READY_AUTO -> Build
    mock_client.agenerate_completion.side_effect = [
        "Ablation study", # generate_raw_idea
        '["What metrics?"]', # generate_idea_questions
        '["Latency"]', # self_answer_questions
        "Expert Persona", # build_prompt sub-call
        "High Complexity", # build_prompt sub-call
        '["main.py"]', # file selection
        "Analyst 1", # parallel analyst
        "Synthesis", # synthesizer
        "Optimized Final" # prompt optimizer
    ]
    
    state.generated_idea = await generate_raw_idea(mock_client, state.project_context_str, current_choice)
    logger.info(f"  Generated Idea: {state.generated_idea}")
    
    qs = await generate_idea_questions(mock_client, state.project_context_str, state.generated_idea, current_choice)
    self_qa = await clarifier.self_answer_questions(state.generated_idea, qs)
    state.idea_qa_history = self_qa
    state.idea_clarification_status = "READY_AUTO"
    
    logger.info("  Processing READY_AUTO flow...")
    if state.idea_clarification_status == "READY_AUTO":
        final_p, d_paths = await generate_idea_and_prompt(
            mock_client, builder, state.project_context_str, current_choice, 
            state.generated_idea, state.idea_qa_history, root_path=".", auto_discover=True
        )
        state.generated_prompt = final_p
        state.idea_clarification_status = "READY"

    logger.info(f"  ✅ Tab 2 Evolution Successful. Choice used: {current_choice}")

    logger.info("\n✨ APP LOGIC SIMULATION SUCCESSFUL.")

if __name__ == "__main__":
    asyncio.run(simulate_ui_journey())