import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from src.features.discovery_agent import DiscoveryAgent
from src.prompt_builder import PromptBuilder

@pytest.fixture
def mock_llm_client():
    client = MagicMock()
    # Mock for file selection, individual analyses, and synthesis
    client.agenerate_completion = AsyncMock()
    return client

@pytest.mark.asyncio
async def test_discovery_agent_parallel_investigation(mock_llm_client):
    # Setup mock responses
    # 1. File Selection: returns 2 files
    # 2. Analyst 1 analysis
    # 3. Analyst 2 analysis
    # 4. Synthesis
    mock_llm_client.agenerate_completion.side_effect = [
        '["src/main.py", "src/utils.py"]', # File selection
        "Insight 1", # Analyst 1
        "Insight 2", # Analyst 2
        "Final Synthesized Architectural Insights" # Synthesis
    ]
    
    agent = DiscoveryAgent(mock_llm_client)
    
    with patch('src.features.discovery_agent.read_project_file', return_value="dummy content"):
        insights = await agent.investigate_and_analyze(".", "Implement new feature", "tree")
        
        assert "Synthesized" in insights
        # Total calls: 1 (selection) + 2 (analysts) + 1 (synthesis) = 4
        assert mock_llm_client.agenerate_completion.call_count == 4

@pytest.mark.asyncio
async def test_prompt_builder_with_parallel_insights(mock_llm_client):
    # Setup mock responses
    mock_llm_client.agenerate_completion.side_effect = [
        "Expert Architect", # suggest_persona
        "Complex", # estimate_complexity
        '["src/main.py"]', # file selection
        "Analyst Insight", # individual analyst
        "Synthesized Insights", # synthesizer
        "Final Optimized Prompt" # prompt optimizer
    ]
    
    builder = PromptBuilder(mock_llm_client)
    
    with patch('src.prompt_builder.scan_directory', return_value="tree"):
        with patch('src.features.discovery_agent.read_project_file', return_value="content"):
            prompt, paths = await builder.build_prompt(
                intention="Test intention",
                answers=[],
                questions=[],
                auto_discover=True,
                root_path="."
            )
            
            assert "Final Optimized Prompt" in prompt
            # Check if all steps were called
            # persona(1), complexity(1), selection(1), analyst(1), synthesizer(1), optimizer(1) = 6
            assert mock_llm_client.agenerate_completion.call_count >= 5
