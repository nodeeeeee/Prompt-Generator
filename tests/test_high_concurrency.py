import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from src.features.discovery_agent import DiscoveryAgent
from src.llm_integration import LLMClient

@pytest.mark.asyncio
async def test_high_concurrency_file_analysis():
    mock_client = MagicMock(spec=LLMClient)
    # 1 selection call + 15 analyst calls + 1 synthesis call = 17 calls
    mock_client.agenerate_completion = AsyncMock(side_effect=["[file%d.py for d in range(15)]"] + ["Insight"]*15 + ["Final Synthesis"])
    
    # Mock return values for file selection to return 15 paths
    file_list = [f"file{i}.py" for i in range(15)]
    mock_client.agenerate_completion.side_effect = [
        str(file_list).replace("'", '"'), # Selection
        *([f"Insight for {f}" for f in file_list]), # 15 Analyst calls
        "Final Synthesized Architectural Directive" # Synthesis
    ]
    
    agent = DiscoveryAgent(mock_client)
    
    with patch('src.features.discovery_agent.read_project_file', return_value="def code(): pass"):
        # This will trigger 15 parallel analyst calls
        res = await agent.investigate_and_analyze(".", "intent", "tree")
        
        assert "Final Synthesized" in res
        assert mock_client.agenerate_completion.call_count == 17
        print(f"Successfully handled {mock_client.agenerate_completion.call_count} parallel/sequential calls.")
