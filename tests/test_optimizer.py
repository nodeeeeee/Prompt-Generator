import pytest
from unittest.mock import MagicMock, AsyncMock
from src.features.prompt_optimizer import PromptOptimizer

@pytest.fixture
def mock_llm_client():
    client = MagicMock()
    client.agenerate_completion = AsyncMock()
    return client

@pytest.mark.asyncio
async def test_optimize_prompt_structure(mock_llm_client):
    optimizer = PromptOptimizer(mock_llm_client)
    mock_llm_client.agenerate_completion.return_value = "Optimized Super Prompt"
    
    result = await optimizer.optimize_prompt("raw", "iterative", "test intention")
    
    assert result == "Optimized Super Prompt"
    args, _ = mock_llm_client.agenerate_completion.call_args
    # Check if system instructions were passed
    assert "Prompt Engineer" in args[0][0]["content"]
    # Check if raw prompt was passed
    assert "raw" in args[0][1]["content"]

@pytest.mark.asyncio
async def test_optimize_prompt_fallback(mock_llm_client):
    optimizer = PromptOptimizer(mock_llm_client)
    mock_llm_client.agenerate_completion.side_effect = Exception("API Down")
    
    result = await optimizer.optimize_prompt("raw_content", "one-shot", "intention")
    
    assert "Optimization Failed" in result
    assert "raw_content" in result