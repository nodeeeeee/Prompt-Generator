import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from src.features.prompt_refiner import PromptRefiner

@pytest.mark.asyncio
async def test_prompt_refiner():
    mock_client = MagicMock()
    mock_client.agenerate_completion = AsyncMock(return_value="Refined Prompt Version")
    
    refiner = PromptRefiner(mock_client)
    res = await refiner.refine_prompt("Old Prompt", "Make it better")
    
    assert res == "Refined Prompt Version"
    assert mock_client.agenerate_completion.called
    
    # Verify call structure
    args, kwargs = mock_client.agenerate_completion.call_args
    assert "Old Prompt" in args[0][1]["content"]
    assert "Make it better" in args[0][1]["content"]
