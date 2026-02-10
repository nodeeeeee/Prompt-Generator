import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from src.prompt_builder import PromptBuilder

@pytest.fixture
def mock_llm_client():
    client = MagicMock()
    client.agenerate_completion = AsyncMock()
    return client

@pytest.mark.asyncio
@patch("src.prompt_builder.suggest_persona", new_callable=AsyncMock)
@patch("src.prompt_builder.estimate_complexity", new_callable=AsyncMock)
@patch("src.features.prompt_optimizer.PromptOptimizer.optimize_prompt", new_callable=AsyncMock)
async def test_build_prompt_one_shot(mock_optimize, mock_estimate_complexity, mock_suggest_persona, mock_llm_client):
    builder = PromptBuilder(mock_llm_client)
    
    mock_suggest_persona.return_value = "You are a Python Expert."
    mock_estimate_complexity.return_value = "Medium complexity."
    # Make the optimizer return the raw prompt for easy testing of logic
    mock_optimize.side_effect = lambda raw, mode, int: raw
    
    intention = "Create a Flask app"
    questions = ["What database?", "Any auth?"]
    answers = ["PostgreSQL", "No"]
    
    prompt, disc = await builder.build_prompt(intention, answers, questions, mode="one-shot")
    
    assert "MISSION" in prompt
    assert "Create a Flask app" in prompt
    assert "PostgreSQL" in prompt

@pytest.mark.asyncio
@patch("src.prompt_builder.suggest_persona", new_callable=AsyncMock)
@patch("src.prompt_builder.estimate_complexity", new_callable=AsyncMock)
@patch("src.features.prompt_optimizer.PromptOptimizer.optimize_prompt", new_callable=AsyncMock)
async def test_build_prompt_iterative(mock_optimize, mock_estimate_complexity, mock_suggest_persona, mock_llm_client):
    builder = PromptBuilder(mock_llm_client)
    mock_suggest_persona.return_value = "Expert"
    mock_estimate_complexity.return_value = "Low"
    mock_optimize.side_effect = lambda raw, mode, int: raw
    
    prompt, disc = await builder.build_prompt("Task", [], [], mode="iterative")
    
    assert "CREATIVE ITERATIVE EVOLUTION" in prompt
    assert "Atomic Cycle" in prompt