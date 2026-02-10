import pytest
from unittest.mock import AsyncMock, MagicMock
from src.clarification_agent import ClarificationAgent

@pytest.fixture
def mock_llm_client():
    client = MagicMock()
    client.agenerate_completion = AsyncMock()
    return client

@pytest.mark.asyncio
async def test_generate_questions(mock_llm_client):
    agent = ClarificationAgent(mock_llm_client)
    
    # Mock LLM response with JSON format now required by analyze_status
    mock_llm_client.agenerate_completion.return_value = (
        '{"status": "REFINING", "questions": ["Q1?", "Q2?", "Q3?"], "estimated_turns_remaining": 1}'
    )
    
    questions = await agent.generate_questions("Build a todo app", num_questions=3)
    
    assert len(questions) == 3
    assert questions[0] == "Q1?"
    
    # Check if LLM was called
    assert mock_llm_client.agenerate_completion.called

@pytest.mark.asyncio
async def test_generate_questions_fallback(mock_llm_client):
    agent = ClarificationAgent(mock_llm_client)
    
    # Mock LLM response with invalid JSON to trigger fallback
    mock_llm_client.agenerate_completion.return_value = "Invalid Response"
    
    questions = await agent.generate_questions("Build a todo app", num_questions=1)
    
    # Fallback provides a specific question
    assert len(questions) == 1
    assert "architectural constraints" in questions[0]