import pytest
from unittest.mock import MagicMock, AsyncMock
from src.clarification_agent import ClarificationAgent

@pytest.fixture
def mock_llm_client():
    client = MagicMock()
    client.agenerate_completion = AsyncMock()
    return client

@pytest.mark.asyncio
async def test_analyze_status_refining(mock_llm_client):
    agent = ClarificationAgent(mock_llm_client)
    
    # Mock LLM response with JSON including estimated_turns_remaining
    mock_llm_client.agenerate_completion.return_value = (
        '{"status": "REFINING", "questions": ["What is the tech stack?"], '
        '"estimated_turns_remaining": 2, "rationale": "Need tech info"}'
    )
    
    result = await agent.analyze_status("Build an app")
    
    assert result["status"] == "REFINING"
    assert result["questions"] == ["What is the tech stack?"]
    assert result["estimated_turns_remaining"] == 2

@pytest.mark.asyncio
async def test_analyze_status_ready(mock_llm_client):
    agent = ClarificationAgent(mock_llm_client)
    
    mock_llm_client.agenerate_completion.return_value = (
        '{"status": "READY", "questions": [], "estimated_turns_remaining": 0, "rationale": "All clear"}'
    )
    
    result = await agent.analyze_status("Build a Flask app with PostgreSQL")
    
    assert result["status"] == "READY"
    assert result["estimated_turns_remaining"] == 0

@pytest.mark.asyncio
async def test_analyze_status_json_fluff(mock_llm_client):
    agent = ClarificationAgent(mock_llm_client)
    
    # Mock LLM response with some text before/after JSON
    mock_llm_client.agenerate_completion.return_value = (
        'Sure, here is the analysis: {"status": "READY", "questions": [], '
        '"estimated_turns_remaining": 0} Hope this helps!'
    )
    
    result = await agent.analyze_status("Build an app")
    
    assert result["status"] == "READY"
    assert result["estimated_turns_remaining"] == 0
