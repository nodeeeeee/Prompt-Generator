import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock
from src.features.idea_generator import generate_idea_and_prompt, generate_idea_questions, generate_raw_idea

@pytest.fixture
def mock_client():
    client = MagicMock()
    client.agenerate_completion = AsyncMock()
    return client

@pytest.fixture
def mock_builder():
    builder = MagicMock()
    builder.build_prompt = AsyncMock(return_value="Built Prompt Content")
    return builder

@pytest.mark.asyncio
async def test_generate_raw_idea(mock_client):
    mock_client.agenerate_completion.return_value = "Brilliant Idea"
    res = await generate_raw_idea(mock_client, "context", "new features")
    assert res == "Brilliant Idea"

@pytest.mark.asyncio
async def test_generate_idea_questions(mock_client):
    mock_client.agenerate_completion.return_value = '["Q1", "Q2"]'
    res = await generate_idea_questions(mock_client, "context", "Idea", "new features")
    assert res == ["Q1", "Q2"]

@pytest.mark.asyncio
async def test_generate_idea_and_prompt_feature(mock_client, mock_builder):
    qa_history = [{"q": "Q1", "a": "A1"}]
    mock_builder.build_prompt.return_value = ("Built Prompt Content", ["file1.py"])
    prompt, disc = await generate_idea_and_prompt(
        mock_client, mock_builder, "context", "new features", "My Idea", qa_history
    )
    
    assert prompt == "Built Prompt Content"
    assert disc == ["file1.py"]
    args, kwargs = mock_builder.build_prompt.call_args
    assert kwargs["mode"] == "iterative"
    assert "Q: Q1" in kwargs["experiment_context"]

@pytest.mark.asyncio
async def test_generate_idea_and_prompt_experiment(mock_client, mock_builder):
    qa_history = [{"q": "Q1", "a": "A1"}]
    mock_builder.build_prompt.return_value = ("Built Prompt Content", [])
    prompt, disc = await generate_idea_and_prompt(
        mock_client, mock_builder, "context", "conduct experiment", "My Exp", qa_history
    )
    
    assert prompt == "Built Prompt Content"
    args, kwargs = mock_builder.build_prompt.call_args
    assert kwargs["mode"] == "chain-of-thought"
    assert "### Experimental Setup" in kwargs["experiment_context"]
    assert "Q: Q1" in kwargs["experiment_context"]

@pytest.mark.asyncio
async def test_generate_idea_and_prompt_auto_discover(mock_client, mock_builder):
    # This tests if the flags are passed correctly to the builder
    mock_builder.build_prompt.return_value = ("Built Prompt Content", ["auto.py"])
    prompt, disc = await generate_idea_and_prompt(
        mock_client, mock_builder, "context", "new features", "My Idea", 
        qa_history=[], root_path="/tmp", auto_discover=True
    )
    
    assert prompt == "Built Prompt Content"
    assert disc == ["auto.py"]
    args, kwargs = mock_builder.build_prompt.call_args
    assert kwargs["root_path"] == "/tmp"
    assert kwargs["auto_discover"] is True