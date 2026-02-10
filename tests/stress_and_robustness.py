import pytest
import asyncio
import io
from unittest.mock import MagicMock, AsyncMock, patch
from src.features.pdf_parser import extract_text_from_pdf
from src.features.discovery_agent import DiscoveryAgent
from src.security_engine import SecurityEngine
from src.llm_integration import LLMClient, LLMServiceError

@pytest.mark.asyncio
async def test_pdf_parser_edge_cases():
    # Test with corrupted/empty data
    fake_file = io.BytesIO(b"not a pdf")
    res = extract_text_from_pdf(fake_file)
    assert "Error parsing PDF" in res

@pytest.mark.asyncio
async def test_discovery_agent_invalid_path():
    client = MagicMock(spec=LLMClient)
    agent = DiscoveryAgent(client)
    # Non-existent path should not crash
    res = await agent.investigate_and_analyze("/non/existent/path/999", "intent", "tree")
    assert "No relevant files" in res

@pytest.mark.asyncio
async def test_security_engine_extreme_input():
    engine = SecurityEngine()
    # Test very small input
    res = await engine.process_content("a")
    assert res.state.name == "FAILED" # Pydantic validator requires non-empty/non-whitespace and length > 3 context logic
    
    # Test huge input boundary
    huge_input = "A" * 600000 
    res = await engine.process_content(huge_input)
    assert res.state.name == "FAILED"
    assert "size exceeds" in res.error_trace[-1]["message"]

@pytest.mark.asyncio
async def test_llm_client_timeout_handling():
    client = LLMClient(timeout=1)
    messages = [{"role": "user", "content": "Hi"}]
    
    # Mock litellm.acompletion to hang
    with patch("litellm.acompletion", side_effect=asyncio.TimeoutError):
        with pytest.raises(LLMServiceError) as exc:
            await client.agenerate_completion(messages)
        assert "timed out" in str(exc.value)

@pytest.mark.asyncio
async def test_parallel_investigation_partial_failure():
    mock_client = MagicMock(spec=LLMClient)
    # Mock first call (file selection) success, but second call (analyst) failure
    mock_client.agenerate_completion.side_effect = [
        '["file1.py"]', 
        Exception("Analyst Crash"),
        "Synthesized even with failure"
    ]
    
    agent = DiscoveryAgent(mock_client)
    with patch('src.features.discovery_agent.read_project_file', return_value="content"):
        res = await agent.investigate_and_analyze(".", "intent", "tree")
        assert "Synthesized" in res
        assert "Analyst Crash" not in res # Should be handled gracefully inside synthesis
