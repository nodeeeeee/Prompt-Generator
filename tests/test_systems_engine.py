import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from src.systems_engine import SystemsEngine, EngineState, ValidationError, ComponentError

@pytest.fixture
def mock_llm_client():
    client = MagicMock()
    # Mocking both sync and async methods just in case
    client.agenerate_completion = AsyncMock(return_value="Mocked LLM Response")
    client.generate_completion = MagicMock(return_value="Mocked LLM Response")
    return client

@pytest.mark.asyncio
async def test_systems_engine_happy_path(mock_llm_client):
    """
    Test the engine with valid inputs.
    """
    engine = SystemsEngine(llm_client=mock_llm_client)
    intention = "Build a CLI tool for file management in Python."
    result = await engine.run_pipeline(intention, mode="one-shot")
    
    assert result.state == EngineState.COMPLETED
    assert result.final_prompt is not None
    assert "Mocked LLM Response" in result.final_prompt
    assert not result.error_trace

@pytest.mark.asyncio
async def test_systems_engine_validation_failure(mock_llm_client):
    """
    Test the engine with invalid inputs.
    """
    engine = SystemsEngine(llm_client=mock_llm_client)
    
    # Test empty intention - Pydantic will raise ValidationError during instantiation
    # but run_pipeline handles it.
    result = await engine.run_pipeline("", mode="one-shot")
    assert result.state == EngineState.FAILED
    assert any("Intention must not be empty" in str(err.get('message', '')) for err in result.error_trace)

    # Test short intention
    result = await engine.run_pipeline("Too short", mode="one-shot")
    assert result.state == EngineState.FAILED
    assert any("Intention is too brief" in str(err.get('message', '')) for err in result.error_trace)
    
    # Test invalid mode
    result = await engine.run_pipeline("Valid long intention for testing purposes.", mode="invalid-mode")
    assert result.state == EngineState.FAILED
    assert any("Invalid mode" in str(err.get('message', '')) for err in result.error_trace)

@pytest.mark.asyncio
async def test_systems_engine_boundary_checks(mock_llm_client):
    """
    Test the engine with large inputs.
    """
    engine = SystemsEngine(llm_client=mock_llm_client)
    large_intention = "A" * 26000 # Exceeds 25000 limit
    
    result = await engine.run_pipeline(large_intention)
    assert result.state == EngineState.FAILED
    print(f"DEBUG: error_trace content: {result.error_trace}")

@pytest.mark.asyncio
async def test_systems_engine_error_resilience(mock_llm_client):
    """
    Test that the engine handles component failures gracefully (e.g. clarification failure).
    """
    engine = SystemsEngine(llm_client=mock_llm_client)
    
    # Mock clarifier to fail
    engine.clarifier.generate_questions = AsyncMock(side_effect=Exception("Clarifier Crash"))
    
    result = await engine.run_pipeline("Test resilience with crashing clarifier in high performance mode")
    
    # It should still complete because clarification is a soft failure
    assert result.state == EngineState.COMPLETED
    assert any("clarification" == err.get('step') for err in result.error_trace)

@pytest.mark.asyncio
async def test_systems_engine_critical_failure(mock_llm_client):
    """
    Test that the engine fails when a critical component (Builder) crashes.
    """
    engine = SystemsEngine(llm_client=mock_llm_client)
    
    # Mock builder to fail critically
    engine.builder.build_prompt = AsyncMock(side_effect=Exception("Builder CRASH"))
    
    result = await engine.run_pipeline("Test critical failure in the building phase")
    
    assert result.state == EngineState.FAILED
    assert any("Building phase critical failure" in str(err.get('message', '')) for err in result.error_trace)

def test_systems_engine_sync_initialization():
    """
    Verify initialization works.
    """
    try:
        engine = SystemsEngine()
        assert engine is not None
        health = engine.get_health()
        assert health['status'] == "OPERATIONAL"
    except Exception as e:
        pytest.fail(f"Initialization failed: {e}")
