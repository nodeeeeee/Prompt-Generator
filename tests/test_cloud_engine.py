import pytest
import asyncio
from src.cloud_engine import CloudEngine, CloudState, CloudContext
from src.llm_integration import LLMClient
from unittest.mock import AsyncMock, MagicMock

@pytest.fixture
def mock_llm_client():
    client = MagicMock(spec=LLMClient)
    client.agenerate_completion = AsyncMock(return_value="Mocked response")
    return client

@pytest.mark.asyncio
async def test_cloud_engine_initialization():
    engine = CloudEngine()
    assert engine is not None
    health = engine.get_health()
    assert health["status"] == "OPERATIONAL"
    assert "architecture_analysis" in health["capabilities"]

@pytest.mark.asyncio
async def test_cloud_engine_pipeline_success(mock_llm_client):
    engine = CloudEngine(llm_client=mock_llm_client)
    
    intention = "Deploy a serverless API on AWS Lambda with DynamoDB."
    # mock_llm_client.agenerate_completion is already AsyncMock
    
    context = await engine.run_pipeline(intention, cloud_provider="aws")
    
    assert context.state == CloudState.COMPLETED
    assert context.final_prompt is not None
    assert context.metrics.total_duration_ms > 0
    assert context.cloud_provider == "aws"
    assert "architecture_analysis" in context.metadata
    assert "security_assessment" in context.metadata

@pytest.mark.asyncio
async def test_cloud_engine_cost_optimization_trigger(mock_llm_client):
    engine = CloudEngine(llm_client=mock_llm_client)
    
    intention = "Build a cost-optimized cluster."
    context = await engine.run_pipeline(intention)
    
    assert context.state == CloudState.COMPLETED
    assert context.metadata.get("financial_optimization_requested") is True
    assert "cost_optimization" in context.metadata
    assert context.metrics.cost_optimization_ms > 0

@pytest.mark.asyncio
async def test_cloud_engine_validation_error():
    engine = CloudEngine()
    # Too short intention
    context = await engine.run_pipeline("Too short")
    assert context.state == CloudState.FAILED
    assert any("too brief" in err["message"].lower() for err in context.error_trace)

@pytest.mark.asyncio
async def test_cloud_engine_repetitive_input():
    engine = CloudEngine()
    # Highly repetitive intention
    repetitive_intention = "word " * 50
    context = await engine.run_pipeline(repetitive_intention)
    assert context.state == CloudState.FAILED
    assert any("highly repetitive" in err["message"].lower() for err in context.error_trace)

@pytest.mark.asyncio
async def test_cloud_engine_state_transitions(mock_llm_client):
    engine = CloudEngine(llm_client=mock_llm_client)
    context = CloudContext(intention="Sample cloud intention for state test")
    
    # Test valid transition
    engine._update_state(context, CloudState.INITIALIZING)
    assert context.state == CloudState.INITIALIZING
    
    # Test illegal transition (IDLE -> COMPLETED is not allowed, must go through pipeline)
    # Actually INITIALIZING -> COMPLETED is not allowed
    with pytest.raises(Exception): # CloudStateError
        engine._update_state(context, CloudState.COMPLETED)

@pytest.mark.asyncio
async def test_cloud_engine_boundary_check():
    engine = CloudEngine()
    # Giant intention
    giant_intention = "Cloud " * 15001 # Exceeds 60000 chars if word is "Cloud " (6 chars)
    context = await engine.run_pipeline(giant_intention)
    assert context.state == CloudState.FAILED
    assert any("exceeds size limit" in err["message"].lower() for err in context.error_trace)