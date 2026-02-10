import pytest
from unittest.mock import MagicMock
from src.llm_integration import LLMClient, LLMServiceError
import litellm

@pytest.fixture
def llm_client():
    return LLMClient()

def test_generate_completion_success(llm_client, mocker):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Hello world"))]
    
    mocker.patch("litellm.completion", return_value=mock_response)
    
    messages = [{"role": "user", "content": "Hi"}]
    response = llm_client.generate_completion(messages)
    
    assert response == "Hello world"
    litellm.completion.assert_called_once()

def test_generate_completion_custom_model(llm_client, mocker):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Test"))]
    
    mocker.patch("litellm.completion", return_value=mock_response)
    
    messages = [{"role": "user", "content": "Hi"}]
    llm_client.generate_completion(messages, model="claude-3-opus")
    
    args, kwargs = litellm.completion.call_args
    assert kwargs["model"] == "claude-3-opus"

def test_generate_completion_error(llm_client, mocker):
    mocker.patch("litellm.completion", side_effect=Exception("API Error"))
    
    with pytest.raises(LLMServiceError) as excinfo:
        llm_client.generate_completion([{"role": "user", "content": "Hi"}])
    
    assert "API Error" in str(excinfo.value)
