import pytest
from unittest.mock import patch, MagicMock
from src.features.benchmark_runner import BenchmarkRunner
from src.features.benchmark_variants import BENCHMARK_VARIANTS

@pytest.fixture
def mock_litellm():
    with patch("litellm.completion") as mock:
        yield mock

def test_benchmark_runner_success(mock_litellm):
    # Setup mock response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Success output with try-except"
    mock_response.usage.completion_tokens = 100
    mock_litellm.return_value = mock_response
    
    runner = BenchmarkRunner(output_file="results/test_results.jsonl")
    result = runner.run_benchmark_trial("robustness_hardened", "Test intention")
    
    assert result["metrics"]["success"] is True
    assert result["metrics"]["wall_time_ms"] >= 0
    assert result["metrics"]["stability_score"] > 0
    assert "robustness_hardened" == result["variant_id"]

def test_benchmark_runner_failure(mock_litellm):
    # Setup mock failure
    mock_litellm.side_effect = Exception("API Error")
    
    runner = BenchmarkRunner(output_file="results/test_results.jsonl")
    result = runner.run_benchmark_trial("baseline", "Test intention")
    
    assert result["metrics"]["success"] is False
    assert result["metrics"]["error_type"] == "Exception"
    assert result["metrics"]["stability_score"] == 0

def test_variant_generation():
    intention = "Test Intention"
    for variant_id, config in BENCHMARK_VARIANTS.items():
        prompt = config["func"](intention)
        assert intention in prompt
        assert len(prompt) > 0

def test_tool_strict_format_check():
    runner = BenchmarkRunner()
    valid_json = '{"status": "success", "data": {}}'
    invalid_json = "Not a json"
    
    assert runner._check_format_adherence("tool_strict", valid_json) is True
    assert runner._check_format_adherence("tool_strict", invalid_json) is False
