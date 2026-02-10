import pytest
from src.features.robustness import PerturbationEngine, ExperimentRunner, RobustnessOracle

def test_perturbation_engine():
    engine = PerturbationEngine(seed=123)
    text = "Hello World"
    # With 0.0 noise, should be same
    assert engine.inject_noise(text, 0.0) == text
    
    # With high noise, should be different
    perturbed = engine.inject_noise(text, 0.5)
    assert perturbed != text
    assert len(perturbed) > 0

def test_experiment_runner_baseline():
    runner = ExperimentRunner()
    # Run baseline (0 noise)
    result = runner.run_trial("Create a web app", noise_level=0.0)
    
    assert result["crashed"] is False
    assert result["valid"] is True
    assert result["error"] == ""

def test_experiment_runner_high_noise():
    runner = ExperimentRunner()
    result = runner.run_trial("Create a web app", noise_level=0.5)
    
    # Even with noise, the system shouldn't crash (mock LLM handles strings)
    assert result["crashed"] is False
    # Validity might fail if noise destroys the prompt structure (unlikely with this MockLLM implementation,
    # as MockLLM returns a fixed string structure, but let's see).
    # Actually, MockLLM returns "MOCKED_LLM_RESPONSE: ...".
    # PromptBuilder wraps this.
    # RobustnessOracle looks for "# ROLE".
    # Since PromptBuilder templates inject "# ROLE", it should be valid unless PromptOptimizer fails heavily.
    # PromptOptimizer also calls LLM.
    assert result["valid"] is True 

def test_oracle():
    valid_prompt = "# ROLE\\nAI\\n# MISSION\\nDo it."
    invalid_prompt = "Just some random text."
    
    assert RobustnessOracle.is_structurally_valid(valid_prompt) is True
    assert RobustnessOracle.is_structurally_valid(invalid_prompt) is False
