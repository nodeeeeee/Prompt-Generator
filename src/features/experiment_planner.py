from typing import List, Optional

def generate_experiment_prompt_snippet(
    experiment_type: str, 
    parameters: Optional[str] = None, 
    hypothesis: Optional[str] = None
) -> str:
    """
    Generates a prompt section specific to experimental setups.
    """
    snippet = "\\n\\n### Experimental Setup\\n"
    
    if experiment_type == "ablation":
        snippet += "Goal: Perform an ablation study to verify the impact of specific components.\\n"
        if parameters:
            snippet += f"Target Components: {parameters}\\n"
        snippet += "Requirement: Generate a script that allows selectively disabling these components via command-line arguments or config.\\n"
        
    elif experiment_type == "hyperparameter_search":
        snippet += "Goal: Perform a hyperparameter search.\\n"
        if parameters:
            snippet += f"Search Space: {parameters}\\n"
        snippet += "Requirement: Create a training loop that iterates over these combinations and logs metrics (loss, accuracy, etc.) to a file or TensorBoard.\\n"
        
    elif experiment_type == "robustness_test":
        snippet += "Goal: Test model robustness against noise or adversarial inputs.\\n"
        snippet += "Requirement: Create a test script that adds noise/perturbations to the input data and evaluates performance degradation.\\n"
        
    if hypothesis:
        snippet += f"Hypothesis: {hypothesis}\\n"
        
    return snippet
