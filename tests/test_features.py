import pytest
import os
from src.features.context_manager import scan_directory, read_key_files
from src.features.experiment_planner import generate_experiment_prompt_snippet

def test_scan_directory():
    # Use the current directory as test
    cwd = os.getcwd()
    tree = scan_directory(cwd, max_depth=1)
    
    assert "Project Root" in tree
    assert "src" in tree
    assert "tests" in tree
    assert "docker-compose.yml" in tree

def test_read_key_files():
    # We know README.md and requirements.txt exist
    cwd = os.getcwd()
    files = read_key_files(cwd)
    
    assert "README.md" in files
    assert "requirements.txt" in files
    assert len(files["README.md"]) > 0

def test_experiment_planner():
    snippet = generate_experiment_prompt_snippet(
        "ablation", 
        parameters="LayerNorm", 
        hypothesis="Removing LayerNorm degrades performance"
    )
    
    assert "Ablation Setup" in snippet or "Experimental Setup" in snippet
    assert "LayerNorm" in snippet
    assert "Hypothesis: Removing LayerNorm" in snippet
