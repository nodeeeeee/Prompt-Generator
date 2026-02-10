import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from src.prompt_builder import PromptBuilder

@pytest.mark.asyncio
async def test_project_context_pruning():
    mock_client = MagicMock()
    # 1. persona, 2. complexity, 3. optimization
    mock_client.agenerate_completion = AsyncMock(side_effect=[
        "Persona", "Complexity", "Final Clean Prompt"
    ])
    
    builder = PromptBuilder(mock_client)
    
    # Context containing a tree and a raw file block
    dirty_context = """
    Project Structure:
    ├── src
    │   └── main.py
    └── README.md
    
    ### SELECTED FILE CONTENTS
    --- FILE: src/main.py ---
    print("hello")
    --- END FILE ---
    
    Important Research Note: This algorithm is linear.
    """
    
    # We call build_prompt without auto_discover to test the pruning logic
    await builder.build_prompt(
        intention="Test pruning",
        answers=[],
        questions=[],
        mode="one-shot",
        project_context=dirty_context
    )
    
    # Verify the call to optimizer contained the note but not the tree
    args, kwargs = mock_client.agenerate_completion.call_args_list[-1]
    prompt_sent_to_optimizer = args[0][1]["content"] 
    
    assert "├──" not in prompt_sent_to_optimizer
    assert "--- FILE:" not in prompt_sent_to_optimizer
    assert "print(\"hello\")" not in prompt_sent_to_optimizer
    assert "Important Research Note" in prompt_sent_to_optimizer
    
@pytest.mark.asyncio
async def test_auto_discover_replaces_context():
    mock_client = MagicMock()
    mock_client.agenerate_completion = AsyncMock()
    # 1. persona, 2. complexity, 3. discovery-selection, 4. discovery-analyst, 5. discovery-synthesis, 6. optimization
    mock_client.agenerate_completion.side_effect = [
        "Persona", "Complexity", '["f1.py"]', "Analyst Insight", "THESE ARE THE INSIGHTS", "Final Prompt"
    ]
    
    builder = PromptBuilder(mock_client)
    
    dirty_context = "├── tree structure"
    
    # With auto_discover=True, the insights should REPLACE the dirty context
    with patch('src.prompt_builder.scan_directory', return_value="tree"):
        with patch('src.features.discovery_agent.read_project_file', return_value="content"):
            await builder.build_prompt(
                intention="Test replacement",
                answers=[],
                questions=[],
                auto_discover=True,
                root_path=".",
                project_context=dirty_context
            )
    
    # Verify the call to optimizer (the last one) contained the insights but not the tree
    args, kwargs = mock_client.agenerate_completion.call_args_list[-1]
    raw_prompt_in_user_content = args[0][1]["content"]
    
    assert "THESE ARE THE INSIGHTS" in raw_prompt_in_user_content
    assert "├──" not in raw_prompt_in_user_content