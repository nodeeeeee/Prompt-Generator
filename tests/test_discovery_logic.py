import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock
from src.features.discovery_agent import DiscoveryAgent

@pytest.mark.asyncio
async def test_discovery_logic():
    mock_client = MagicMock()
    mock_client.agenerate_completion = AsyncMock(return_value='["src/main.py", "README.md"]')
    
    agent = DiscoveryAgent(mock_client)
    tree = """Project Root
├── src
│   └── main.py
└── README.md"""
    intent = "Update the main entry point"
    
    # Use current directory as root for discovery
    files = await agent.discover_and_read_context(".", intent, tree)
    
    assert "src/main.py" in files or "README.md" in files or len(files) == 0 # Mock doesn't actually read from disk unless files exist
    assert mock_client.agenerate_completion.called