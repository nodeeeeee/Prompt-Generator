import asyncio
import os
import sys
from typing import Any
from unittest.mock import MagicMock, AsyncMock

# Add project root
sys.path.append(os.getcwd())

from src.features.test_intention import IntentionTester
from src.systems_engine import SystemsEngine
from src.security_engine import SecurityEngine
from src.features.file_interface import read_project_file

async def chaos_monkey():
    print("🚀 Starting Chaos Monkey Stress Test...")
    
    # Create a mock client that always returns a valid string
    mock_client = MagicMock()
    mock_client.agenerate_completion = AsyncMock(return_value="Valid Mocked AI Response")
    
    # Inject it directly
    it = IntentionTester()
    se = SecurityEngine()
    sys_engine = SystemsEngine(llm_client=mock_client)

    print("\n[Test 1] Empty Input Handling...")
    res = await it.test_intention("   ")
    print(f"  Result: State={res.state.name}, Valid={res.is_valid}")

    print("\n[Test 2] Extreme Length Buffer Stress...")
    giant_input = "Build this " * 50000 
    res = await se.process_content(giant_input)
    print(f"  Result: State={res.state.name}")

    print("\n[Test 3] State Machine Integrity...")
    from src.security_engine import SecurityState, SecurityContext
    ctx = SecurityContext(content="test")
    try:
        se._update_state(ctx, SecurityState.COMPLETED)
        print("  ❌ Bug: Illegal state transition allowed!")
    except Exception as e:
        print(f"  ✅ Correctly blocked: {e}")

    print("\n[Test 4] Path Traversal Deep Probe...")
    res = read_project_file(".", "../../../../../../../etc/passwd")
    if "Security violation" in res:
        print("  ✅ Path traversal blocked.")
    else:
        print(f"  ❌ Bug: Leak detected: {res[:50]}")

    print("\n[Test 5] Concurrency Stress...")
    tasks = [sys_engine.run_pipeline(f"Build system {i}") for i in range(5)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    success_count = 0
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            print(f"  ❌ Intent {i} crashed: {r}")
        elif r.state.name == "COMPLETED":
            success_count += 1
        else:
            print(f"  ❌ Intent {i} failed: {r.error_trace[-1] if r.error_trace else 'No trace'}")
            
    print(f"  Result: {success_count}/5 successful concurrent runs.")

if __name__ == "__main__":
    asyncio.run(chaos_monkey())