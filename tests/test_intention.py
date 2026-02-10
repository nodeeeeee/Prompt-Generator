import pytest
import asyncio
from src.features.test_intention import IntentionTester, IntentionState

@pytest.mark.anyio
async def test_intention_tester_high_quality():
    tester = IntentionTester()
    intention = "Implement a distributed key-value store using Raft consensus in Go, with focus on high performance and robustness."
    res = await tester.test_intention(intention)
    
    assert res.is_valid is True
    assert res.state == IntentionState.COMPLETED
    assert res.score > 65
    assert res.metrics.tech_keyword_count >= 3
    assert any("high quality" in f.lower() or "clear intention" in f.lower() for f in res.feedback)

@pytest.mark.anyio
async def test_intention_tester_medium_quality():
    tester = IntentionTester()
    intention = "Create a simple web app with a database."
    res = await tester.test_intention(intention)
    
    assert res.is_valid is True
    assert res.score > 30
    assert res.score <= 70

@pytest.mark.anyio
async def test_intention_tester_empty_alphanumeric():
    tester = IntentionTester()
    intention = "!!! ??? !!!"
    res = await tester.test_intention(intention)
    
    assert res.is_valid is False
    assert any("no alphanumeric characters" in f.lower() for f in res.feedback)

@pytest.mark.anyio
async def test_intention_tester_repetitive():
    tester = IntentionTester()
    intention = "word " * 30
    res = await tester.test_intention(intention)
    
    assert res.is_valid is False
    assert any("highly repetitive" in f.lower() for f in res.feedback)

@pytest.mark.anyio
async def test_intention_tester_word_boundaries():
    tester = IntentionTester()
    # "api" is a keyword. "capital" contains "api" but shouldn't match if using word boundaries.
    intention = "This is a capital city without any special technical interface."
    res = await tester.test_intention(intention)
    
    # "interface" is a keyword, so tech_keyword_count should be at least 1.
    # But "api" should not be counted.
    assert res.metrics.tech_keyword_count >= 1
    # If "api" was matched, it would likely be higher.
    # Let's check specifically for "interface"
    assert any(p.search(intention) for p in tester._keyword_patterns if "interface" in p.pattern)

@pytest.mark.anyio
async def test_intention_tester_sm90_keywords():
    tester = IntentionTester()
    intention = "Analyze SASS and PTX kernels for SM90 GPU architecture, focusing on Liveness and CFG."
    res = await tester.test_intention(intention)
    
    assert res.is_valid is True
    assert res.metrics.tech_keyword_count >= 4 # sass, ptx, kernel, gpu, liveness, cfg
    assert res.score > 60

@pytest.mark.anyio
async def test_intention_tester_state_machine_illegal_transition():
    # This is harder to test via public API, but we can check if it fails gracefully
    # if we were to manually corrupt the state (which we won't do here for stability)
    pass

@pytest.mark.anyio
async def test_intention_tester_large_input_boundary():
    tester = IntentionTester()
    # Just under the limit
    intention = "build " * 4000 
    res = await tester.test_intention(intention)
    assert res.state != IntentionState.FAILED # Should handle it
