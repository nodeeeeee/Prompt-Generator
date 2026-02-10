import asyncio
import os
import sys
import logging
import json
import time
from typing import List, Dict, Any
from src.llm_integration import LLMClient
from src.clarification_agent import ClarificationAgent
from src.prompt_builder import PromptBuilder
from src.features.research_journal import ResearchJournal, ResearchEntry
from src.security_engine import SecurityEngine

# Initialize results dir if missing
os.makedirs("results", exist_ok=True)

# High-fidelity logging
logger = logging.getLogger("MassSimulation")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

file_handler = logging.FileHandler("results/mass_simulation.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

SCENARIOS = [
    {"domain": "Systems", "intent": "Implement a lock-free work-stealing scheduler in C++ for real-time video processing.", "mode": "chain-of-thought", "creative": True},
    {"domain": "Distributed", "intent": "Build a Byzantine Fault Tolerant consensus algorithm for a permissioned blockchain in Go.", "mode": "iterative", "creative": False},
    {"domain": "Security", "intent": "Design an eBPF-based observability tool for tracking TCP retransmissions in a Kubernetes cluster.", "mode": "one-shot", "creative": True},
    {"domain": "ML", "intent": "Create a zero-knowledge proof system for verifying set membership in Rust.", "mode": "chain-of-thought", "creative": False},
    {"domain": "Cloud", "intent": "Develop a serverless auto-scaling engine for AWS Lambda based on custom SQS metrics.", "mode": "iterative", "creative": True},
    {"domain": "Storage", "intent": "Implement a persistent, sharded KV-store with snapshot isolation using LSM-trees.", "mode": "chain-of-thought", "creative": True},
    {"domain": "OS", "intent": "Design a micro-kernel for an ARM-based IoT device focusing on resource isolation.", "mode": "iterative", "creative": False},
    {"domain": "Security", "intent": "Build an automated fuzzer for detecting race conditions in multi-threaded Java applications.", "mode": "one-shot", "creative": True},
    {"domain": "ML", "intent": "Create a transformer-based model architecture optimized for low-power edge deployment.", "mode": "chain-of-thought", "creative": True},
    {"domain": "Networking", "intent": "Implement a RDMA-enabled key-value store for ultra-low latency data centers.", "mode": "iterative", "creative": True},
    {"domain": "Cryptography", "intent": "Design a side-channel resistant implementation of AES-256 for embedded systems.", "mode": "chain-of-thought", "creative": False},
    {"domain": "Compilers", "intent": "Build a compiler frontend for a domain-specific language used in quantum circuit simulation.", "mode": "one-shot", "creative": True},
    {"domain": "Systems", "intent": "Implement a RCU-protected radix tree for high-performance IP lookup.", "mode": "iterative", "creative": True},
    {"domain": "ML", "intent": "Create a federated learning protocol that handles non-IID data distributions in medical networks.", "mode": "chain-of-thought", "creative": True},
    {"domain": "Privacy", "intent": "Implement a differential privacy layer for SQL queries on sensitive census data.", "mode": "iterative", "creative": False},
    {"domain": "Formal Methods", "intent": "Develop a formal verification script using TLA+ for a distributed locking mechanism.", "mode": "chain-of-thought", "creative": True},
    {"domain": "Cloud", "intent": "Design a multi-tenant isolation layer for a Postgres-compatible cloud database.", "mode": "iterative", "creative": True},
    {"domain": "Hardware", "intent": "Create a hardware-in-the-loop simulation environment for autonomous drone flight control.", "mode": "one-shot", "creative": False},
    {"domain": "Algorithms", "intent": "Design a cache-oblivious data structure for large-scale graph traversal.", "mode": "chain-of-thought", "creative": True},
    {"domain": "Cloud Native", "intent": "Build a network-bootable (PXE) thin-client OS based on Alpine Linux.", "mode": "iterative", "creative": True},
    {"domain": "Security", "intent": "Develop a secure enclave (SGX) based confidential computing layer for Python dataframes.", "mode": "chain-of-thought", "creative": True},
    {"domain": "Networking", "intent": "Implement a high-performance HTTP/3 proxy with custom congestion control.", "mode": "iterative", "creative": True},
    {"domain": "Big Data", "intent": "Build a real-time stream processing engine using vectorized execution in C++.", "mode": "chain-of-thought", "creative": True},
    {"domain": "Systems", "intent": "Design a customized garbage collector for a low-latency trading system in Rust.", "mode": "iterative", "creative": True}
]

async def run_scenario(client, clarifier, builder, journal, scenario: Dict[str, Any]):
    domain = scenario["domain"]
    intent = scenario["intent"]
    mode = scenario["mode"]
    creative = scenario["creative"]
    
    logger.info(f"▶️ RUNNING: [{domain}] intent='{intent[:50]}...' mode={mode} creative={creative}")
    
    start_time = time.perf_counter()
    try:
        # 1. Clarification & Self-Answer (if creative)
        questions = []
        answers = []
        
        if creative:
            logger.info(f"  [{domain}] Step 1: Self-Answering...")
            status_res = await clarifier.analyze_status(intent)
            qa_history = await clarifier.self_answer_questions(intent, status_res['questions'])
            questions = [i['q'] for i in qa_history]
            answers = [i['a'] for i in qa_history]
        else:
            logger.info(f"  [{domain}] Step 1: Manual Clarification...")
            answers = ["Use standard libraries where possible", "Priority on safety"]
            questions = ["Q1: Preferred libraries?", "Q2: Performance vs Safety?"]

        # 2. Build Prompt
        logger.info(f"  [{domain}] Step 2: Building Prompt...")
        final_prompt, _ = await builder.build_prompt(intent, answers, questions, mode=mode)
        
        # 3. Security Check
        logger.info(f"  [{domain}] Step 3: Privacy Audit...")
        sec = SecurityEngine()
        await sec.process_content(final_prompt)

        # 4. Journaling
        journal.add_entry(ResearchEntry(
            intention=intent,
            mode=mode,
            final_prompt=final_prompt,
            tags=[domain, "batch-test", "creative" if creative else "manual"]
        ))
        
        duration = time.perf_counter() - start_time
        logger.info(f"✅ SUCCESS: [{domain}] duration={duration:.2f}s prompt_len={len(final_prompt)}")
        return True
    except Exception as e:
        logger.error(f"❌ FAILED: [{domain}] error={e}")
        return False

async def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        return

    client = LLMClient(default_model="o3-mini", timeout=180)
    clarifier = ClarificationAgent(client)
    builder = PromptBuilder(client)
    journal = ResearchJournal("results/batch_simulation_journal.json")
    
    success_count = 0
    total = len(SCENARIOS)
    
    logger.info(f"🚀 Starting Mass Batch Simulation of {total} Scenarios...")
    
    # Run in small batches
    batch_size = 2 
    for i in range(0, total, batch_size):
        current_batch = SCENARIOS[i : i + batch_size]
        tasks = [run_scenario(client, clarifier, builder, journal, s) for s in current_batch]
        results = await asyncio.gather(*tasks)
        success_count += sum(1 for r in results if r)
        logger.info(f"--- Progress: {min(i + batch_size, total)}/{total} ---")

    logger.info(f"\n✨ BATCH SIMULATION COMPLETE. Success Rate: {success_count}/{total}")

if __name__ == "__main__":
    asyncio.run(main())