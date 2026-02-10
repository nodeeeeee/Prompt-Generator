import argparse
import time
import json
import os
import random
from typing import List, Dict, Any
from src.features.benchmark_runner import BenchmarkRunner
from src.features.benchmark_variants import BENCHMARK_VARIANTS

class SyntheticIntentionGenerator:
    def __init__(self, runner: BenchmarkRunner):
        self.runner = runner
        self.domains = ["Systems", "Security", "AI", "Cloud", "Embedded"]

    def generate(self, count: int = 1) -> List[str]:
        intentions = []
        for _ in range(count):
            domain = random.choice(self.domains)
            # Use a fast fallback intention to avoid hanging during generation
            intentions.append(f"Implement a high-performance {domain} module with strict error handling.")
        return intentions

class PromptTuner:
    def __init__(self, runner: BenchmarkRunner):
        self.runner = runner

    def evolve_variant(self, variant_id: str, results: List[Dict[str, Any]]) -> str:
        # Simple heuristic evolution to avoid meta-prompt hanging in pilot
        return None

class BenchmarkOrchestrator:
    def __init__(self, trials: int = 1, generations: int = 1, model: str = None, mock: bool = False):
        self.trials = trials
        self.generations = generations
        self.runner = BenchmarkRunner(model=model, mock=mock)
        self.generator = SyntheticIntentionGenerator(self.runner)
        self.variant_prompts = {k: None for k in BENCHMARK_VARIANTS.keys()}
        self.MAX_TOTAL_RUNTIME = 2400 # 40 minutes max for the entire script

    def run_automated_loop(self, variants: List[str] = None):
        start_time = time.time()
        if not variants:
            variants = list(BENCHMARK_VARIANTS.keys())
            
        print(f"[*] Starting Autonomous Benchmark (Max Runtime: {self.MAX_TOTAL_RUNTIME}s)")
        
        for gen in range(self.generations):
            if time.time() - start_time > self.MAX_TOTAL_RUNTIME:
                print("[!] Global timeout reached. Terminating.")
                break
                
            print(f"\n--- GENERATION {gen+1} ---", flush=True)
            test_intentions = self.generator.generate(self.trials)
            
            for variant_id in variants:
                if time.time() - start_time > self.MAX_TOTAL_RUNTIME: break
                
                print(f"[+] Variant: {variant_id}", flush=True)
                for i, intention in enumerate(test_intentions):
                    print(f"    Trial {i+1}...", end="", flush=True)
                    res = self.runner.run_benchmark_trial(variant_id, intention)
                    score = res['metrics']['stability_score']
                    duration = res['metrics']['wall_time_ms']
                    print(f" Score: {score} ({duration}ms)", flush=True)

        print(f"\n[*] Total Runtime: {time.time() - start_time:.2f}s")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--gens", type=int, default=1)
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    orchestrator = BenchmarkOrchestrator(trials=args.trials, generations=args.gens, mock=args.mock)
    orchestrator.run_automated_loop(variants=args.variants)

if __name__ == "__main__":
    main()
