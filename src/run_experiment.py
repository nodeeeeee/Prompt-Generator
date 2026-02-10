import argparse
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features.robustness import ExperimentRunner, PerturbationEngine

def main():
    parser = argparse.ArgumentParser(description="Robustness Verification Experiment")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--trials", type=int, default=5, help="Trials per noise level")
    parser.add_argument("--output", type=str, default="results/robustness.jsonl", help="Output log file")
    
    args = parser.parse_args()
    
    print(f"[*] Starting Robustness Experiment (Seed: {args.seed})...")
    
    runner = ExperimentRunner(output_file=args.output)
    engine = PerturbationEngine(seed=args.seed)
    
    baseline_intention = "Create a distributed key-value store in Go using Raft consensus."
    
    # 1. Noise Sweep
    print("\\n[+] Phase 1: Noise Injection Sweep")
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.5]
    
    for noise in noise_levels:
        print(f"    Testing Noise Level: {noise} ({args.trials} trials)")
        crashes = 0
        invalids = 0
        
        for _ in range(args.trials):
            res = runner.run_trial(baseline_intention, noise_level=noise)
            if res["crashed"]: crashes += 1
            if not res["crashed"] and not res["valid"]: invalids += 1
            
        print(f"    -> Crashes: {crashes}, Invalid Outputs: {invalids}")

    # 2. Adversarial Sweep
    print("\\n[+] Phase 2: Adversarial Perturbation Sweep")
    attacks = engine.get_adversarial_strings()
    
    for attack in attacks:
        print(f"    Testing Attack: {attack[:30]}...")
        res = runner.run_trial(baseline_intention, adversarial_str=attack)
        status = "CRASHED" if res["crashed"] else ("INVALID" if not res["valid"] else "PASS")
        print(f"    -> Result: {status} (Duration: {res['duration']:.4f}s)")

    print(f"\\n[*] Experiment Complete. Results saved to {args.output}")

if __name__ == "__main__":
    main()
