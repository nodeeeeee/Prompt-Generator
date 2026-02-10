#!/bin/bash
set -e

# Create results directory
mkdir -p results

echo "========================================================"
echo "   ROBUSTNESS VERIFICATION EXPERIMENT (RESEARCH TRACK)  "
echo "========================================================"
echo "Date: $(date)"
echo "Repo: Prompt Generator"
echo "--------------------------------------------------------"

# 1. Run Unit Tests (Pilot Verification)
echo "[*] Step 1: Verifying Test Harness (Unit Tests)..."
python -m pytest tests/test_robustness_pilot.py -v

# 2. Run Full Experiment
echo ""
echo "[*] Step 2: Executing Main Robustness Sweep..."
python src/run_experiment.py --seed 1234 --trials 10 --output results/experiment_run.jsonl

# 3. Run Prompt Variant Benchmark (Pilot Run)
echo ""
echo "[*] Step 3: Executing Prompt Variant Benchmark (Pilot)..."
python src/run_benchmark.py --trials 1 --variants baseline robustness_hardened

# 4. Generate Summary Report (Simple Grep/Awk stats)
echo ""
echo "[*] Step 4: Experiment Summary"
echo "Total Trials:"
wc -l results/experiment_run.jsonl
echo "Crash Count:"
grep '"crashed": true' results/experiment_run.jsonl | wc -l
echo "Validation Failure Count:"
grep '"valid": false' results/experiment_run.jsonl | wc -l

echo ""
echo "[SUCCESS] Experiment completed. Artifacts in results/"
