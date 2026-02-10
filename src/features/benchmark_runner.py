import time
import json
import os
import uuid
import subprocess
from typing import List, Dict, Any, Optional
from src.features.benchmark_variants import BENCHMARK_VARIANTS

class BenchmarkRunner:
    """
    Executes prompt-script variants using the Gemini CLI and collects high-fidelity metrics.
    """
    def __init__(self, output_file: str = "results/benchmark_results.jsonl", model: str = None, mock: bool = False):
        self.output_file = output_file
        self.model = model
        self.mock = mock
        # Hard limits to prevent dead loops - increased for deep reasoning
        self.MAX_TRIAL_TIMEOUT = 1200 # 20 minutes per trial
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

    def log_result(self, result: Dict[str, Any]):
        try:
            with open(self.output_file, "a") as f:
                f.write(json.dumps(result) + "\n")
        except Exception as e:
            print(f"[!] Logging error: {e}")

    def run_benchmark_trial(
        self, 
        variant_id: str, 
        intention: str, 
        context: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        if variant_id not in BENCHMARK_VARIANTS:
            raise ValueError(f"Unknown variant: {variant_id}")

        variant_config = BENCHMARK_VARIANTS[variant_id]
        if context and context.startswith("# ROLE"):
            prompt = f"{context}\n\nTask: {intention}"
        else:
            prompt = variant_config["func"](intention, context)
        
        run_id = str(uuid.uuid4())
        start_time = time.time()
        
        success = False
        output = ""
        error_type = None
        error_msg = ""
        tokens_out = 0
        
        try:
            if self.mock:
                time.sleep(0.5)
                output = f"MOCKED RESPONSE: {intention[:20]}"
                success = True
            else:
                cmd = ["gemini", "--prompt", prompt, "--output-format", "json", "--approval-mode", "yolo"]
                if self.model:
                    cmd.extend(["--model", self.model])
                
                # Execute command with Popen to allow heartbeat
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    stdin=subprocess.DEVNULL,
                    text=True
                )
                
                # Polling loop with heartbeat
                stdout = ""
                stderr = ""
                elapsed = 0
                while elapsed < self.MAX_TRIAL_TIMEOUT:
                    if process.poll() is not None:
                        stdout, stderr = process.communicate()
                        break
                    time.sleep(30)
                    elapsed += 30
                    print(f"      ... still reasoning ({elapsed}s)", flush=True)
                
                if process.poll() is None:
                    process.kill()
                    stdout, stderr = process.communicate()
                    error_type = "Timeout"
                    error_msg = f"Trial exceeded {self.MAX_TRIAL_TIMEOUT}s limit"
                elif process.returncode == 0:
                    try:
                        import re
                        json_match = re.search(r'(\{.*\})', stdout, re.DOTALL)
                        if json_match:
                            parsed = json.loads(json_match.group(1))
                            output = parsed.get("response", stdout)
                            stats = parsed.get("stats", {})
                            models_stats = stats.get("models", {})
                            if models_stats:
                                first_model = list(models_stats.values())[0]
                                tokens_out = first_model.get("tokens", {}).get("candidates", 0)
                        else:
                            output = stdout
                        success = True
                    except json.JSONDecodeError:
                        output = stdout
                        success = True
                else:
                    success = False
                    error_msg = stderr
                    error_type = "CLIFailure"
        
        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            
        duration = time.time() - start_time
        latency_ms = int(duration * 1000)
        
        format_adherence = self._check_format_adherence(variant_id, output)
        constraint_adherence = self._check_constraint_adherence(variant_id, output)
        
        stability_score = 0
        if success:
            stability_score += 40
            if format_adherence: stability_score += 30
            if constraint_adherence: stability_score += 30
            
            # Lenient penalty for deep reasoning models: 1 point per 10s
            penalty = (duration // 10) * 1
            stability_score = max(0, stability_score - penalty)

        result = {
            "run_id": run_id,
            "timestamp": time.time(),
            "variant_id": variant_id,
            "model": self.model or "default-cli",
            "metrics": {
                "success": success,
                "wall_time_ms": latency_ms,
                "tokens_out": tokens_out or len(output.split()),
                "error_type": error_type,
                "stability_score": stability_score
            },
            "error_msg": error_msg[:200] if error_msg else None,
            "output_preview": output[:500] if output else None
        }
        
        self.log_result(result)
        return result

    def _check_format_adherence(self, variant_id: str, output: str) -> bool:
        if not output: return False
        return len(output.strip()) > 0

    def _check_constraint_adherence(self, variant_id: str, output: str) -> bool:
        if not output: return False
        if variant_id == "robustness_hardened":
            return any(k in output for k in ["try:", "except", "validate"])
        return True
