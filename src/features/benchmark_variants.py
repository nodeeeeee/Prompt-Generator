from typing import List, Optional, Dict

def get_variant_baseline(intention: str, context: str = "") -> str:
    return f"""# ROLE
Expert Software Engineer

# TASK
{intention}

# CONTEXT
{context}

# OUTPUT
Complete implementation. Concise, no preamble."""

def get_variant_robustness_hardened(intention: str, context: str = "") -> str:
    return f"""# ROLE\nRobust Systems Engineer\n\n# MISSION\nImplement: {intention}\n\n# CONTEXT\n{context}\n\n# ROBUSTNESS CONSTRAINTS\n1. ERROR HANDLING: Every function must have explicit try-except blocks.\n2. VALIDATION: Validate all inputs before processing.\n3. BOUNDARIES: Ensure no buffer overflows or recursion depth issues.\n4. STATE MANAGEMENT: Use a clean state machine for complex logic.\n\n# EXECUTION\nProvide the full implementation with a focus on edge cases and failure modes."""

def get_variant_tool_strict(intention: str, context: str = "") -> str:
    return f"""# ROLE\nTechnical Architect (JSON Schema Specialist)\n\n# MISSION\nImplement: {intention}\n\n# CONTEXT\n{context}\n\n# FORMATTING REQUIREMENT\nYour entire output MUST be a valid JSON object following this schema:\n{{\n  "status": "success" | "error",\n  "data": {{\n    "files": [\n      {{ "path": "string", "content": "string" }}\n    ],\n    "tests": "string",\n    "documentation": "string"\n  }},\n  "metrics": {{ "complexity": number }}\n}}\n\n# MANDATE\nDo not include any text outside the JSON block. Ensure all code is properly escaped."""

def get_variant_long_run_stabilizer(intention: str, context: str = "") -> str:
    return f"""# ROLE\nLong-Term Project Lead\n\n# MISSION\nImplement: {intention}\n\n# LONG-RUN STABILITY PROTOCOL\n1. CONTEXT HYGIENE: Periodically summarize the architectural state.\n2. MEMORY MANAGEMENT: Avoid redundant declarations.\n3. INCREMENTAL VERIFICATION: Build and verify each module sequentially.\n4. DRIFT PREVENTION: Realign if MISSION drift is detected.\n\n# CONTEXT\n{context}\n\n# EXECUTION\nProceed following the stability protocol."""

# This registry can be updated during runtime by the Tuner
BENCHMARK_VARIANTS = {
    "baseline": {"func": get_variant_baseline, "hypothesis": "Baseline script."},
    "robustness_hardened": {"func": get_variant_robustness_hardened, "hypothesis": "Error-handling focus."},
    "tool_strict": {"func": get_variant_tool_strict, "hypothesis": "JSON enforcement."},
    "long_run_stabilizer": {"func": get_variant_long_run_stabilizer, "hypothesis": "Stability focus."}
}