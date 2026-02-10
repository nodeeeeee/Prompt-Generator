from typing import List, Optional

def get_one_shot_template(
    persona: str,
    intention: str,
    qa_context: str,
    project_context: Optional[str] = None,
    experiment_context: Optional[str] = None
) -> str:
    env_section = f"\n# ENVIRONMENT CONTEXT\n{project_context}" if project_context else ""
    exp_section = f"\n# EXPERIMENT SPEC\n{experiment_context}" if experiment_context else ""

    return f"""# ROLE
{persona}

# MISSION
Directly architect and implement the final production version of: **"{intention}"**

# SPECIFICATIONS
{qa_context}
{env_section}
{exp_section}

# MANDATORY DEVELOPMENT STANDARDS
1. **Comprehensive Implementation**: Deliver a complete, ready-to-use solution.
2. **Unit Testing**: Include a full suite of unit tests verifying core logic.
3. **Documentation**: Use high-quality comments for architectural decisions.
4. **Validation**: Describe a verification protocol for the final output.

Think once, deeply, and provide the complete production code.
"""

def get_iterative_long_form_template(
    persona: str,
    intention: str,
    qa_context: str,
    complexity: str,
    project_context: Optional[str] = None,
    experiment_context: Optional[str] = None
) -> str:
    """
    Template for the 'Iterative' mode: Limited initial planning, implement, then supplement planning.
    Designed for long-duration execution and creativity utilization.
    """
    env_section = f"\n# PROJECT CONTEXT\n{project_context}" if project_context else ""
    exp_section = f"\n# EXPERIMENT CONTEXT\n{experiment_context}" if experiment_context else ""

    return f"""# ROLE
{persona}

# OBJECTIVE
**"{intention}"**

# DEVELOPMENT MODE: CREATIVE ITERATIVE EVOLUTION
You are to build this project through an evolving, iterative process. Do NOT attempt to plan the entire system at the start. Instead, utilize your creativity to allow the architecture to emerge from implementation.

## THE ITERATIVE PROTOCOL
For every module or feature you build, follow this **Atomic Cycle**:
1. **Targeted Planning**: Plan ONLY the current module based on the existing implementation and immediate needs.
2. **Implementation**: Write the code for the module with production-grade comments.
3. **Unit Testing**: Immediately write unit tests for the module.
4. **Pilot & Refine**: Conduct a "Pilot Run" of the module. Generate a report of the results/feedback. Refine the module based on this feedback until it is robust.
5. **Evolutionary Step**: Based on what you just built, re-evaluate the next step. Supplement your overall plan with new creative ideas that occurred to you during implementation.

## CORE REQUIREMENTS
- **Complexity Assessment**: {complexity}
{qa_context}
{env_section}
{exp_section}

# YOUR MANDATE
This process should last for a long duration to fully exhaust your creative potential. Do not rush to finish. Build, verify, learn, and evolve.

Start with a `<thinking>` block to identify the very first atomic module to implement.
"""

def get_detailed_cot_template(
    persona: str,
    intention: str,
    qa_context: str,
    complexity: str,
    project_context: Optional[str] = None,
    experiment_context: Optional[str] = None
) -> str:
    """
    Template for 'CoT' mode: Extremely careful research and long planning before implementation.
    """
    env_section = f"\n# PROJECT CONTEXT\n{project_context}" if project_context else ""
    exp_section = f"\n# EXPERIMENT CONTEXT\n{experiment_context}" if experiment_context else ""

    return f"""# ROLE
{persona}

# MISSION-CRITICAL OBJECTIVE
**"{intention}"** (Complexity Profile: {complexity})

# DEVELOPMENT MODE: EXHAUSTIVE RESEARCH & ARCHITECTURAL DESIGN
You are to spend a significant amount of time in the **Research and Planning** phase before writing any production code. This is a high-stakes CS project requiring total precision.

## PHASE 1: SYSTEMIC RESEARCH & PLANNING
1. **Deep Thinking**: Conduct an exhaustive internal monologue in a `<thinking>` block. Analyze requirement ontologies, trade-offs (Pareto Optimization), and potential failure modes.
2. **Detailed Blueprint**: Create a granular architectural map including module hierarchies, data schemas, and state transition models.
3. **Pilot Verification Strategy**: Define exactly how each module will be pilot-tested and refined.

## PHASE 2: RIGOROUS IMPLEMENTATION
Only after the plan is fully solidified, begin implementation following these standards:
1. **Test-Driven Rigor**: Every module MUST have corresponding unit tests.
2. **Module-Level Verification**: For every module: Implement -> Pilot Run -> Get Report -> Refine.
3. **Technical Documentation**: Write necessary comments explaining the "Why" behind the logic.

# DATA DISCOVERY
{qa_context}
{env_section}
{exp_section}

# FINAL INSTRUCTION
This project requires long-term thinking and fine-grained results. Do not provide a generic solution. Be pedantic about quality and scientific accuracy.

Start with your exhaustive `<thinking>` process.
"""

def get_research_experiment_template(
    persona: str,
    intention: str,
    qa_context: str,
    project_context: str,
    experiment_context: str
) -> str:
    """
    Specialized template for CS researchers doing experiments on existing codebases.
    """
    template = f"""# IDENTITY
{persona}

# RESEARCH OBJECTIVE
You are assisting in a Computer Science research experiment. 
Task: {intention}

# EXPERIMENTAL SETUP & CONTEXT: PILOT & VERIFY
The current project environment is detailed below:
{project_context}

The specific experimental requirements are:
{experiment_context}

Additional Clarifications:
{qa_context}

# EXECUTION FRAMEWORK
As a research assistant, your goal is reproducibility and scientific rigor. Follow these steps:

1. **HYPOTHESIS ANALYSIS**: Analyze how the requested changes align with the experimental hypothesis. Identify potential confounding variables.
2. **IMPACT ASSESSMENT**: Scan the existing project structure. Which files and modules will be affected?
3. **PILOT IMPLEMENTATION**: Provide a pilot implementation for 1-2 representative assets before full-scale processing.
4. **MODULAR MODIFICATION**:
   - Implement the experimental changes in a way that is "pluggable".
   - Use configuration flags (e.g., argparse, yaml configs) to enable/disable experimental features.
   - Ensure metrics collection is integrated into the core loop.
5. **REPRODUCIBILITY SCRIPT**: Generate a `run_experiment.sh` or similar script that automates the execution of the full experiment pipeline.

# OUTPUT FORMAT
- Start with a technical summary of the plan.
- Provide the full source code for modified and new files.
- End with instructions on how to run the experiment and interpret results.

Think step-by-step and prioritize correctness over speed.
"""
    return template