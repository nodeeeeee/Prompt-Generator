from typing import List, Optional

def get_one_shot_template(
    persona: str,
    intention: str,
    qa_context: str,
    project_context: Optional[str] = None,
    experiment_context: Optional[str] = None
) -> str:
    insight_section = f"\n# ARCHITECTURE & TECHNICAL INSIGHTS\n{project_context}" if project_context else ""
    exp_section = f"\n# EXPERIMENTAL SPECIFICATIONS\n{experiment_context}" if experiment_context else ""

    return f"""# MISSION
Directly architect and implement the final production version of: **"{intention}"**

# ROLE
{persona}

# SPECIFICATIONS
{qa_context}
{insight_section}
{exp_section}

# IMPLEMENTATION PROTOCOL
1. **Comprehensive Implementation**: Deliver a complete, ready-to-use solution.
2. **Unit Testing**: Include a full suite of unit tests verifying core logic.
3. **Documentation**: Use high-quality comments for architectural decisions.

# EVALUATION
Describe a rigorous verification protocol for the final output.

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
    insight_section = f"\n# ARCHITECTURE & TECHNICAL INSIGHTS\n{project_context}" if project_context else ""
    exp_section = f"\n# EXPERIMENTAL SPECIFICATIONS\n{experiment_context}" if experiment_context else ""

    return f"""# MISSION
**"{intention}"**

# ROLE
{persona}

# IMPLEMENTATION PROTOCOL: ATOMIC CYCLE
For every module or feature you build, follow this:
1. **Targeted Planning**: Plan ONLY the current module.
2. **Implementation**: Write code with production-grade comments.
3. **Unit Testing**: Immediately write unit tests.
4. **Pilot & Refine**: Conduct a "Pilot Run". Refine based on results.
5. **Evolution**: Supplement your overall plan with new creative ideas.

# SPECIFICATIONS
- **Complexity Assessment**: {complexity}
{qa_context}
{insight_section}
{exp_section}

# EVALUATION
Include a formal benchmarking section for every atomic cycle.

Start with a `<thinking>` block to identify the first module.
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
    insight_section = f"\n# ARCHITECTURE & TECHNICAL INSIGHTS\n{project_context}" if project_context else ""
    exp_section = f"\n# EXPERIMENTAL SPECIFICATIONS\n{experiment_context}" if experiment_context else ""

    return f"""# MISSION
**"{intention}"** (Complexity Profile: {complexity})

# ROLE
{persona}

# IMPLEMENTATION PROTOCOL: RESEARCH-DRIVEN
## PHASE 1: SYSTEMIC RESEARCH & PLANNING
1. **Deep Thinking**: Exhaustive analysis in a `<thinking>` block.
2. **Detailed Blueprint**: Create a granular architectural map.
3. **Pilot Verification Strategy**: Define verification protocols.

## PHASE 2: RIGOROUS IMPLEMENTATION
1. **Test-Driven Rigor**: Every module MUST have unit tests.
2. **Module-Level Verification**: Implement -> Pilot Run -> Refine.

# SPECIFICATIONS
{qa_context}
{insight_section}
{exp_section}

# EVALUATION
Include a comprehensive formal benchmarking and evaluation section.

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
    template = f"""# MISSION
**"{intention}"**

# ROLE
{persona}

# ARCHITECTURE & TECHNICAL INSIGHTS
{project_context}

# EXPERIMENTAL SPECIFICATIONS
{experiment_context}

# SPECIFICATIONS: CLARIFICATIONS
{qa_context}

# IMPLEMENTATION PROTOCOL: RESEARCH RIGOR
As a research assistant, your goal is reproducibility and scientific rigor. Follow these steps:

1. **HYPOTHESIS ANALYSIS**: Analyze how the requested changes align with the experimental hypothesis.
2. **IMPACT ASSESSMENT**: Identify affected files and modules.
3. **PILOT IMPLEMENTATION**: Provide a pilot implementation for 1-2 representative assets.
4. **MODULAR MODIFICATION**: Implement changes in a "pluggable" way with configuration flags.
5. **REPRODUCIBILITY**: Generate a script (e.g., `run_experiment.sh`) that automates the full pipeline.

# EVALUATION
Include a detailed methodology for verifying results and ensuring reproducibility.

Think step-by-step and prioritize correctness over speed.
"""
    return template