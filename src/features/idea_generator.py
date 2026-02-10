import json
from typing import Dict, Any, Tuple, List
from src.llm_integration import LLMClient
from src.prompt_builder import PromptBuilder
from src.features.experiment_planner import generate_experiment_prompt_snippet

async def generate_idea_questions(
    client: LLMClient,
    project_context: str,
    idea: str,
    choice: str
) -> List[str]:
    """
    Generates targeted questions to better grasp the technical details of an idea.
    """
    if choice == "conduct experiment":
        system_instruction = """You are a Principal Research Scientist. 
Your goal is to design a rigorous scientific experiment for a software system.
Ask 3-4 pedantic questions about:
1. The formal HYPOTHESIS.
2. The INDEPENDENT and DEPENDENT variables.
3. The BASELINE for comparison.
4. The METRIC collection strategy (instrumentation).

Respond ONLY with a JSON list of strings."""
    else:
        system_instruction = """You are a Senior Software Architect. 
Your goal is to plan a sophisticated new feature for a codebase.
Ask 3-4 strategic questions about:
1. The ARCHITECTURAL IMPACT (changes to existing state/data flow).
2. The USER INTERFACE or API surface.
3. ERROR HANDLING and edge cases specific to this feature.
4. SCALABILITY and technical debt considerations.

Respond ONLY with a JSON list of strings."""

    user_content = f"CONTEXT:\n{project_context[:5000]}\n\n{choice.upper()} IDEA: {idea}"
    
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_content}
    ]

    response_text = await client.agenerate_completion(messages, temperature=0.7)
    
    try:
        import re
        json_match = re.search(r'(\[.*\])', response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
        return json.loads(response_text)
    except Exception:
        return ["What is the primary technical goal?", "How should this integrate with existing modules?"]

async def generate_raw_idea(
    client: LLMClient,
    project_context: str,
    choice: str
) -> str:
    """Suggests a single high-impact idea."""
    if choice == "conduct experiment":
        system_instruction = """You are a Lead Researcher. 
Suggest ONE scientific experiment (e.g. ablation, benchmark, robustness test) to run on this codebase. 
Focus on proving/disproving a technical hypothesis.
Return ONLY the description."""
    else:
        system_instruction = """You are a Lead Product Architect. 
Suggest ONE sophisticated new feature or capability to add to this codebase. 
Focus on enhancing functionality or solving a core limitation.
Return ONLY the description."""

    user_content = f"CONTEXT:\n{project_context[:5000]}"
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_content}
    ]
    return await client.agenerate_completion(messages, temperature=0.9)

async def generate_idea_and_prompt(
    client: LLMClient,
    builder: PromptBuilder,
    project_context: str,
    choice: str,
    idea: str,
    qa_history: List[Dict[str, str]] = None,
    root_path: str = None,
    auto_discover: bool = False
) -> Tuple[str, List[str]]:
    """
    Generates the final prompt based on the idea, project context, and clarifications.
    Returns (final_prompt, discovered_file_paths).
    """
    qa_context = ""
    if qa_history:
        qa_context = "\n".join([f"Q: {item['q']}\nA: {item['a']}" for item in qa_history])
    
    # We use the idea as the core intention
    intention = idea

    if choice == "conduct experiment":
        # For experiments, we still want to guess the type/params or extract from QA
        # Here we'll just use a generic but high-quality experiment snippet
        exp_snip = f"\n### Experimental Setup\n{qa_context}\n"
        
        return await builder.build_prompt(
            intention=intention,
            answers=[],
            questions=[],
            mode="chain-of-thought",
            project_context=project_context,
            experiment_context=exp_snip,
            root_path=root_path,
            auto_discover=auto_discover
        )
    else:
        # New features use iterative mode
        return await builder.build_prompt(
            intention=intention,
            answers=[],
            questions=[],
            mode="iterative",
            project_context=project_context,
            experiment_context=f"Additional Requirements:\n{qa_context}",
            root_path=root_path,
            auto_discover=auto_discover
        )