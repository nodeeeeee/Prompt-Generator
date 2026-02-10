import logging
import asyncio
from typing import List, Optional, Tuple
from src.llm_integration import LLMClient
from src.features.persona import suggest_persona
from src.features.complexity import estimate_complexity
from src.features.prompt_templates import (
    get_detailed_cot_template, 
    get_research_experiment_template, 
    get_iterative_long_form_template,
    get_one_shot_template
)

from src.features.prompt_optimizer import PromptOptimizer
from src.features.discovery_agent import DiscoveryAgent
from src.features.context_manager import scan_directory

logger = logging.getLogger(__name__)

class PromptBuilder:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.optimizer = PromptOptimizer(llm_client)
        self.discovery_agent = DiscoveryAgent(llm_client)

    async def build_prompt(
        self, 
        intention: str, 
        answers: List[str], 
        questions: List[str],
        mode: str = "one-shot",
        project_context: Optional[str] = None,
        experiment_context: Optional[str] = None,
        root_path: Optional[str] = None,
        auto_discover: bool = False
    ) -> Tuple[str, List[str]]:
        """
        Builds the final prompt using specialized templates and LLM optimization.
        Returns (final_prompt, discovered_file_paths).
        """
        
        # High performance: parallelize persona suggestion and complexity estimation
        try:
            # Added 45s safety timeout for the parallel tasks
            persona, complexity = await asyncio.wait_for(
                asyncio.gather(
                    suggest_persona(intention, self.llm_client),
                    estimate_complexity(intention, answers, self.llm_client),
                    return_exceptions=True
                ),
                timeout=45.0
            )
            if isinstance(persona, Exception) or not persona: persona = "Expert Software Engineer"
            if isinstance(complexity, Exception) or not complexity: complexity = "Standard"
        except (asyncio.TimeoutError, Exception):
            persona = "Expert Software Engineer"
            complexity = "Standard"
        
        # Construct Context from Q&A
        qa_pairs = []
        for q, a in zip(questions, answers):
            qa_pairs.append(f"Q: {q}\nA: {a}")
        
        qa_context = "\n".join(qa_pairs) if qa_pairs else "No additional clarifications provided."

        discovered_paths = []
        # Autonomous Discovery Phase
        if auto_discover and root_path:
            # The tree/context provided is for the prompt agent's knowledge
            # We use it to generate high-density insights
            knowledge_map = project_context or scan_directory(root_path)
            insights = await self.discovery_agent.investigate_and_analyze(root_path, intention, knowledge_map)
            
            # We explicitly overwrite project_context to ONLY include synthesized insights.
            # We do NOT include the raw project tree or file contents in the final prompt
            # because the consumer coding agent will read the codebase itself.
            project_context = f"### ARCHITECTURAL INSIGHTS & INTEGRATION GUIDELINES\n{insights}"
            discovered_paths = ["Autonomous Analysis performed"]
        elif project_context:
            # If not auto-discovering but context was provided, we still prune 
            # obvious project structure/raw files to keep the prompt clean.
            import re
            # Prune tree-like structures and raw file blocks
            project_context = re.sub(r'([│├└─]{2,}.*\n?)+', '', project_context) # Prune tree
            project_context = re.sub(r'--- FILE: .* ---[\s\S]*?--- END FILE ---', '', project_context) # Prune raw blocks
            project_context = project_context.strip()

        # Generate Mechanical Base Prompt
        if project_context and experiment_context:
            raw_prompt = get_research_experiment_template(
                persona=persona,
                intention=intention,
                qa_context=qa_context,
                project_context=project_context,
                experiment_context=experiment_context
            )
        elif mode == "iterative":
            raw_prompt = get_iterative_long_form_template(
                persona=persona,
                intention=intention,
                qa_context=qa_context,
                complexity=complexity,
                project_context=project_context,
                experiment_context=experiment_context
            )
        elif mode == "chain-of-thought":
            raw_prompt = get_detailed_cot_template(
                persona=persona,
                intention=intention,
                qa_context=qa_context,
                complexity=complexity,
                project_context=project_context,
                experiment_context=experiment_context
            )
        else:
            raw_prompt = get_one_shot_template(
                persona=persona,
                intention=intention,
                qa_context=qa_context,
                project_context=project_context,
                experiment_context=experiment_context
            )

        # Use AI to optimize and expand the prompt
        try:
            # Add evaluation strategy requirement to the optimizer context
            eval_requirement = "\n4) Include a formal EVALUATION & BENCHMARKING section with specific metrics (e.g., latency, throughput, energy efficiency, or formal proof goals)."
            
            optimized_prompt = await self.optimizer.optimize_prompt(raw_prompt, mode, intention + eval_requirement)
            if not optimized_prompt:
                optimized_prompt = raw_prompt
        except Exception:
            optimized_prompt = raw_prompt
            
        return optimized_prompt, discovered_paths
