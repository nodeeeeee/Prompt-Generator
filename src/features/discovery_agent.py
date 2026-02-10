import json
import logging
import asyncio
from typing import List, Dict
from src.llm_integration import LLMClient
from src.features.file_interface import read_project_file

logger = logging.getLogger("DiscoveryAgent")

class DiscoveryAgent:
    """
    Autonomous agent that decides which files to read based on the task intention
    and the project's directory structure.
    """
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    async def discover_and_read_context(
        self, 
        root_path: str, 
        intention: str, 
        tree_str: str,
        max_files: int = 15
    ) -> Dict[str, str]:
        """
        1. Analyzes the tree to pick relevant files.
        2. Reads those files automatically.
        """
        system_instruction = f"""You are a Lead Codebase Explorer. 
Your goal is to identify the most relevant files in a project that an AI needs to understand to fulfill a specific development intention.

GUIDELINES:
1. Analyze the provided directory tree and the user's intention.
2. Select up to {max_files} files that likely contain core logic, interfaces, or configuration related to the task.
3. Prioritize source code (py, rs, go, js, etc.) and configuration (yaml, toml, json).
4. Avoid large data files or logs.

Respond ONLY with a JSON list of relative file paths: ["path/to/file1", "path/to/file2", ...]
"""

        user_content = f"""PROJECT TREE:
{tree_str}

INTENTION: {intention}

Select the most important files."""
        
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_content}
        ]

        try:
            response_text = await self.llm_client.agenerate_completion(messages, temperature=0.2)
            
            # Robust JSON extraction
            import re
            json_match = re.search(r'(\[.*\])', response_text, re.DOTALL)
            if json_match:
                paths = json.loads(json_match.group(1))
            else:
                paths = json.loads(response_text)
            
            discovered_content = {}
            for path in paths:
                logger.info(f"Discovery Agent reading: {path}")
                content = read_project_file(root_path, path)
                if not content.startswith("Error:"):
                    discovered_content[path] = content
                
            return discovered_content

        except Exception as e:
            logger.error(f"Discovery Agent failed: {e}")
            return {}

    async def investigate_and_analyze(
        self,
        root_path: str,
        intention: str,
        tree_str: str,
        max_files: int = 15
    ) -> str:
        """
        Parallelized multi-agent investigation with Global Context:
        1. Explorer Agent: Selects files.
        2. Global Context Agent: Summarizes anchor files (README, main).
        3. Analyst Agents (Parallel): Analyze each file individually using global context.
        4. Synthesizer Agent: Combines analyses into a final report.
        """
        discovered_files = await self.discover_and_read_context(root_path, intention, tree_str, max_files)
        
        if not discovered_files:
            return "No relevant files could be discovered for analysis."

        # --- GLOBAL CONTEXT ACQUISITION ---
        anchor_files = [f for f in discovered_files.keys() if any(x in f.lower() for x in ["readme", "architecture", "design", "main", "requirements", "package.json"])]
        global_context_snippet = ""
        for af in anchor_files:
            global_context_snippet += f"\n--- GLOBAL CONTEXT from {af} ---\n{discovered_files[af][:2000]}\n"

        # Parallelized Analysis Phase (Multi-Agent Simulation)
        async def analyze_file(path: str, content: str) -> str:
            analyst_prompt = f"""You are a Specialized Code Analyst. 
Analyze the following file in the context of the user's intention: {intention}

Use the following global project context for reference:
{global_context_snippet}

FILE TO ANALYZE: {path}
CONTENT:
{content[:8000]}

Provide 2-3 deep technical insights about this specific file's role and constraints, especially how it fits into the global architecture.
"""
            try:
                return await self.llm_client.agenerate_completion(
                    [{"role": "user", "content": analyst_prompt}], 
                    temperature=0.3,
                    timeout=30
                )
            except Exception as e:
                return f"Error analyzing {path}: {e}"

        logger.info(f"Starting parallel analysis of {len(discovered_files)} files with global context...")
        analysis_tasks = [analyze_file(p, c) for p, c in discovered_files.items()]
        individual_analyses = await asyncio.gather(*analysis_tasks)

        # Synthesis Phase
        synthesis_instruction = f"""You are a Lead Systems Architect. 
You are given a collection of technical analyses from various sub-agents who investigated a codebase for the intention: {intention}.
Your task is to synthesize these into a coherent 'Architectural Implementation Directive'.

Include:
1. **Systemic Logic**: High-level patterns and cross-module dependencies.
2. **Implementation Protocol**: Step-by-step technical guidelines.
3. **Safety & Robustness**: Critical constraints and error-handling requirements.

Do not repeat the raw analyses. Provide a unified, high-density architectural strategy.
"""
        
        combined_analysis_text = "\n\n".join(individual_analyses)
        user_content = f"INTENTION: {intention}\n\nSUB-AGENT ANALYSES:\n{combined_analysis_text}"

        try:
            insights = await self.llm_client.agenerate_completion(
                [
                    {"role": "system", "content": synthesis_instruction},
                    {"role": "user", "content": user_content}
                ], 
                temperature=0.4,
                timeout=60
            )
            return insights
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return "Failed to synthesize architectural insights."