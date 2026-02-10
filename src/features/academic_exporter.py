from typing import Dict, Any, Optional

class AcademicExporter:
    """
    Transforms AI-generated technical insights into formal academic formats.
    """
    
    @staticmethod
    def to_latex_methodology(insights: str, intention: str) -> str:
        """
        Converts architectural insights into a LaTeX Methodology section.
        """
        latex = r"\section{Methodology}" + "\n"
        latex += f"% Generated based on intention: {intention}\n\n"
        
        # Simple transformation logic (could be enhanced with LLM)
        lines = insights.split("\n")
        current_section = ""
        
        for line in lines:
            line = line.strip()
            if not line: continue
            
            # Robust header detection for any level (# to ###) or bold text
            if line.startswith("#") or line.startswith("**"):
                # Prune markdown symbols
                section_name = line.lstrip("#").replace("*", "").strip()
                if not section_name: continue
                latex += r"\subsection{" + section_name + "}\n"
            elif line.startswith("-") or line.startswith("*"):
                if current_section != "list":
                    latex += r"\begin{itemize}" + "\n"
                    current_section = "list"
                latex += r"  \item " + line[1:].strip() + "\n"
            else:
                if current_section == "list":
                    latex += r"\end{itemize}" + "\n"
                    current_section = ""
                latex += line + "\n"
                
        if current_section == "list":
            latex += r"\end{itemize}" + "\n"
            
        return latex

    @staticmethod
    def to_latex_experiment(experiment_design: str) -> str:
        """
        Converts experiment ideas into a LaTeX Evaluation section.
        """
        latex = r"\section{Experimental Design}" + "\n"
        latex += experiment_design.replace("_", r"\_").replace("#", r"\#")
        return latex

    @staticmethod
    def get_bibtex() -> str:
        """
        Returns BibTeX entries for the tools used.
        """
        return """@software{gemini_cli_prompt_gen,
  author = {Gemini CLI},
  title = {High-Performance Research Prompt Generator},
  year = {2026},
  url = {https://github.com/google/gemini-cli}
}

@article{litellm,
  title={LiteLLM: A Unified Interface for LLM APIs},
  author={LiteLLM Authors},
  year={2024},
  journal={GitHub Repository}
}"""