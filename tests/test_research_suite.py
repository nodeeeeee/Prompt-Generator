import pytest
import os
import json
from src.features.research_journal import ResearchJournal, ResearchEntry
from src.features.academic_exporter import AcademicExporter

def test_research_journal_functionality():
    temp_journal_path = "results/test_journal.json"
    if os.path.exists(temp_journal_path):
        os.remove(temp_journal_path)
        
    journal = ResearchJournal(storage_path=temp_journal_path)
    entry = ResearchEntry(
        intention="Test research intention",
        mode="iterative",
        insights="""### Insight
- Detail 1""",
        final_prompt="Final Prompt Text"
    )
    
    journal.add_entry(entry)
    entries = journal.get_entries()
    
    assert len(entries) == 1
    assert entries[0].intention == "Test research intention"
    assert "Detail 1" in journal.export_as_markdown()
    
    os.remove(temp_journal_path)

def test_academic_exporter_latex():
    insights = """### Core Logic
- Uses Paxos for consensus.
- Implements RCU."""
    latex = AcademicExporter.to_latex_methodology(insights, "Distributed KV Store")
    
    assert r"\section{Methodology}" in latex
    assert r"\subsection{Core Logic}" in latex
    assert r"\begin{itemize}" in latex
    assert "Paxos" in latex

def test_latex_experiment_export():

    design = """### Hypothesis

Performance scales linearly with nodes."""

    latex = AcademicExporter.to_latex_experiment(design)

    assert r"\section{Experimental Design}" in latex

    assert "Hypothesis" in latex



def test_bibtex_export():

    bib = AcademicExporter.get_bibtex()

    assert "@software{gemini_cli_prompt_gen" in bib

    assert "@article{litellm" in bib
