import json
import os
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class ResearchEntry(BaseModel):
    entry_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    intention: str
    mode: str
    insights: Optional[str] = None
    final_prompt: Optional[str] = None
    metrics: Dict[str, Any] = {}
    tags: List[str] = []

class ResearchJournal:
    """
    Manages the persistent history of research sessions to ensure reproducibility.
    """
    def __init__(self, storage_path: str = "results/research_journal.json"):
        self.storage_path = storage_path
        self._ensure_storage()

    def _ensure_storage(self):
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        if not os.path.exists(self.storage_path):
            with open(self.storage_path, "w") as f:
                json.dump([], f)

    def add_entry(self, entry: ResearchEntry):
        try:
            with open(self.storage_path, "r") as f:
                journal = json.load(f)
            journal.append(entry.model_dump())
            with open(self.storage_path, "w") as f:
                json.dump(journal, f, indent=2)
        except Exception as e:
            print(f"Failed to write to journal: {e}")

    def get_entries(self, tag: Optional[str] = None) -> List[ResearchEntry]:
        try:
            with open(self.storage_path, "r") as f:
                journal = json.load(f)
            entries = [ResearchEntry(**e) for e in journal]
            if tag:
                return [e for e in entries if tag in e.tags]
            return entries
        except Exception:
            return []

    def export_as_markdown(self) -> str:
        entries = self.get_entries()
        md = "# CS Research Journal\n\n"
        for e in reversed(entries):
            md += f"## Session: {e.timestamp}\n"
            md += f"**Intention**: {e.intention}\n\n"
            md += f"**Mode**: {e.mode}\n\n"
            if e.insights:
                md += f"### Architectural Insights\n{e.insights}\n\n"
            md += "---\n"
        return md