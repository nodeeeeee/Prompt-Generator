import os
from typing import List, Dict

IGNORE_DIRS = {
    "__pycache__", ".git", ".idea", ".vscode", "node_modules", 
    "venv", "env", ".env", "dist", "build", "coverage", ".pytest_cache"
}

IGNORE_FILES = {
    ".DS_Store", "*.pyc", "*.pyo", "package-lock.json", "yarn.lock"
}

def scan_directory(root_path: str, max_depth: int = 3) -> str:
    """
    Generates a tree view of the directory structure.
    """
    if not os.path.exists(root_path):
        return f"Error: Path '{root_path}' does not exist."

    tree_str = f"Project Root: {os.path.basename(os.path.abspath(root_path))}\\n"
    
    def _walk(path: str, prefix: str = "", current_depth: int = 0):
        nonlocal tree_str
        if current_depth > max_depth:
            return

        try:
            entries = sorted(os.listdir(path))
        except PermissionError:
            return

        # Filter ignored
        entries = [e for e in entries if e not in IGNORE_DIRS and not e.startswith('.')]
        
        for i, entry in enumerate(entries):
            full_path = os.path.join(path, entry)
            is_last = (i == len(entries) - 1)
            connector = "└── " if is_last else "├── "
            
            tree_str += f"{prefix}{connector}{entry}\\n"
            
            if os.path.isdir(full_path):
                extension = "    " if is_last else "│   "
                _walk(full_path, prefix + extension, current_depth + 1)

    _walk(root_path)
    return tree_str

def read_key_files(root_path: str) -> Dict[str, str]:
    """
    Reads content of key files like README.md, requirements.txt, pyproject.toml
    """
    key_files = ["README.md", "requirements.txt", "pyproject.toml", "setup.py", "Dockerfile"]
    found_content = {}
    
    for filename in key_files:
        path = os.path.join(root_path, filename)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                    # Truncate if too long (simple heuristic for token limits)
                    if len(content) > 2000:
                        content = content[:2000] + "\\n...[Truncated]..."
                    found_content[filename] = content
            except Exception as e:
                found_content[filename] = f"Error reading file: {e}"
                
    return found_content
