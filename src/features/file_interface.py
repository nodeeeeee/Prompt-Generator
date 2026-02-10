import os
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger("FileInterface")

class FileAccessError(Exception):
    """Base exception for file access errors."""
    pass

def read_project_file(root_path: str, relative_path: str, max_chars: int = 10000) -> str:
    """
    Safely reads a specific file from the project directory.
    
    Robustness features:
    1. Path Traversal Protection: Ensures the file is within root_path.
    2. Size Boundaries: Limits the number of characters read.
    3. Binary Handling: Detects and skips binary files to prevent corruption.
    4. Explicit Error Handling: Informative messages for missing files or permission issues.
    """
    try:
        # 1. Resolve and normalize paths
        root_abs = os.path.abspath(root_path)
        file_abs = os.path.abspath(os.path.join(root_abs, relative_path))
        
        # 2. Security Check: Path Traversal
        if not file_abs.startswith(root_abs):
            raise FileAccessError(f"Security violation: Path '{relative_path}' is outside the project root.")

        # 3. Existence Check
        if not os.path.exists(file_abs):
            raise FileAccessError(f"File not found: '{relative_path}'")
        
        if not os.path.isfile(file_abs):
            raise FileAccessError(f"'{relative_path}' is a directory, not a file.")

        # 4. Binary Check (Small sample check)
        try:
            with open(file_abs, 'rb') as f:
                chunk = f.read(1024)
                if b'\x00' in chunk:
                    return f"[INFO] Binary file detected: '{relative_path}'. Content skipped."
        except Exception:
            pass

        # 5. Read with size limit
        with open(file_abs, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read(max_chars + 1)
            
            if len(content) > max_chars:
                return content[:max_chars] + f"\n\n...[Truncated: File exceeds {max_chars} chars]..."
            
            return content

    except FileAccessError as e:
        logger.warning(f"File access denied: {e}")
        return f"Error: {e}"
    except Exception as e:
        logger.error(f"Unexpected error reading '{relative_path}': {e}")
        return f"Error: Internal failure while reading file."

def get_file_metadata(root_path: str, relative_path: str) -> Dict[str, Any]:
    """Returns basic metadata about a file."""
    try:
        root_abs = os.path.abspath(root_path)
        file_abs = os.path.abspath(os.path.join(root_abs, relative_path))
        
        if not file_abs.startswith(root_abs) or not os.path.exists(file_abs):
            return {"error": "Invalid file path"}
            
        stats = os.stat(file_abs)
        return {
            "name": os.path.basename(file_abs),
            "size_bytes": stats.st_size,
            "modified": stats.st_mtime,
            "extension": os.path.splitext(file_abs)[1]
        }
    except Exception as e:
        return {"error": str(e)}