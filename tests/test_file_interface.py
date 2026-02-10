import pytest
import os
import tempfile
from src.features.file_interface import read_project_file

def test_read_project_file_success():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test file
        file_path = os.path.join(tmpdir, "test.py")
        content = "print('hello')"
        with open(file_path, "w") as f:
            f.write(content)
            
        # Read it
        res = read_project_file(tmpdir, "test.py")
        assert res == content

def test_read_project_file_traversal_protection():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Attempt to read something outside
        res = read_project_file(tmpdir, "../../../etc/passwd")
        assert "Security violation" in res

def test_read_project_file_not_found():
    with tempfile.TemporaryDirectory() as tmpdir:
        res = read_project_file(tmpdir, "missing.py")
        assert "File not found" in res

def test_read_project_file_truncation():
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "large.txt")
        content = "A" * 100
        with open(file_path, "w") as f:
            f.write(content)
            
        # Read with small limit
        res = read_project_file(tmpdir, "large.txt", max_chars=10)
        assert len(res) > 10
        assert "...[Truncated" in res
        assert res.startswith("A" * 10)

def test_read_project_file_binary():
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "binary.bin")
        with open(file_path, "wb") as f:
            f.write(b"\x00\x01\x02\x03")
            
        res = read_project_file(tmpdir, "binary.bin")
        assert "Binary file detected" in res
