"""Shared pytest fixtures."""

from __future__ import annotations

import os
import subprocess
import textwrap
from pathlib import Path

import pytest


@pytest.fixture()
def tmp_cwd(tmp_path: Path) -> str:
    """Return a fresh temporary directory as a string path."""
    return str(tmp_path)


@pytest.fixture()
def sample_file(tmp_path: Path) -> Path:
    """Write a simple Python-like file and return its path."""
    p = tmp_path / "sample.py"
    p.write_text(
        textwrap.dedent("""\
            def greet(name):
                return f"Hello, {name}!"

            def add(a, b):
                return a + b

            if __name__ == "__main__":
                print(greet("World"))
        """),
        encoding="utf-8",
    )
    return p


@pytest.fixture()
def git_repo(tmp_path: Path) -> Path:
    """Create a minimal git repository for git-related tool tests."""
    subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
    )
    readme = tmp_path / "README.md"
    readme.write_text("# Test repo\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial commit"],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
    )
    return tmp_path
