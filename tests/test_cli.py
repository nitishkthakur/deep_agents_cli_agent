"""Tests for the CLI entry point."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestCLIArgParsing:
    def test_help_exits_zero(self):
        result = subprocess.run(
            [sys.executable, "cli.py", "--help"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        assert result.returncode == 0
        assert "usage" in result.stdout.lower() or "Usage" in result.stdout

    def test_missing_dir_exits_nonzero(self, tmp_path: Path):
        result = subprocess.run(
            [sys.executable, "cli.py", "--cwd", "/no/such/dir/xyz", "hello"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        assert result.returncode != 0

    def test_single_shot_mode(self, tmp_path: Path):
        """With a patched CodingAgent, verify single-shot invocation works."""
        env = {**os.environ, "PYTHONPATH": str(Path(__file__).parent.parent)}
        script = f"""
import sys
sys.path.insert(0, "{Path(__file__).parent.parent}")
from unittest.mock import MagicMock, patch

with patch("agent.agent.ChatOllama") as mock_llm, \\
     patch("agent.agent.create_react_agent") as mock_graph:
    mock_lm = MagicMock()
    mock_lm.return_value = MagicMock()
    mock_llm.return_value = mock_lm
    mock_compiled = MagicMock()
    mock_compiled.stream = MagicMock(return_value=iter([]))
    mock_graph.return_value = mock_compiled

    import cli
    import argparse

    # Patch parse_args to return known values
    with patch.object(cli, "_parse_args", return_value=argparse.Namespace(
        config=None,
        cwd="{tmp_path}",
        mcp_servers=[],
        message="hello test",
    )):
        with patch("builtins.print"):
            try:
                cli.main()
            except SystemExit:
                pass
print("OK")
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
        )
        assert "OK" in result.stdout or result.returncode == 0


class TestCLIGetCwd:
    def test_provided_cwd_accepted(self, tmp_path: Path):
        from cli import _get_cwd
        result = _get_cwd(str(tmp_path))
        assert result == str(tmp_path)

    def test_nonexistent_dir_exits(self):
        with pytest.raises(SystemExit):
            from cli import _get_cwd
            _get_cwd("/absolutely/does/not/exist/xyz")

    def test_prompted_cwd_uses_input(self, tmp_path: Path):
        from cli import _get_cwd
        with patch("builtins.input", return_value=str(tmp_path)):
            result = _get_cwd(None)
        assert result == str(tmp_path)

    def test_empty_input_uses_cwd(self):
        import os
        from cli import _get_cwd
        with patch("builtins.input", return_value=""):
            result = _get_cwd(None)
        assert result == os.getcwd()


class TestCLIBuildMCPList:
    def test_valid_json(self):
        from cli import _build_mcp_list
        raw = ['{"name":"s","command":"c","args":[]}']
        result = _build_mcp_list(raw)
        assert result[0]["name"] == "s"

    def test_invalid_json_skipped(self, capsys):
        from cli import _build_mcp_list
        result = _build_mcp_list(["not json!"])
        assert result == []
