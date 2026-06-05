"""Tests for the CodingAgent class."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.agent import CodingAgent, _truncate
from agent.tools import create_tools


# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------

class TestTruncate:
    def test_short_string_unchanged(self):
        assert _truncate("hello", 10) == "hello"

    def test_exact_limit_unchanged(self):
        assert _truncate("a" * 10, 10) == "a" * 10

    def test_long_string_truncated(self):
        result = _truncate("a" * 300, 200)
        assert len(result) > 200  # includes the "… [N more chars]" suffix
        assert result.startswith("a" * 200)
        assert "more chars" in result

    def test_limit_200_chars(self):
        """Tool-call inputs must be limited to 200 chars."""
        s = "x" * 500
        result = _truncate(s, 200)
        assert result[:200] == "x" * 200

    def test_limit_500_chars(self):
        """Tool results must be limited to 500 chars."""
        s = "y" * 1000
        result = _truncate(s, 500)
        assert result[:500] == "y" * 500


# ---------------------------------------------------------------------------
# CodingAgent initialisation (no real LLM needed)
# ---------------------------------------------------------------------------

class TestCodingAgentInit:
    def test_init_with_defaults(self, tmp_path: Path):
        """CodingAgent should initialise without error using mocked Ollama."""
        with patch("agent.agent.ChatOllama") as mock_llm_cls, \
             patch("agent.agent.create_react_agent") as mock_graph:
            mock_llm = MagicMock()
            mock_llm_cls.return_value = mock_llm
            mock_graph.return_value = MagicMock()

            agent = CodingAgent(config_path=None, cwd=str(tmp_path))

        assert agent.cwd == str(tmp_path)
        assert len(agent.tools) == 15
        mock_llm_cls.assert_called_once()

    def test_init_uses_config_model_name(self, tmp_path: Path):
        config_file = tmp_path / "custom.yaml"
        config_file.write_text(
            "model:\n  name: qwen2.5\n  base_url: http://localhost:11434\n",
            encoding="utf-8",
        )
        with patch("agent.agent.ChatOllama") as mock_llm_cls, \
             patch("agent.agent.create_react_agent") as mock_graph:
            mock_llm_cls.return_value = MagicMock()
            mock_graph.return_value = MagicMock()

            CodingAgent(config_path=str(config_file), cwd=str(tmp_path))

        _, kwargs = mock_llm_cls.call_args
        assert kwargs.get("model") == "qwen2.5"

    def test_init_uses_config_base_url(self, tmp_path: Path):
        config_file = tmp_path / "url.yaml"
        config_file.write_text(
            "model:\n  name: llama3.2\n  base_url: http://my-ollama-host:11434\n",
            encoding="utf-8",
        )
        with patch("agent.agent.ChatOllama") as mock_llm_cls, \
             patch("agent.agent.create_react_agent") as mock_graph:
            mock_llm_cls.return_value = MagicMock()
            mock_graph.return_value = MagicMock()

            CodingAgent(config_path=str(config_file), cwd=str(tmp_path))

        _, kwargs = mock_llm_cls.call_args
        assert kwargs.get("base_url") == "http://my-ollama-host:11434"

    def test_cwd_defaults_to_getcwd(self, tmp_path: Path):
        with patch("agent.agent.ChatOllama") as mock_llm_cls, \
             patch("agent.agent.create_react_agent") as mock_graph:
            mock_llm_cls.return_value = MagicMock()
            mock_graph.return_value = MagicMock()
            import os
            expected = os.getcwd()
            agent = CodingAgent(config_path=None)

        assert agent.cwd == expected

    def test_mcp_servers_in_config_loaded(self, tmp_path: Path):
        config_file = tmp_path / "mcp.yaml"
        config_file.write_text(
            "mcp_servers:\n  - name: test\n    command: echo\n    args: ['hi']\n",
            encoding="utf-8",
        )
        with patch("agent.agent.ChatOllama") as mock_llm_cls, \
             patch("agent.agent.create_react_agent") as mock_graph, \
             patch("agent.agent.create_mcp_tools") as mock_mcp:
            mock_llm_cls.return_value = MagicMock()
            mock_graph.return_value = MagicMock()
            mock_mcp.return_value = []

            CodingAgent(config_path=str(config_file), cwd=str(tmp_path))

        mock_mcp.assert_called_once()
        called_with = mock_mcp.call_args[0][0]
        assert any(s["name"] == "test" for s in called_with)


# ---------------------------------------------------------------------------
# CodingAgent.invoke — streaming simulation
# ---------------------------------------------------------------------------

def _make_ai_message(content="", tool_calls=None):
    """Build a mock AIMessage."""
    msg = MagicMock()
    msg.__class__.__name__ = "AIMessage"
    from langchain_core.messages import AIMessage
    msg.__class__ = AIMessage
    msg.content = content
    msg.tool_calls = tool_calls or []
    return msg


def _make_tool_message(content):
    """Build a mock ToolMessage."""
    from langchain_core.messages import ToolMessage
    msg = MagicMock()
    msg.__class__ = ToolMessage
    msg.content = content
    return msg


class TestCodingAgentInvoke:
    def _make_agent(self, tmp_path, stream_chunks):
        """Helper: create a CodingAgent with a mocked _agent.stream."""
        with patch("agent.agent.ChatOllama") as mock_llm_cls, \
             patch("agent.agent.create_react_agent") as mock_graph:
            mock_llm_cls.return_value = MagicMock()
            mock_compiled = MagicMock()
            mock_compiled.stream.return_value = iter(stream_chunks)
            mock_graph.return_value = mock_compiled
            agent = CodingAgent(config_path=None, cwd=str(tmp_path))

        return agent

    def test_returns_final_content(self, tmp_path: Path):
        from langchain_core.messages import AIMessage

        final_msg = AIMessage(content="Here is your answer.")
        chunk = {"agent": {"messages": [final_msg]}}

        agent = self._make_agent(tmp_path, [chunk])
        # Replace the _agent.stream to return our chunk
        agent._agent.stream = MagicMock(return_value=iter([chunk]))

        result = agent.invoke("What is 2+2?")
        assert "Here is your answer." in result

    def test_prints_tool_call(self, tmp_path: Path, capsys):
        from langchain_core.messages import AIMessage, ToolMessage

        tool_call_msg = AIMessage(content="")
        tool_call_msg.tool_calls = [{"name": "list_dir", "args": {"path": "."}, "id": "tc1"}]

        tool_result_msg = ToolMessage(content="[FILE] foo.txt", tool_call_id="tc1")

        final_msg = AIMessage(content="Done.")

        chunks = [
            {"agent": {"messages": [tool_call_msg]}},
            {"tools": {"messages": [tool_result_msg]}},
            {"agent": {"messages": [final_msg]}},
        ]

        agent = self._make_agent(tmp_path, chunks)
        agent._agent.stream = MagicMock(return_value=iter(chunks))

        agent.invoke("List files")
        captured = capsys.readouterr()
        assert "list_dir" in captured.out
        assert "foo.txt" in captured.out

    def test_tool_call_truncated_at_200(self, tmp_path: Path, capsys):
        from langchain_core.messages import AIMessage

        long_args = {"path": "x" * 500}
        tool_call_msg = AIMessage(content="")
        tool_call_msg.tool_calls = [{"name": "read_file", "args": long_args, "id": "tc2"}]

        final_msg = AIMessage(content="Done.")
        chunks = [
            {"agent": {"messages": [tool_call_msg]}},
            {"agent": {"messages": [final_msg]}},
        ]

        agent = self._make_agent(tmp_path, chunks)
        agent._agent.stream = MagicMock(return_value=iter(chunks))

        agent.invoke("read something")
        captured = capsys.readouterr()

        # Find the line with the args
        args_lines = [l for l in captured.out.splitlines() if "read_file" in l or "x" * 10 in l]
        # Combined line content should not exceed 200 chars of the actual arg
        for line in args_lines:
            # The raw arg part should be truncated
            assert len(line) < 600  # generous bound; main guard is below

        # The "more chars" indicator must appear because we have 500 'x's
        assert "more chars" in captured.out

    def test_tool_result_truncated_at_500(self, tmp_path: Path, capsys):
        from langchain_core.messages import AIMessage, ToolMessage

        long_result = "r" * 1000
        tool_result_msg = ToolMessage(content=long_result, tool_call_id="tc3")
        final_msg = AIMessage(content="Done.")

        chunks = [
            {"tools": {"messages": [tool_result_msg]}},
            {"agent": {"messages": [final_msg]}},
        ]

        agent = self._make_agent(tmp_path, chunks)
        agent._agent.stream = MagicMock(return_value=iter(chunks))

        agent.invoke("get results")
        captured = capsys.readouterr()
        assert "more chars" in captured.out

    def test_fallback_when_no_content(self, tmp_path: Path):
        from langchain_core.messages import AIMessage

        empty_msg = AIMessage(content="")
        chunks = [{"agent": {"messages": [empty_msg]}}]

        agent = self._make_agent(tmp_path, chunks)
        agent._agent.stream = MagicMock(return_value=iter(chunks))

        result = agent.invoke("hello")
        assert result == "No response."
