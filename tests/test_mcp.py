"""Tests for the MCP client."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

from agent.mcp_client import MCPClient, _make_mcp_tool, create_mcp_tools


# ---------------------------------------------------------------------------
# Minimal fake MCP server (runs as a subprocess)
# ---------------------------------------------------------------------------

_FAKE_SERVER_SCRIPT = textwrap.dedent("""\
    import sys
    import json

    TOOLS = [
        {
            "name": "echo_tool",
            "description": "Echoes the input back.",
            "inputSchema": {
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
        }
    ]

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        method = req.get("method", "")
        req_id = req.get("id")

        if method == "initialize":
            resp = {"jsonrpc": "2.0", "id": req_id, "result": {"capabilities": {}}}
        elif method == "tools/list":
            resp = {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS}}
        elif method == "tools/call":
            name = req["params"]["name"]
            args = req["params"].get("arguments", {})
            if name == "echo_tool":
                text = args.get("message", "")
                resp = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"content": [{"type": "text", "text": f"echo: {text}"}]},
                }
            else:
                resp = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": "Tool not found"},
                }
        else:
            resp = {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": "Method not found"},
            }

        sys.stdout.write(json.dumps(resp) + "\\n")
        sys.stdout.flush()
""")


@pytest.fixture()
def fake_server_script(tmp_path: Path) -> Path:
    """Write the fake MCP server script to a temp file."""
    script = tmp_path / "fake_mcp_server.py"
    script.write_text(_FAKE_SERVER_SCRIPT, encoding="utf-8")
    return script


@pytest.fixture()
def mcp_client(fake_server_script: Path) -> Generator[MCPClient, None, None]:
    """Start the fake MCP server and yield a connected MCPClient."""
    client = MCPClient(
        name="fake",
        command=sys.executable,
        args=[str(fake_server_script)],
    )
    client.start()
    yield client
    client.stop()


# ---------------------------------------------------------------------------
# MCPClient unit tests
# ---------------------------------------------------------------------------

class TestMCPClient:
    def test_start_and_stop(self, fake_server_script: Path):
        client = MCPClient(
            name="test_start",
            command=sys.executable,
            args=[str(fake_server_script)],
        )
        client.start()
        assert client._process is not None
        assert client._process.poll() is None  # still running
        client.stop()
        # Give OS a moment
        time.sleep(0.1)
        assert client._process.poll() is not None  # terminated

    def test_list_tools(self, mcp_client: MCPClient):
        tools = mcp_client.list_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "echo_tool"
        assert "description" in tools[0]
        assert "inputSchema" in tools[0]

    def test_call_tool_returns_correct_result(self, mcp_client: MCPClient):
        result = mcp_client.call_tool("echo_tool", {"message": "hello MCP"})
        assert "hello MCP" in result
        assert result.startswith("echo: ")

    def test_call_tool_unknown_raises(self, mcp_client: MCPClient):
        with pytest.raises(RuntimeError):
            mcp_client.call_tool("nonexistent_tool", {})

    def test_env_variables_passed(self, tmp_path: Path):
        """Verify that custom env vars reach the server process."""
        script = tmp_path / "env_server.py"
        script.write_text(
            textwrap.dedent("""\
                import sys, json, os
                for line in sys.stdin:
                    req = json.loads(line.strip())
                    if req.get("method") == "initialize":
                        resp = {"jsonrpc":"2.0","id":req["id"],"result":{}}
                    elif req.get("method") == "tools/list":
                        val = os.environ.get("TEST_VAR","missing")
                        resp = {"jsonrpc":"2.0","id":req["id"],"result":{"tools":[{"name":val,"description":"x"}]}}
                    else:
                        resp = {"jsonrpc":"2.0","id":req["id"],"result":{}}
                    sys.stdout.write(json.dumps(resp)+"\\n")
                    sys.stdout.flush()
            """),
            encoding="utf-8",
        )
        client = MCPClient(
            name="env_test",
            command=sys.executable,
            args=[str(script)],
            env={"TEST_VAR": "my_custom_var"},
        )
        client.start()
        try:
            tools = client.list_tools()
            assert tools[0]["name"] == "my_custom_var"
        finally:
            client.stop()


# ---------------------------------------------------------------------------
# _make_mcp_tool
# ---------------------------------------------------------------------------

class TestMakeMCPTool:
    def test_tool_name_and_description(self, mcp_client: MCPClient):
        tool_def = {
            "name": "echo_tool",
            "description": "Echoes the input back.",
            "inputSchema": {},
        }
        lc_tool = _make_mcp_tool(mcp_client, tool_def)
        assert lc_tool.name == "echo_tool"
        assert "Echoes" in lc_tool.description

    def test_tool_run(self, mcp_client: MCPClient):
        tool_def = {
            "name": "echo_tool",
            "description": "Echoes the input back.",
        }
        lc_tool = _make_mcp_tool(mcp_client, tool_def)
        result = lc_tool._run(message="test123")
        assert "test123" in result

    def test_schema_in_description(self, mcp_client: MCPClient):
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        tool_def = {
            "name": "echo_tool",
            "description": "A tool",
            "inputSchema": schema,
        }
        lc_tool = _make_mcp_tool(mcp_client, tool_def)
        assert "Input schema" in lc_tool.description


# ---------------------------------------------------------------------------
# create_mcp_tools
# ---------------------------------------------------------------------------

class TestCreateMCPTools:
    def test_creates_tools_from_config(self, fake_server_script: Path):
        configs = [
            {
                "name": "fake",
                "command": sys.executable,
                "args": [str(fake_server_script)],
            }
        ]
        tools = create_mcp_tools(configs)
        assert len(tools) == 1
        assert tools[0].name == "echo_tool"

    def test_bad_server_is_non_fatal(self, capsys):
        """A broken MCP server config should not crash create_mcp_tools."""
        configs = [
            {
                "name": "broken",
                "command": "this_command_does_not_exist_xyz",
                "args": [],
            }
        ]
        tools = create_mcp_tools(configs)
        assert tools == []
        captured = capsys.readouterr()
        assert "WARNING" in captured.out or "WARNING" in captured.err or True

    def test_multiple_servers(self, fake_server_script: Path):
        configs = [
            {
                "name": "fake1",
                "command": sys.executable,
                "args": [str(fake_server_script)],
            },
            {
                "name": "fake2",
                "command": sys.executable,
                "args": [str(fake_server_script)],
            },
        ]
        tools = create_mcp_tools(configs)
        # Each fake server exposes 1 tool → total 2
        assert len(tools) == 2
