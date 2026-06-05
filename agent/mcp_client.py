"""MCP (Model Context Protocol) client — connect to external tool servers.

Each MCP server is started as a child process and communicates over stdio
using JSON-RPC 2.0.  The client exposes the server's tools as LangChain
``BaseTool`` instances.

Reference: https://spec.modelcontextprotocol.io/
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import Field


class MCPClient:
    """Thin stdio JSON-RPC client for a single MCP server process."""

    def __init__(self, name: str, command: str, args: list[str], env: dict | None = None):
        self.name = name
        self.command = command
        self.args = args
        self.env: dict[str, str] = {**os.environ, **(env or {})}
        self._process: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._req_id = 0

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Start the MCP server and perform the initialisation handshake."""
        self._process = subprocess.Popen(
            [self.command, *self.args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=self.env,
        )
        self._send("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "clientInfo": {"name": "coding-agent", "version": "1.0.0"},
        })

    def stop(self) -> None:
        """Terminate the MCP server process."""
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()

    # ------------------------------------------------------------------ #
    # JSON-RPC helpers
    # ------------------------------------------------------------------ #

    def _send(self, method: str, params: dict) -> dict:
        """Send a JSON-RPC request and return the result dict."""
        with self._lock:
            self._req_id += 1
            request = {
                "jsonrpc": "2.0",
                "id": self._req_id,
                "method": method,
                "params": params,
            }
            assert self._process and self._process.stdin
            self._process.stdin.write(json.dumps(request) + "\n")
            self._process.stdin.flush()

            line = self._process.stdout.readline()  # type: ignore[union-attr]
            if not line:
                raise RuntimeError(f"MCP server '{self.name}' closed stdout unexpectedly.")
            response = json.loads(line)
            if "error" in response:
                raise RuntimeError(f"MCP error from '{self.name}': {response['error']}")
            return response.get("result", {})

    # ------------------------------------------------------------------ #
    # Tool discovery & invocation
    # ------------------------------------------------------------------ #

    def list_tools(self) -> list[dict]:
        """Return the list of tool descriptors from the MCP server."""
        result = self._send("tools/list", {})
        return result.get("tools", [])

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Invoke a named tool and return its text output."""
        result = self._send("tools/call", {"name": tool_name, "arguments": arguments})
        content: list[dict] = result.get("content", [])
        texts = [c["text"] for c in content if c.get("type") == "text"]
        return "\n".join(texts) if texts else json.dumps(result)


# ---------------------------------------------------------------------------
# LangChain tool wrapper
# ---------------------------------------------------------------------------

def _make_mcp_tool(client: MCPClient, tool_def: dict) -> BaseTool:
    """Wrap a single MCP tool definition as a LangChain ``BaseTool``."""

    tool_name: str = tool_def["name"]
    tool_description: str = tool_def.get("description", f"MCP tool: {tool_name}")

    # Build a JSON-schema-aware description so the LLM knows what args to pass
    schema = tool_def.get("inputSchema", {})
    if schema:
        tool_description += f"\nInput schema: {json.dumps(schema)}"

    class _MCPTool(BaseTool):
        name: str = Field(default=tool_name)
        description: str = Field(default=tool_description)

        def _run(self, **kwargs: Any) -> str:
            return client.call_tool(tool_name, kwargs)

        async def _arun(self, **kwargs: Any) -> str:
            return self._run(**kwargs)

    return _MCPTool()


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def create_mcp_tools(server_configs: list[dict]) -> list[BaseTool]:
    """Start every MCP server listed in *server_configs* and collect their tools.

    Each entry in *server_configs* must have:
      - ``name``    — human-friendly label
      - ``command`` — executable to run
      - ``args``    — list of command-line arguments
      - ``env``     — (optional) extra environment variables

    Returns a flat list of :class:`BaseTool` instances ready to hand to
    ``create_react_agent``.
    """
    all_tools: list[BaseTool] = []
    for cfg in server_configs:
        try:
            client = MCPClient(
                name=cfg["name"],
                command=cfg["command"],
                args=cfg.get("args", []),
                env=cfg.get("env"),
            )
            client.start()
            for tool_def in client.list_tools():
                all_tools.append(_make_mcp_tool(client, tool_def))
        except Exception as exc:
            # Non-fatal — log and continue so the agent still starts
            print(f"[WARNING] Could not load MCP server '{cfg.get('name')}': {exc}")

    return all_tools
