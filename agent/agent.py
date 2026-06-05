"""CodingAgent — the main agent class.

Usage
-----
::

    from agent import CodingAgent

    agent = CodingAgent(config_path="config.yaml", cwd="/my/project")
    response = agent.invoke("What files are in this directory?")
    print(response)
"""

from __future__ import annotations

import os
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

from .config import load_config
from .mcp_client import create_mcp_tools
from .tools import create_tools

# ANSI colour codes (degrade gracefully when the terminal doesn't support them)
_CYAN = "\033[96m"
_GREEN = "\033[92m"
_RESET = "\033[0m"


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f" … [{len(text) - limit} more chars]"


class CodingAgent:
    """General-purpose coding agent with 15 GitHub Copilot-inspired tools.

    Parameters
    ----------
    config_path:
        Path to the YAML configuration file (see ``config.yaml`` for the
        full schema).  Falls back to built-in defaults when omitted.
    cwd:
        Working directory exposed to the agent's file / shell tools.
        Defaults to the current process working directory.
    mcp_servers:
        Additional MCP server configurations to load **in addition to** any
        servers declared inside *config_path*.
    """

    def __init__(
        self,
        config_path: str | None = "config.yaml",
        cwd: str | None = None,
        mcp_servers: list[dict] | None = None,
    ) -> None:
        self.config = load_config(config_path)
        self.cwd = os.path.abspath(cwd or os.getcwd())

        # ------------------------------------------------------------------
        # Build tool list
        # ------------------------------------------------------------------
        self.tools = create_tools(self.cwd)

        combined_mcp = list(mcp_servers or []) + self.config.get("mcp_servers", [])
        if combined_mcp:
            self.tools.extend(create_mcp_tools(combined_mcp))

        # ------------------------------------------------------------------
        # LLM
        # ------------------------------------------------------------------
        model_cfg = self.config.get("model", {})
        self.llm = ChatOllama(
            model=model_cfg.get("name", "llama3.2"),
            base_url=model_cfg.get("base_url", "http://localhost:11434"),
            temperature=model_cfg.get("temperature", 0),
        )

        # ------------------------------------------------------------------
        # Agent graph (LangGraph ReAct)
        # ------------------------------------------------------------------
        agent_cfg = self.config.get("agent", {})
        system_prompt: str = agent_cfg.get(
            "system_prompt",
            "You are a helpful general-purpose coding assistant.",
        )
        self._max_iterations: int = agent_cfg.get("max_iterations", 20)

        self._agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=system_prompt,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def invoke(self, message: str) -> str:
        """Invoke the agent and return its final text response.

        Tool calls (≤ 200 chars of input) and their results (≤ 500 chars)
        are printed to stdout as the agent reasons through the problem.

        Parameters
        ----------
        message:
            The user message / task description.

        Returns
        -------
        str
            The agent's final answer.
        """
        final_content = ""

        for chunk in self._agent.stream(
            {"messages": [HumanMessage(content=message)]},
            stream_mode="updates",
            config={"recursion_limit": self._max_iterations * 3},
        ):
            # chunk is a dict keyed by node name: {"agent": {...}, "tools": {...}}
            for node_name, node_state in chunk.items():
                msgs: list[Any] = node_state.get("messages", [])
                for msg in msgs:
                    if isinstance(msg, AIMessage):
                        # Print any tool-call invocations
                        for tc in getattr(msg, "tool_calls", []):
                            args_str = str(tc.get("args", {}))
                            print(
                                f"\n{_CYAN}🔧 [{tc['name']}]{_RESET} "
                                f"{_truncate(args_str, 200)}"
                            )
                        # Capture the final text response
                        if msg.content and not getattr(msg, "tool_calls", []):
                            final_content = str(msg.content)

                    elif isinstance(msg, ToolMessage):
                        result_str = str(msg.content)
                        print(
                            f"{_GREEN}   → {_truncate(result_str, 500)}{_RESET}"
                        )

        return final_content or "No response."
