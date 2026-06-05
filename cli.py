#!/usr/bin/env python3
"""CLI entry point for the Deep Agents coding agent.

Usage
-----
Interactive mode (prompts for working directory):
    python cli.py

Single-shot mode:
    python cli.py "List all Python files in this project"

Custom config / working directory:
    python cli.py --config /path/to/config.yaml --cwd /my/project "Fix the bug"

Add extra MCP servers at runtime:
    python cli.py --mcp-server '{"name":"fs","command":"npx","args":["-y","@mcp/server-filesystem","."]}' "..."
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="coding-agent",
        description="Deep Agents CLI — general-purpose coding agent powered by Ollama",
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="FILE",
        help="Path to YAML config file (default: ./config.yaml or built-in)",
    )
    parser.add_argument(
        "--cwd",
        default=None,
        metavar="DIR",
        help="Working directory for file / shell tools (prompted if omitted)",
    )
    parser.add_argument(
        "--mcp-server",
        action="append",
        dest="mcp_servers",
        default=[],
        metavar="JSON",
        help=(
            "Extra MCP server as a JSON object with keys: name, command, args, env. "
            "Can be repeated."
        ),
    )
    parser.add_argument(
        "message",
        nargs="?",
        default=None,
        help="Optional message to send (if omitted, enter interactive mode)",
    )
    return parser.parse_args()


def _get_cwd(provided: str | None) -> str:
    if provided:
        cwd = os.path.abspath(provided)
    else:
        default = os.getcwd()
        try:
            raw = input(f"Working directory [{default}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            raw = ""
        cwd = os.path.abspath(raw) if raw else default

    if not os.path.isdir(cwd):
        print(f"Error: '{cwd}' is not a directory.", file=sys.stderr)
        sys.exit(1)
    return cwd


def _build_mcp_list(raw: list[str]) -> list[dict]:
    result = []
    for entry in raw:
        try:
            result.append(json.loads(entry))
        except json.JSONDecodeError as exc:
            print(f"Warning: ignoring invalid --mcp-server JSON: {exc}", file=sys.stderr)
    return result


def main() -> None:  # noqa: C901
    args = _parse_args()

    cwd = _get_cwd(args.cwd)
    mcp_servers = _build_mcp_list(args.mcp_servers)

    print(f"Working directory: {cwd}")
    print("Loading agent …", end=" ", flush=True)

    # Import here so startup errors surface clearly
    from agent import CodingAgent  # noqa: PLC0415

    try:
        agent = CodingAgent(
            config_path=args.config,
            cwd=cwd,
            mcp_servers=mcp_servers or None,
        )
    except Exception as exc:
        print(f"\nFailed to initialise agent: {exc}", file=sys.stderr)
        sys.exit(1)

    print("ready.\n")

    if args.message:
        # ---------------------------------------------------------------- #
        # Single-shot mode
        # ---------------------------------------------------------------- #
        response = agent.invoke(args.message)
        print(f"\nAgent: {response}")
        return

    # ---------------------------------------------------------------------- #
    # Interactive mode
    # ---------------------------------------------------------------------- #
    print("Deep Agents CLI Agent — type 'exit' or 'quit' to stop.")
    print("-" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break

        response = agent.invoke(user_input)
        print(f"\nAgent: {response}")


if __name__ == "__main__":
    main()
