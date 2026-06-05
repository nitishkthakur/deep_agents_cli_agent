"""Configuration loader for the coding agent."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# Bundled default config lives next to this file
_DEFAULT_CONFIG = Path(__file__).parent.parent / "config.yaml"

_BUILTIN_DEFAULTS: dict[str, Any] = {
    "model": {
        "provider": "ollama",
        "name": "llama3.2",
        "base_url": "http://localhost:11434",
        "temperature": 0,
    },
    "agent": {
        "max_iterations": 20,
        "system_prompt": (
            "You are a helpful general-purpose coding assistant. "
            "Use the available tools to help the user with their coding tasks."
        ),
    },
    "mcp_servers": [],
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (non-destructive copy)."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str | None = None) -> dict[str, Any]:
    """Load configuration from *config_path*, falling back to the bundled default.

    Priority (highest first):
    1. Explicit *config_path* argument
    2. ``./config.yaml`` in the current working directory
    3. ``~/.config/deep_agents_cli/config.yaml``
    4. Built-in defaults
    """
    candidates: list[Path] = []
    if config_path:
        candidates.append(Path(config_path))
    candidates.append(Path("config.yaml"))
    candidates.append(
        Path.home() / ".config" / "deep_agents_cli" / "config.yaml"
    )
    candidates.append(_DEFAULT_CONFIG)

    config: dict[str, Any] = dict(_BUILTIN_DEFAULTS)
    for path in candidates:
        if path.exists():
            with open(path, encoding="utf-8") as fh:
                loaded = yaml.safe_load(fh) or {}
            config = _deep_merge(config, loaded)
            break

    return config
