"""Tests for the config loader."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agent.config import load_config


class TestLoadConfig:
    def test_returns_defaults_when_no_file(self, tmp_path: Path, monkeypatch):
        """With no config file present, built-in defaults should be returned."""
        monkeypatch.chdir(tmp_path)
        cfg = load_config(config_path="nonexistent.yaml")
        assert cfg["model"]["provider"] == "ollama"
        assert cfg["model"]["name"] == "llama3.2"
        assert cfg["agent"]["max_iterations"] == 20
        assert cfg["mcp_servers"] == []

    def test_loads_custom_model_name(self, tmp_path: Path):
        config_file = tmp_path / "c.yaml"
        config_file.write_text(
            yaml.dump({"model": {"name": "mistral"}}), encoding="utf-8"
        )
        cfg = load_config(str(config_file))
        assert cfg["model"]["name"] == "mistral"
        # Other defaults preserved
        assert cfg["model"]["provider"] == "ollama"

    def test_deep_merge_preserves_unset_keys(self, tmp_path: Path):
        config_file = tmp_path / "partial.yaml"
        config_file.write_text(
            yaml.dump({"agent": {"max_iterations": 5}}), encoding="utf-8"
        )
        cfg = load_config(str(config_file))
        assert cfg["agent"]["max_iterations"] == 5
        assert "system_prompt" in cfg["agent"]

    def test_mcp_servers_list(self, tmp_path: Path):
        config_file = tmp_path / "mcp.yaml"
        config_file.write_text(
            yaml.dump(
                {
                    "mcp_servers": [
                        {"name": "srv", "command": "python", "args": ["srv.py"]}
                    ]
                }
            ),
            encoding="utf-8",
        )
        cfg = load_config(str(config_file))
        assert len(cfg["mcp_servers"]) == 1
        assert cfg["mcp_servers"][0]["name"] == "srv"

    def test_explicit_path_takes_priority(self, tmp_path: Path, monkeypatch):
        """Explicit path beats ./config.yaml."""
        (tmp_path / "config.yaml").write_text(
            yaml.dump({"model": {"name": "local_default"}}), encoding="utf-8"
        )
        explicit = tmp_path / "explicit.yaml"
        explicit.write_text(
            yaml.dump({"model": {"name": "explicit_model"}}), encoding="utf-8"
        )
        monkeypatch.chdir(tmp_path)
        cfg = load_config(str(explicit))
        assert cfg["model"]["name"] == "explicit_model"
