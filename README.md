# deep_agents_cli_agent

A general-purpose **CLI coding agent** powered by [LangGraph](https://github.com/langchain-ai/langgraph) and [Ollama](https://ollama.com), with 15 GitHub Copilot-inspired tools and optional [MCP](https://spec.modelcontextprotocol.io/) server support.

---

## Features

- **15 coding tools** (file I/O, regex search, Python runner, shell executor, git, URL fetch, …)
- **CLI-first** — interactive REPL or single-shot mode
- **Prompts for working directory** on startup (or pass `--cwd`)
- **Live tool tracing** — every tool call (≤ 200 chars) and result (≤ 500 chars) is printed in the terminal
- **Custom MCP servers** — plug in any stdio-based MCP server via `config.yaml` or `--mcp-server`
- **Config-driven model** — change the Ollama model name and endpoint in `config.yaml` without touching code

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Pull a model (requires Ollama running locally)
ollama pull llama3.2

# 3. Run the agent (interactive mode — it will ask for the working directory)
python cli.py

# Single-shot mode
python cli.py --cwd /my/project "List all Python files and summarise what they do"
```

---

## Tool List

| # | Name | Description |
|---|------|-------------|
| 1 | `read_file` | Read file contents with optional line range |
| 2 | `write_file` | Write / overwrite a file |
| 3 | `edit_file` | Search-and-replace within a file |
| 4 | `list_dir` | List directory contents |
| 5 | `find_files` | Find files by glob **or** regex name pattern |
| 6 | `grep_search` | Regex search inside file contents |
| 7 | `run_python` | Execute Python using the **current** interpreter (all installed packages available) |
| 8 | `run_command` | Execute shell commands |
| 9 | `create_directory` | Create directory (including parents) |
| 10 | `delete_file` | Delete a file or directory tree |
| 11 | `move_file` | Move / rename a file or directory |
| 12 | `get_file_info` | Metadata (size, mtime, permissions …) |
| 13 | `read_url` | Fetch a URL and return its text content |
| 14 | `git_diff` | Show git diff (working-tree or staged) |
| 15 | `git_log` | Show recent commit history |

---

## Configuration (`config.yaml`)

```yaml
model:
  provider: ollama          # currently supported: "ollama"
  name: llama3.2            # any model pulled with `ollama pull <name>`
  base_url: http://localhost:11434
  temperature: 0

agent:
  max_iterations: 20
  system_prompt: "You are a helpful coding assistant. …"

mcp_servers: []
  # - name: filesystem
  #   command: npx
  #   args: ["-y", "@modelcontextprotocol/server-filesystem", "."]
  #   env: {}
```

Pass a custom config file with `--config /path/to/config.yaml`.

---

## MCP Servers

Any stdio-based MCP server can be added:

```bash
python cli.py \
  --mcp-server '{"name":"fs","command":"npx","args":["-y","@modelcontextprotocol/server-filesystem","."]}' \
  "What is in the current directory?"
```

Or declare servers in `config.yaml` under `mcp_servers`.

---

## CLI Options

```
usage: coding-agent [-h] [--config FILE] [--cwd DIR] [--mcp-server JSON] [message]

positional arguments:
  message               Optional message (interactive mode if omitted)

options:
  --config FILE         Path to YAML config file
  --cwd DIR             Working directory (prompted if omitted)
  --mcp-server JSON     Extra MCP server config (repeatable)
```

---

## Tests

```bash
pytest -v
```

96 tests covering every tool, the agent class (with mocked LLM), the MCP client (with a real in-process fake server), the config loader, and the CLI.

---

## Python API

```python
from agent import CodingAgent

agent = CodingAgent(
    config_path="config.yaml",   # optional
    cwd="/my/project",           # optional, defaults to os.getcwd()
    mcp_servers=[                # optional extra MCP servers
        {"name": "srv", "command": "python", "args": ["my_mcp_server.py"]}
    ],
)

response = agent.invoke("Refactor the utils.py module to use dataclasses")
print(response)
```
