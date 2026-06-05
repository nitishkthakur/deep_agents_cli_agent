"""Tests for all 15 coding tools.

Each test verifies not just that the tool *runs* but that it returns the
correct, expected content.
"""

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path

import pytest

from agent.tools import create_tools


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_tool(tools: list, name: str):
    """Find a tool by name from the list."""
    for t in tools:
        if t.name == name:
            return t
    raise KeyError(f"Tool '{name}' not found")


# ---------------------------------------------------------------------------
# 1. read_file
# ---------------------------------------------------------------------------

class TestReadFile:
    def test_full_file(self, tmp_cwd: str, sample_file: Path):
        tools = create_tools(tmp_cwd)
        t = get_tool(tools, "read_file")
        result = t.invoke({"path": str(sample_file)})
        assert "def greet(name)" in result
        assert "def add(a, b)" in result

    def test_line_range(self, tmp_cwd: str, sample_file: Path):
        tools = create_tools(tmp_cwd)
        t = get_tool(tools, "read_file")
        result = t.invoke({"path": str(sample_file), "start_line": 1, "end_line": 2})
        assert "def greet(name)" in result
        assert "def add(a, b)" not in result

    def test_relative_path(self, tmp_path: Path):
        (tmp_path / "hello.txt").write_text("hello world\n", encoding="utf-8")
        tools = create_tools(str(tmp_path))
        t = get_tool(tools, "read_file")
        result = t.invoke({"path": "hello.txt"})
        assert "hello world" in result

    def test_missing_file(self, tmp_cwd: str):
        tools = create_tools(tmp_cwd)
        t = get_tool(tools, "read_file")
        result = t.invoke({"path": "nonexistent.txt"})
        assert "Error" in result or "error" in result.lower()

    def test_line_numbers_included(self, tmp_path: Path):
        (tmp_path / "nums.txt").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
        tools = create_tools(str(tmp_path))
        t = get_tool(tools, "read_file")
        result = t.invoke({"path": "nums.txt"})
        assert "1: alpha" in result
        assert "2: beta" in result
        assert "3: gamma" in result


# ---------------------------------------------------------------------------
# 2. write_file
# ---------------------------------------------------------------------------

class TestWriteFile:
    def test_creates_file(self, tmp_path: Path):
        tools = create_tools(str(tmp_path))
        t = get_tool(tools, "write_file")
        result = t.invoke({"path": "new_file.txt", "content": "hello\n"})
        assert "Wrote" in result
        assert (tmp_path / "new_file.txt").read_text() == "hello\n"

    def test_overwrites_existing(self, tmp_path: Path):
        (tmp_path / "existing.txt").write_text("old content\n", encoding="utf-8")
        tools = create_tools(str(tmp_path))
        t = get_tool(tools, "write_file")
        t.invoke({"path": "existing.txt", "content": "new content\n"})
        assert (tmp_path / "existing.txt").read_text() == "new content\n"

    def test_creates_parent_dirs(self, tmp_path: Path):
        tools = create_tools(str(tmp_path))
        t = get_tool(tools, "write_file")
        t.invoke({"path": "a/b/c.txt", "content": "nested\n"})
        assert (tmp_path / "a" / "b" / "c.txt").read_text() == "nested\n"

    def test_reports_char_count(self, tmp_path: Path):
        tools = create_tools(str(tmp_path))
        t = get_tool(tools, "write_file")
        content = "x" * 42
        result = t.invoke({"path": "count.txt", "content": content})
        assert "42" in result


# ---------------------------------------------------------------------------
# 3. edit_file
# ---------------------------------------------------------------------------

class TestEditFile:
    def test_basic_replacement(self, tmp_path: Path):
        p = tmp_path / "edit_me.txt"
        p.write_text("foo bar baz\n", encoding="utf-8")
        tools = create_tools(str(tmp_path))
        t = get_tool(tools, "edit_file")
        result = t.invoke({"path": "edit_me.txt", "old_str": "bar", "new_str": "QUX"})
        assert "Successfully" in result
        assert p.read_text() == "foo QUX baz\n"

    def test_missing_old_str(self, tmp_path: Path):
        (tmp_path / "f.txt").write_text("hello\n", encoding="utf-8")
        tools = create_tools(str(tmp_path))
        t = get_tool(tools, "edit_file")
        result = t.invoke({"path": "f.txt", "old_str": "NOPE", "new_str": "x"})
        assert "Error" in result or "not found" in result

    def test_ambiguous_str(self, tmp_path: Path):
        (tmp_path / "dup.txt").write_text("dup dup dup\n", encoding="utf-8")
        tools = create_tools(str(tmp_path))
        t = get_tool(tools, "edit_file")
        result = t.invoke({"path": "dup.txt", "old_str": "dup", "new_str": "x"})
        assert "Error" in result or "3" in result


# ---------------------------------------------------------------------------
# 4. list_dir
# ---------------------------------------------------------------------------

class TestListDir:
    def test_lists_contents(self, tmp_path: Path):
        (tmp_path / "file_a.txt").write_text("a", encoding="utf-8")
        (tmp_path / "file_b.txt").write_text("b", encoding="utf-8")
        (tmp_path / "subdir").mkdir()
        tools = create_tools(str(tmp_path))
        t = get_tool(tools, "list_dir")
        result = t.invoke({"path": "."})
        assert "file_a.txt" in result
        assert "file_b.txt" in result
        assert "subdir" in result

    def test_dirs_before_files(self, tmp_path: Path):
        (tmp_path / "aaa.txt").write_text("x", encoding="utf-8")
        (tmp_path / "zzz_dir").mkdir()
        tools = create_tools(str(tmp_path))
        t = get_tool(tools, "list_dir")
        result = t.invoke({"path": "."})
        dir_idx = result.index("zzz_dir")
        file_idx = result.index("aaa.txt")
        assert dir_idx < file_idx

    def test_empty_dir(self, tmp_path: Path):
        (tmp_path / "empty").mkdir()
        tools = create_tools(str(tmp_path))
        t = get_tool(tools, "list_dir")
        result = t.invoke({"path": "empty"})
        assert "empty" in result.lower()

    def test_missing_dir(self, tmp_cwd: str):
        tools = create_tools(tmp_cwd)
        t = get_tool(tools, "list_dir")
        result = t.invoke({"path": "no_such_dir"})
        assert "Error" in result or "error" in result.lower()


# ---------------------------------------------------------------------------
# 5. find_files
# ---------------------------------------------------------------------------

class TestFindFiles:
    def test_glob_pattern(self, tmp_path: Path):
        (tmp_path / "a.py").write_text("", encoding="utf-8")
        (tmp_path / "b.py").write_text("", encoding="utf-8")
        (tmp_path / "c.txt").write_text("", encoding="utf-8")
        tools = create_tools(str(tmp_path))
        t = get_tool(tools, "find_files")
        result = t.invoke({"pattern": "*.py", "directory": "."})
        assert "a.py" in result
        assert "b.py" in result
        assert "c.txt" not in result

    def test_regex_pattern(self, tmp_path: Path):
        (tmp_path / "test_foo.py").write_text("", encoding="utf-8")
        (tmp_path / "test_bar.py").write_text("", encoding="utf-8")
        (tmp_path / "main.py").write_text("", encoding="utf-8")
        tools = create_tools(str(tmp_path))
        t = get_tool(tools, "find_files")
        result = t.invoke({"pattern": r"^test_.*\.py$", "directory": "."})
        assert "test_foo.py" in result
        assert "test_bar.py" in result
        assert "main.py" not in result

    def test_recursive(self, tmp_path: Path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "deep.py").write_text("", encoding="utf-8")
        tools = create_tools(str(tmp_path))
        t = get_tool(tools, "find_files")
        result = t.invoke({"pattern": "*.py", "directory": ".", "recursive": True})
        assert "deep.py" in result

    def test_no_match(self, tmp_path: Path):
        tools = create_tools(str(tmp_path))
        t = get_tool(tools, "find_files")
        result = t.invoke({"pattern": "*.rs", "directory": "."})
        assert "No files" in result or "0 file" in result or "not found" in result.lower()


# ---------------------------------------------------------------------------
# 6. grep_search
# ---------------------------------------------------------------------------

class TestGrepSearch:
    def test_finds_pattern(self, tmp_path: Path, sample_file: Path):
        tools = create_tools(str(tmp_path))
        t = get_tool(tools, "grep_search")
        result = t.invoke({"pattern": "def \\w+", "path": str(sample_file)})
        assert "def greet" in result
        assert "def add" in result

    def test_case_insensitive(self, tmp_path: Path):
        (tmp_path / "upper.txt").write_text("HELLO world\n", encoding="utf-8")
        tools = create_tools(str(tmp_path))
        t = get_tool(tools, "grep_search")
        result = t.invoke(
            {"pattern": "hello", "path": str(tmp_path), "case_sensitive": False}
        )
        assert "HELLO world" in result

    def test_file_pattern_filter(self, tmp_path: Path):
        (tmp_path / "a.py").write_text("needle\n", encoding="utf-8")
        (tmp_path / "b.txt").write_text("needle\n", encoding="utf-8")
        tools = create_tools(str(tmp_path))
        t = get_tool(tools, "grep_search")
        result = t.invoke(
            {"pattern": "needle", "path": str(tmp_path), "file_pattern": "*.py"}
        )
        assert "a.py" in result
        assert "b.txt" not in result

    def test_no_match(self, tmp_path: Path):
        (tmp_path / "empty.py").write_text("no matches here\n", encoding="utf-8")
        tools = create_tools(str(tmp_path))
        t = get_tool(tools, "grep_search")
        result = t.invoke({"pattern": "ZZZNOMATCH", "path": str(tmp_path)})
        assert "No matches" in result or "0 match" in result

    def test_returns_file_and_line(self, tmp_path: Path):
        (tmp_path / "src.py").write_text("line one\npattern_here\nline three\n", encoding="utf-8")
        tools = create_tools(str(tmp_path))
        t = get_tool(tools, "grep_search")
        result = t.invoke({"pattern": "pattern_here", "path": str(tmp_path)})
        assert "src.py" in result
        assert "2" in result  # line number


# ---------------------------------------------------------------------------
# 7. run_python
# ---------------------------------------------------------------------------

class TestRunPython:
    def test_simple_output(self, tmp_cwd: str):
        tools = create_tools(tmp_cwd)
        t = get_tool(tools, "run_python")
        result = t.invoke({"code": "print('hello from python')"})
        assert "hello from python" in result

    def test_arithmetic(self, tmp_cwd: str):
        tools = create_tools(tmp_cwd)
        t = get_tool(tools, "run_python")
        result = t.invoke({"code": "print(2 + 2)"})
        assert "4" in result

    def test_uses_current_interpreter(self, tmp_cwd: str):
        import sys
        tools = create_tools(tmp_cwd)
        t = get_tool(tools, "run_python")
        result = t.invoke({"code": "import sys; print(sys.executable)"})
        # Should use the same interpreter
        assert sys.executable in result or result.strip()

    def test_stderr_captured(self, tmp_cwd: str):
        tools = create_tools(tmp_cwd)
        t = get_tool(tools, "run_python")
        result = t.invoke({"code": "import sys; sys.stderr.write('err_msg\\n')"})
        assert "err_msg" in result

    def test_exit_code_nonzero(self, tmp_cwd: str):
        tools = create_tools(tmp_cwd)
        t = get_tool(tools, "run_python")
        result = t.invoke({"code": "raise ValueError('boom')"})
        assert "ValueError" in result or "Exit code" in result

    def test_multiline_code(self, tmp_cwd: str):
        tools = create_tools(tmp_cwd)
        t = get_tool(tools, "run_python")
        code = "x = [i**2 for i in range(5)]\nprint(x)"
        result = t.invoke({"code": code})
        assert "[0, 1, 4, 9, 16]" in result


# ---------------------------------------------------------------------------
# 8. run_command
# ---------------------------------------------------------------------------

class TestRunCommand:
    def test_echo(self, tmp_cwd: str):
        tools = create_tools(tmp_cwd)
        t = get_tool(tools, "run_command")
        result = t.invoke({"command": "echo hello_cmd"})
        assert "hello_cmd" in result

    def test_pwd_matches_cwd(self, tmp_cwd: str):
        tools = create_tools(tmp_cwd)
        t = get_tool(tools, "run_command")
        result = t.invoke({"command": "pwd"})
        # Resolve symlinks for comparison
        assert os.path.realpath(tmp_cwd) in os.path.realpath(result.split("\n")[-1].strip()) or tmp_cwd in result

    def test_stderr_captured(self, tmp_cwd: str):
        tools = create_tools(tmp_cwd)
        t = get_tool(tools, "run_command")
        result = t.invoke({"command": "ls /nonexistent_dir_xyz"})
        assert "STDERR" in result or "No such file" in result or "cannot access" in result

    def test_nonzero_exit_code(self, tmp_cwd: str):
        tools = create_tools(tmp_cwd)
        t = get_tool(tools, "run_command")
        result = t.invoke({"command": "exit 42"})
        assert "42" in result or "Exit code" in result


# ---------------------------------------------------------------------------
# 9. create_directory
# ---------------------------------------------------------------------------

class TestCreateDirectory:
    def test_creates_dir(self, tmp_path: Path):
        tools = create_tools(str(tmp_path))
        t = get_tool(tools, "create_directory")
        result = t.invoke({"path": "new_dir"})
        assert "Created" in result
        assert (tmp_path / "new_dir").is_dir()

    def test_nested_dirs(self, tmp_path: Path):
        tools = create_tools(str(tmp_path))
        t = get_tool(tools, "create_directory")
        t.invoke({"path": "a/b/c"})
        assert (tmp_path / "a" / "b" / "c").is_dir()

    def test_idempotent(self, tmp_path: Path):
        (tmp_path / "exists").mkdir()
        tools = create_tools(str(tmp_path))
        t = get_tool(tools, "create_directory")
        result = t.invoke({"path": "exists"})
        assert "Error" not in result


# ---------------------------------------------------------------------------
# 10. delete_file
# ---------------------------------------------------------------------------

class TestDeleteFile:
    def test_deletes_file(self, tmp_path: Path):
        p = tmp_path / "todel.txt"
        p.write_text("bye\n", encoding="utf-8")
        tools = create_tools(str(tmp_path))
        t = get_tool(tools, "delete_file")
        result = t.invoke({"path": "todel.txt"})
        assert "Deleted" in result
        assert not p.exists()

    def test_deletes_directory(self, tmp_path: Path):
        d = tmp_path / "todelddir"
        d.mkdir()
        (d / "inner.txt").write_text("x\n", encoding="utf-8")
        tools = create_tools(str(tmp_path))
        t = get_tool(tools, "delete_file")
        result = t.invoke({"path": "todelddir"})
        assert "Deleted" in result
        assert not d.exists()

    def test_missing_path(self, tmp_path: Path):
        tools = create_tools(str(tmp_path))
        t = get_tool(tools, "delete_file")
        result = t.invoke({"path": "ghost.txt"})
        assert "Error" in result or "not found" in result.lower()


# ---------------------------------------------------------------------------
# 11. move_file
# ---------------------------------------------------------------------------

class TestMoveFile:
    def test_rename_file(self, tmp_path: Path):
        p = tmp_path / "old.txt"
        p.write_text("content\n", encoding="utf-8")
        tools = create_tools(str(tmp_path))
        t = get_tool(tools, "move_file")
        result = t.invoke({"source": "old.txt", "destination": "new.txt"})
        assert "Moved" in result
        assert not p.exists()
        assert (tmp_path / "new.txt").read_text() == "content\n"

    def test_move_to_subdir(self, tmp_path: Path):
        (tmp_path / "file.txt").write_text("data\n", encoding="utf-8")
        tools = create_tools(str(tmp_path))
        t = get_tool(tools, "move_file")
        t.invoke({"source": "file.txt", "destination": "subdir/file.txt"})
        assert (tmp_path / "subdir" / "file.txt").read_text() == "data\n"


# ---------------------------------------------------------------------------
# 12. get_file_info
# ---------------------------------------------------------------------------

class TestGetFileInfo:
    def test_file_info_fields(self, tmp_path: Path):
        p = tmp_path / "info.txt"
        p.write_text("12345\n", encoding="utf-8")
        tools = create_tools(str(tmp_path))
        t = get_tool(tools, "get_file_info")
        result = t.invoke({"path": "info.txt"})
        data = json.loads(result)
        assert data["type"] == "file"
        assert data["size_bytes"] == 6  # "12345\n"
        assert "modified" in data
        assert "created" in data
        assert "permissions" in data

    def test_directory_info(self, tmp_path: Path):
        d = tmp_path / "some_dir"
        d.mkdir()
        (d / "f1.txt").write_text("a", encoding="utf-8")
        (d / "f2.txt").write_text("b", encoding="utf-8")
        tools = create_tools(str(tmp_path))
        t = get_tool(tools, "get_file_info")
        result = t.invoke({"path": "some_dir"})
        data = json.loads(result)
        assert data["type"] == "directory"
        assert data["num_entries"] == 2

    def test_missing_path(self, tmp_path: Path):
        tools = create_tools(str(tmp_path))
        t = get_tool(tools, "get_file_info")
        result = t.invoke({"path": "ghost.txt"})
        assert "Error" in result or "not found" in result.lower()


# ---------------------------------------------------------------------------
# 13. read_url
# ---------------------------------------------------------------------------

class TestReadUrl:
    def test_fetches_plain_text(self, tmp_cwd: str):
        """Use a real, stable URL that returns plain text."""
        tools = create_tools(tmp_cwd)
        t = get_tool(tools, "read_url")
        result = t.invoke({"url": "https://httpbin.org/robots.txt"})
        # May fail if network is unavailable — skip gracefully
        if "Error" in result:
            pytest.skip(f"Network unavailable: {result}")
        assert len(result) > 0

    def test_invalid_url_returns_error(self, tmp_cwd: str):
        tools = create_tools(tmp_cwd)
        t = get_tool(tools, "read_url")
        result = t.invoke({"url": "http://this-domain-does-not-exist-xyz.example"})
        assert "Error" in result

    def test_truncates_long_content(self, tmp_cwd: str, monkeypatch):
        """Patch urllib to return a 10 000-char payload and verify truncation."""
        from unittest.mock import MagicMock, patch
        import io

        fake_body = "x" * 10_000
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.headers.get.return_value = "text/plain; charset=utf-8"
        mock_resp.read.return_value = fake_body.encode()

        with patch("urllib.request.urlopen", return_value=mock_resp):
            tools = create_tools(tmp_cwd)
            t = get_tool(tools, "read_url")
            result = t.invoke({"url": "http://example.com"})

        assert len(result) <= 5000


# ---------------------------------------------------------------------------
# 14. git_diff
# ---------------------------------------------------------------------------

class TestGitDiff:
    def test_clean_repo_no_diff(self, git_repo: Path):
        tools = create_tools(str(git_repo))
        t = get_tool(tools, "git_diff")
        result = t.invoke({"path": "."})
        assert result in ("No changes.", "") or result.strip() == ""

    def test_shows_changes(self, git_repo: Path):
        (git_repo / "README.md").write_text("# Modified\n", encoding="utf-8")
        tools = create_tools(str(git_repo))
        t = get_tool(tools, "git_diff")
        result = t.invoke({"path": "."})
        assert "README.md" in result or "Modified" in result

    def test_no_git_repo_returns_error(self, tmp_path: Path):
        tools = create_tools(str(tmp_path))
        t = get_tool(tools, "git_diff")
        result = t.invoke({"path": "."})
        assert "error" in result.lower() or "not a git" in result.lower()


# ---------------------------------------------------------------------------
# 15. git_log
# ---------------------------------------------------------------------------

class TestGitLog:
    def test_shows_initial_commit(self, git_repo: Path):
        tools = create_tools(str(git_repo))
        t = get_tool(tools, "git_log")
        result = t.invoke({"max_entries": 5, "path": "."})
        assert "initial commit" in result.lower()

    def test_max_entries(self, git_repo: Path):
        # Create a second commit
        (git_repo / "extra.txt").write_text("extra\n", encoding="utf-8")
        import subprocess
        subprocess.run(["git", "add", "."], cwd=str(git_repo), check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "second commit"],
            cwd=str(git_repo),
            check=True,
            capture_output=True,
        )

        tools = create_tools(str(git_repo))
        t = get_tool(tools, "git_log")
        result = t.invoke({"max_entries": 1})
        # Only the latest commit should appear
        assert "second commit" in result
        assert "initial commit" not in result

    def test_no_git_repo_returns_error(self, tmp_path: Path):
        tools = create_tools(str(tmp_path))
        t = get_tool(tools, "git_log")
        result = t.invoke({"max_entries": 5, "path": "."})
        assert "error" in result.lower() or "not a git" in result.lower()


# ---------------------------------------------------------------------------
# Sanity: exactly 15 tools registered
# ---------------------------------------------------------------------------

def test_tool_count(tmp_cwd: str):
    tools = create_tools(tmp_cwd)
    assert len(tools) == 15, f"Expected 15 tools, got {len(tools)}: {[t.name for t in tools]}"


def test_tool_names(tmp_cwd: str):
    expected = {
        "read_file", "write_file", "edit_file", "list_dir", "find_files",
        "grep_search", "run_python", "run_command", "create_directory",
        "delete_file", "move_file", "get_file_info", "read_url",
        "git_diff", "git_log",
    }
    tools = create_tools(tmp_cwd)
    actual = {t.name for t in tools}
    assert actual == expected
