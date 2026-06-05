"""15 GitHub Copilot-inspired tools for the coding agent.

All tools are created via :func:`create_tools`, which wraps each function in a
LangChain ``@tool`` and closes over the working directory (*cwd*).

Tool list
---------
1.  read_file       — read file with optional line range
2.  write_file      — create / overwrite a file
3.  edit_file       — search-and-replace within a file
4.  list_dir        — list directory contents
5.  find_files      — find files by glob or regex name pattern
6.  grep_search     — regex search inside file contents
7.  run_python      — execute Python using the current interpreter
8.  run_command     — execute a shell command
9.  create_directory — create directory (including parents)
10. delete_file     — delete a file or directory tree
11. move_file       — move / rename a file or directory
12. get_file_info   — metadata (size, mtime, permissions …)
13. read_url        — fetch a URL and return its text content
14. git_diff        — show git diff (working-tree or staged)
15. git_log         — show recent git commit history
"""

from __future__ import annotations

import fnmatch
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve(path: str, cwd: str) -> str:
    """Resolve *path* relative to *cwd*; absolute paths are left as-is."""
    if os.path.isabs(path):
        return path
    return os.path.join(cwd, path)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def create_tools(cwd: str) -> list:
    """Return the list of 15 tools, each scoped to *cwd*."""

    # ------------------------------------------------------------------ #
    # 1. read_file
    # ------------------------------------------------------------------ #
    @tool
    def read_file(path: str, start_line: int = 1, end_line: int = -1) -> str:
        """Read the contents of a file and return them with line numbers.

        Args:
            path: Path to the file (absolute or relative to the working dir).
            start_line: First line to include (1-indexed, default 1).
            end_line: Last line to include inclusive (-1 means read to EOF).
        """
        abs_path = _resolve(path, cwd)
        try:
            with open(abs_path, encoding="utf-8", errors="replace") as fh:
                lines = fh.readlines()

            sl = max(1, start_line)
            el = len(lines) if end_line == -1 else end_line
            selected = lines[sl - 1 : el]

            numbered = [f"{sl + i}: {line}" for i, line in enumerate(selected)]
            header = f"File: {abs_path}  (lines {sl}–{sl + len(selected) - 1})\n"
            return header + "".join(numbered)
        except FileNotFoundError:
            return f"Error: file not found: {abs_path}"
        except Exception as exc:
            return f"Error reading {abs_path}: {exc}"

    # ------------------------------------------------------------------ #
    # 2. write_file
    # ------------------------------------------------------------------ #
    @tool
    def write_file(path: str, content: str) -> str:
        """Write *content* to *path*, creating parent directories as needed.

        Overwrites the file if it already exists.

        Args:
            path: Destination path (absolute or relative to the working dir).
            content: Text content to write.
        """
        abs_path = _resolve(path, cwd)
        try:
            os.makedirs(os.path.dirname(abs_path) or cwd, exist_ok=True)
            with open(abs_path, "w", encoding="utf-8") as fh:
                fh.write(content)
            return f"Wrote {len(content)} characters to {abs_path}"
        except Exception as exc:
            return f"Error writing {abs_path}: {exc}"

    # ------------------------------------------------------------------ #
    # 3. edit_file
    # ------------------------------------------------------------------ #
    @tool
    def edit_file(path: str, old_str: str, new_str: str) -> str:
        """Replace exactly one occurrence of *old_str* with *new_str* in a file.

        *old_str* must appear **exactly once** in the file (make it specific
        enough to be unambiguous).

        Args:
            path: Path to the file (absolute or relative to the working dir).
            old_str: The exact text to search for.
            new_str: The replacement text.
        """
        abs_path = _resolve(path, cwd)
        try:
            with open(abs_path, encoding="utf-8") as fh:
                content = fh.read()

            count = content.count(old_str)
            if count == 0:
                return f"Error: old_str not found in {abs_path}"
            if count > 1:
                return (
                    f"Error: old_str found {count} times in {abs_path}. "
                    "Add more context to make it unique."
                )

            new_content = content.replace(old_str, new_str, 1)
            with open(abs_path, "w", encoding="utf-8") as fh:
                fh.write(new_content)
            return f"Successfully edited {abs_path}"
        except FileNotFoundError:
            return f"Error: file not found: {abs_path}"
        except Exception as exc:
            return f"Error editing {abs_path}: {exc}"

    # ------------------------------------------------------------------ #
    # 4. list_dir
    # ------------------------------------------------------------------ #
    @tool
    def list_dir(path: str = ".") -> str:
        """List the contents of a directory.

        Directories are shown first, then files, both sorted alphabetically.

        Args:
            path: Directory to list (default: working directory).
        """
        abs_path = _resolve(path, cwd)
        try:
            entries = sorted(os.scandir(abs_path), key=lambda e: (not e.is_dir(), e.name))
            lines: list[str] = []
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    lines.append(f"[DIR]  {entry.name}/")
                else:
                    lines.append(f"[FILE] {entry.name}  ({entry.stat().st_size} B)")
            if not lines:
                return f"{abs_path} is empty."
            return f"Contents of {abs_path}:\n" + "\n".join(lines)
        except FileNotFoundError:
            return f"Error: directory not found: {abs_path}"
        except Exception as exc:
            return f"Error listing {abs_path}: {exc}"

    # ------------------------------------------------------------------ #
    # 5. find_files
    # ------------------------------------------------------------------ #
    @tool
    def find_files(pattern: str, directory: str = ".", recursive: bool = True) -> str:
        """Find files whose **names** match a glob or regex pattern.

        If *pattern* contains ``*``, ``?``, or ``[`` **and** does not contain
        regex anchors (``^``, ``$``) or escape sequences (``\\``), it is
        treated as a shell glob; otherwise it is compiled as a Python regex
        applied to the file name.

        Args:
            pattern: Glob or regex pattern to match against file names.
            directory: Root directory to search (default: working directory).
            recursive: Search subdirectories when True (default).
        """
        import glob as _glob

        abs_dir = _resolve(directory, cwd)
        try:
            results: list[str] = []
            # A pattern is a glob only when it has glob chars but no regex-
            # specific markers (anchors, escapes, groups, alternation …).
            _regex_markers = ("^", "$", "\\", "(", ")", "|", "+", "{")
            has_glob_chars = any(c in pattern for c in ("*", "?", "["))
            has_regex_markers = any(c in pattern for c in _regex_markers)
            is_glob = has_glob_chars and not has_regex_markers

            if is_glob:
                base = os.path.join(abs_dir, "**", pattern) if recursive else os.path.join(abs_dir, pattern)
                matches = _glob.glob(base, recursive=recursive)
                results = [os.path.relpath(m, cwd) for m in sorted(matches)]
            else:
                regex = re.compile(pattern)
                for root, dirs, files in os.walk(abs_dir):
                    dirs[:] = [d for d in sorted(dirs) if not d.startswith(".")]
                    for fname in sorted(files):
                        if regex.search(fname):
                            results.append(os.path.relpath(os.path.join(root, fname), cwd))
                    if not recursive:
                        break

            if not results:
                return f"No files matching '{pattern}' found in {abs_dir}"
            truncated = results[:200]
            out = f"Found {len(results)} file(s):\n" + "\n".join(truncated)
            if len(results) > 200:
                out += f"\n… ({len(results) - 200} more, truncated)"
            return out
        except re.error as exc:
            return f"Invalid regex '{pattern}': {exc}"
        except Exception as exc:
            return f"Error searching {abs_dir}: {exc}"

    # ------------------------------------------------------------------ #
    # 6. grep_search
    # ------------------------------------------------------------------ #
    @tool
    def grep_search(
        pattern: str,
        path: str = ".",
        file_pattern: str = "*",
        case_sensitive: bool = True,
    ) -> str:
        """Search for a regex pattern inside file contents.

        Returns matches as ``file:line: content`` lines (up to 200 results).

        Args:
            pattern: Python regex to search for.
            path: File or directory to search (default: working directory).
            file_pattern: Glob filter applied to file names, e.g. ``*.py``.
            case_sensitive: Use case-sensitive matching (default True).
        """
        abs_path = _resolve(path, cwd)
        try:
            flags = 0 if case_sensitive else re.IGNORECASE
            regex = re.compile(pattern, flags)
        except re.error as exc:
            return f"Invalid regex '{pattern}': {exc}"

        results: list[str] = []

        def _search_one(filepath: str) -> bool:
            """Return True when the global limit has been reached."""
            try:
                with open(filepath, encoding="utf-8", errors="replace") as fh:
                    for lineno, line in enumerate(fh, 1):
                        if regex.search(line):
                            rel = os.path.relpath(filepath, cwd)
                            results.append(f"{rel}:{lineno}: {line.rstrip()}")
                            if len(results) >= 200:
                                return True
            except Exception:
                pass
            return False

        if os.path.isfile(abs_path):
            _search_one(abs_path)
        else:
            for root, dirs, files in os.walk(abs_path):
                dirs[:] = [d for d in sorted(dirs) if not d.startswith(".")]
                for fname in sorted(files):
                    if fnmatch.fnmatch(fname, file_pattern):
                        if _search_one(os.path.join(root, fname)):
                            break

        if not results:
            return f"No matches for '{pattern}' in {abs_path}"
        out = f"Found {len(results)} match(es) for '{pattern}':\n" + "\n".join(results)
        if len(results) == 200:
            out += "\n… (truncated at 200 results)"
        return out

    # ------------------------------------------------------------------ #
    # 7. run_python
    # ------------------------------------------------------------------ #
    @tool
    def run_python(code: str, timeout: int = 60) -> str:
        """Execute Python code using the **current** Python interpreter.

        The script has access to all packages installed in the current
        environment.  It is run with the working directory set to *cwd*.

        Args:
            code: Python source code to execute.
            timeout: Maximum execution time in seconds (default 60).
        """
        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8"
            ) as tmp:
                tmp.write(code)
                tmp_path = tmp.name

            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
            )

            parts: list[str] = []
            if result.stdout:
                parts.append(f"STDOUT:\n{result.stdout.rstrip()}")
            if result.stderr:
                parts.append(f"STDERR:\n{result.stderr.rstrip()}")
            if result.returncode != 0:
                parts.append(f"Exit code: {result.returncode}")
            return "\n".join(parts) if parts else "(no output)"
        except subprocess.TimeoutExpired:
            return f"Error: execution timed out after {timeout} s"
        except Exception as exc:
            return f"Error running Python code: {exc}"
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    # ------------------------------------------------------------------ #
    # 8. run_command
    # ------------------------------------------------------------------ #
    @tool
    def run_command(command: str, timeout: int = 60) -> str:
        """Execute a shell command in the working directory.

        Returns combined stdout / stderr output.

        Args:
            command: Shell command string to execute.
            timeout: Maximum execution time in seconds (default 60).
        """
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
            )
            parts: list[str] = []
            if result.stdout:
                parts.append(f"STDOUT:\n{result.stdout.rstrip()}")
            if result.stderr:
                parts.append(f"STDERR:\n{result.stderr.rstrip()}")
            if result.returncode != 0:
                parts.append(f"Exit code: {result.returncode}")
            return "\n".join(parts) if parts else "(no output)"
        except subprocess.TimeoutExpired:
            return f"Error: command timed out after {timeout} s"
        except Exception as exc:
            return f"Error running command: {exc}"

    # ------------------------------------------------------------------ #
    # 9. create_directory
    # ------------------------------------------------------------------ #
    @tool
    def create_directory(path: str) -> str:
        """Create a directory and all necessary parent directories.

        Args:
            path: Directory path to create.
        """
        abs_path = _resolve(path, cwd)
        try:
            os.makedirs(abs_path, exist_ok=True)
            return f"Created directory: {abs_path}"
        except Exception as exc:
            return f"Error creating directory {abs_path}: {exc}"

    # ------------------------------------------------------------------ #
    # 10. delete_file
    # ------------------------------------------------------------------ #
    @tool
    def delete_file(path: str) -> str:
        """Delete a file or an entire directory tree.

        Args:
            path: Path to the file or directory to delete.
        """
        abs_path = _resolve(path, cwd)
        try:
            if os.path.isfile(abs_path) or os.path.islink(abs_path):
                os.remove(abs_path)
                return f"Deleted file: {abs_path}"
            elif os.path.isdir(abs_path):
                shutil.rmtree(abs_path)
                return f"Deleted directory tree: {abs_path}"
            else:
                return f"Error: path not found: {abs_path}"
        except Exception as exc:
            return f"Error deleting {abs_path}: {exc}"

    # ------------------------------------------------------------------ #
    # 11. move_file
    # ------------------------------------------------------------------ #
    @tool
    def move_file(source: str, destination: str) -> str:
        """Move or rename a file or directory.

        Args:
            source: Current path of the file or directory.
            destination: Target path (the new name / location).
        """
        abs_src = _resolve(source, cwd)
        abs_dst = _resolve(destination, cwd)
        try:
            parent = os.path.dirname(abs_dst)
            if parent:
                os.makedirs(parent, exist_ok=True)
            shutil.move(abs_src, abs_dst)
            return f"Moved {abs_src} → {abs_dst}"
        except Exception as exc:
            return f"Error moving {abs_src} → {abs_dst}: {exc}"

    # ------------------------------------------------------------------ #
    # 12. get_file_info
    # ------------------------------------------------------------------ #
    @tool
    def get_file_info(path: str) -> str:
        """Return metadata for a file or directory as JSON.

        Fields returned: path, type, size (bytes), modified, created,
        permissions (octal), and num_entries (for directories).

        Args:
            path: File or directory path.
        """
        import datetime

        abs_path = _resolve(path, cwd)
        try:
            st = os.stat(abs_path)
            info: dict = {
                "path": abs_path,
                "type": "directory" if os.path.isdir(abs_path) else "file",
                "size_bytes": st.st_size,
                "modified": datetime.datetime.fromtimestamp(st.st_mtime).isoformat(),
                "created": datetime.datetime.fromtimestamp(st.st_ctime).isoformat(),
                "permissions": oct(st.st_mode)[-3:],
            }
            if os.path.isdir(abs_path):
                info["num_entries"] = len(os.listdir(abs_path))
            return json.dumps(info, indent=2)
        except FileNotFoundError:
            return f"Error: path not found: {abs_path}"
        except Exception as exc:
            return f"Error getting info for {abs_path}: {exc}"

    # ------------------------------------------------------------------ #
    # 13. read_url
    # ------------------------------------------------------------------ #
    @tool
    def read_url(url: str) -> str:
        """Fetch a URL and return its text content (first 5 000 characters).

        For HTML pages, script/style tags are stripped and HTML entities are
        decoded.

        Args:
            url: The URL to fetch (http:// or https://).
        """
        import urllib.request

        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "CodingAgent/1.0 (+https://github.com)"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                ctype = resp.headers.get("Content-Type", "")
                raw = resp.read()

            enc = "utf-8"
            if "charset=" in ctype:
                enc = ctype.split("charset=")[-1].split(";")[0].strip()

            text = raw.decode(enc, errors="replace")

            if "html" in ctype.lower():
                text = re.sub(r"<script[^>]*>.*?</script[^>]*>", "", text, flags=re.DOTALL | re.IGNORECASE)
                text = re.sub(r"<style[^>]*>.*?</style[^>]*>", "", text, flags=re.DOTALL | re.IGNORECASE)
                text = re.sub(r"<[^>]+>", " ", text)
                text = html.unescape(text)
                text = re.sub(r"\s+", " ", text).strip()

            return text[:5000]
        except Exception as exc:
            return f"Error fetching {url}: {exc}"

    # ------------------------------------------------------------------ #
    # 14. git_diff
    # ------------------------------------------------------------------ #
    @tool
    def git_diff(path: str = ".", cached: bool = False) -> str:
        """Show the git diff for the working tree or staged changes.

        Args:
            path: File or directory to diff (default: entire repo).
            cached: If True, show staged (cached) changes.
        """
        abs_path = _resolve(path, cwd)
        try:
            cmd = ["git", "diff"]
            if cached:
                cmd.append("--cached")
            if path != ".":
                cmd += ["--", abs_path]

            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=cwd
            )
            if result.returncode != 0:
                return f"git diff error: {result.stderr.strip()}"
            return result.stdout or "No changes."
        except FileNotFoundError:
            return "Error: git is not installed or not in PATH"
        except Exception as exc:
            return f"Error running git diff: {exc}"

    # ------------------------------------------------------------------ #
    # 15. git_log
    # ------------------------------------------------------------------ #
    @tool
    def git_log(max_entries: int = 10, path: str = ".") -> str:
        """Show the recent git commit history.

        Args:
            max_entries: Number of commits to show (default 10).
            path: Limit history to commits that touch this path.
        """
        abs_path = _resolve(path, cwd)
        try:
            cmd = [
                "git", "log",
                f"-{max_entries}",
                "--format=%h  %ai  %an: %s",
            ]
            if path != ".":
                cmd += ["--", abs_path]

            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=cwd
            )
            if result.returncode != 0:
                return f"git log error: {result.stderr.strip()}"
            return result.stdout or "No commits found."
        except FileNotFoundError:
            return "Error: git is not installed or not in PATH"
        except Exception as exc:
            return f"Error running git log: {exc}"

    # ------------------------------------------------------------------ #
    # Return all 15 tools
    # ------------------------------------------------------------------ #
    return [
        read_file,
        write_file,
        edit_file,
        list_dir,
        find_files,
        grep_search,
        run_python,
        run_command,
        create_directory,
        delete_file,
        move_file,
        get_file_info,
        read_url,
        git_diff,
        git_log,
    ]
