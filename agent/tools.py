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

Common tool combinations
-------------------------
- **Explore → Understand → Edit**: Use ``list_dir`` + ``find_files`` to locate
  files, ``read_file`` to examine them, ``grep_search`` to find specific code,
  then ``edit_file`` to apply changes.
- **Refactor / Rename**: Use ``grep_search`` to find all references, then
  ``edit_file`` each occurrence.  Follow up with ``run_command`` to run tests.
- **Scaffold a new feature**: ``create_directory`` + ``write_file`` to build
  the file tree, then ``run_command`` or ``run_python`` to verify.
- **Debug an issue**: ``grep_search`` to find error strings, ``read_file`` to
  inspect surrounding code, ``git_log`` / ``git_diff`` to check recent changes.
- **Review changes before committing**: ``git_diff`` to see what changed,
  ``read_file`` with line ranges to inspect context, ``run_command`` to run
  linters/tests.
- **Research & integrate**: ``read_url`` to fetch API docs or references,
  then ``write_file`` / ``edit_file`` to integrate findings.
- **Clean up**: ``find_files`` to locate stale files, ``get_file_info`` to
  check sizes/dates, then ``delete_file`` or ``move_file`` to reorganise.
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
from typing import Annotated, Optional

from langchain_core.tools import tool
from pydantic import Field


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
    def read_file(
        path: Annotated[str, Field(description="File path, e.g. 'src/main.py'")],
        start_line: Annotated[int, Field(description="First line (1-indexed), e.g. 10")] = 1,
        end_line: Annotated[int, Field(description="Last line inclusive (-1 = EOF), e.g. 25")] = -1,
    ) -> str:
        """Read the contents of a file and return them with line numbers.

        Use this tool to inspect source code, configuration files, logs, or
        any text file.  Supports reading a specific line range to focus on a
        particular section without loading the entire file.

        Args:
            path: Path to the file (absolute or relative to the working dir).
            start_line: First line to include (1-indexed, default 1).
            end_line: Last line to include inclusive (-1 means read to EOF).

        Examples:
            Read an entire file:
                read_file(path="src/main.py")
            Read lines 10–25 only:
                read_file(path="src/main.py", start_line=10, end_line=25)

        Combine with:
            - ``grep_search`` → find a pattern first, then use read_file with
              a line range to inspect the surrounding context.
            - ``edit_file`` → read a file to understand its content, then use
              edit_file to apply targeted changes.
            - ``get_file_info`` → check file size before reading large files.
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
    def write_file(
        path: Annotated[str, Field(description="Destination file path, e.g. 'src/utils.py'")],
        content: Annotated[str, Field(description="Text content to write to the file")],
    ) -> str:
        """Write *content* to *path*, creating parent directories as needed.

        Use this tool to create new files or overwrite existing ones.  Parent
        directories are created automatically, so you can write to deeply
        nested paths in a single call.  For partial modifications of an
        existing file, prefer ``edit_file`` instead.

        Overwrites the file if it already exists.

        Args:
            path: Destination path (absolute or relative to the working dir).
            content: Text content to write.

        Examples:
            Create a new Python module:
                write_file(path="src/utils/helpers.py", content="def greet():\\n    return 'Hello'\\n")
            Overwrite a config file:
                write_file(path=".env", content="DEBUG=true\\nPORT=8080\\n")

        Combine with:
            - ``read_file`` → read a template or reference file, then write a
              new file based on it.
            - ``create_directory`` → although write_file auto-creates parents,
              use create_directory first when you need to set up a whole tree.
            - ``run_command`` / ``run_python`` → write a script then execute it.
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
    def edit_file(
        path: Annotated[str, Field(description="File to edit, e.g. 'app.py'")],
        old_str: Annotated[str, Field(description="Exact text to find (must be unique), e.g. 'import os'")],
        new_str: Annotated[str, Field(description="Replacement text, e.g. 'import os\\nimport sys'")],
    ) -> str:
        """Replace exactly one occurrence of *old_str* with *new_str* in a file.

        Use this tool for surgical, targeted edits to existing files — fixing
        bugs, updating imports, renaming variables, or modifying configuration
        values.  The match must be unique (include enough surrounding context).
        For creating new files or full rewrites, use ``write_file`` instead.

        *old_str* must appear **exactly once** in the file (make it specific
        enough to be unambiguous).

        Args:
            path: Path to the file (absolute or relative to the working dir).
            old_str: The exact text to search for.
            new_str: The replacement text.

        Examples:
            Fix a typo:
                edit_file(path="README.md", old_str="teh quick", new_str="the quick")
            Update an import:
                edit_file(path="app.py",
                          old_str="from utils import old_helper",
                          new_str="from utils import new_helper")
            Add a line after an existing one (include the anchor line):
                edit_file(path="config.yaml",
                          old_str="debug: false",
                          new_str="debug: false\\nverbose: true")

        Combine with:
            - ``read_file`` → inspect the file first to find the exact text to
              replace and verify context.
            - ``grep_search`` → locate all occurrences of a pattern across
              files, then edit_file each one.
            - ``git_diff`` → after editing, verify the change looks correct.
            - ``run_command`` → run tests after editing to confirm correctness.
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
    def list_dir(
        path: Annotated[str, Field(description="Directory to list, e.g. 'src/components'")] = ".",
    ) -> str:
        """List the contents of a directory.

        Use this tool to explore project structure, discover files and
        sub-directories, or verify that files were created/deleted correctly.
        Directories are shown first, then files, both sorted alphabetically.
        File sizes are included to help gauge content.

        Args:
            path: Directory to list (default: working directory).

        Examples:
            List the project root:
                list_dir()
            List a specific subdirectory:
                list_dir(path="src/components")

        Combine with:
            - ``find_files`` → use list_dir for a broad overview, then
              find_files to locate specific files by name pattern.
            - ``read_file`` → list directory contents, then read interesting
              files.
            - ``get_file_info`` → list_dir for an overview, then get_file_info
              for detailed metadata on specific entries.
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
    def find_files(
        pattern: Annotated[str, Field(description="Glob or regex pattern, e.g. '*.py' or '^test_.*\\.py$'")],
        directory: Annotated[str, Field(description="Root directory to search, e.g. 'src'")] = ".",
        recursive: Annotated[bool, Field(description="Search subdirectories (default True)")] = True,
    ) -> str:
        """Find files whose **names** match a glob or regex pattern.

        Use this tool to locate files by name across the project — e.g., find
        all Python test files, all YAML configs, or files matching a naming
        convention.  Supports both shell-style globs (``*.py``, ``test_*``)
        and Python regular expressions for complex patterns.

        If *pattern* contains ``*``, ``?``, or ``[`` **and** does not contain
        regex anchors (``^``, ``$``) or escape sequences (``\\``), it is
        treated as a shell glob; otherwise it is compiled as a Python regex
        applied to the file name.

        Args:
            pattern: Glob or regex pattern to match against file names.
            directory: Root directory to search (default: working directory).
            recursive: Search subdirectories when True (default).

        Examples:
            Find all Python files:
                find_files(pattern="*.py")
            Find test files in a specific directory:
                find_files(pattern="test_*.py", directory="tests")
            Find files matching a regex (e.g., versioned migrations):
                find_files(pattern="^\\d{4}_.*\\.sql$")
            Non-recursive search in one directory:
                find_files(pattern="*.json", directory="config", recursive=False)

        Combine with:
            - ``read_file`` → find files first, then read the ones you need.
            - ``grep_search`` → find_files locates files by *name*;
              grep_search finds files by *content*.  Use both together to
              narrow down search results.
            - ``delete_file`` / ``move_file`` → find stale or misplaced files,
              then clean them up.
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
        pattern: Annotated[str, Field(description="Python regex to search for, e.g. 'def calculate'")],
        path: Annotated[str, Field(description="File or directory to search, e.g. 'src/'")] = ".",
        file_pattern: Annotated[str, Field(description="Glob filter for file names, e.g. '*.py'")] = "*",
        case_sensitive: Annotated[bool, Field(description="Case-sensitive matching (default True)")] = True,
    ) -> str:
        """Search for a regex pattern inside file contents.

        Use this tool to find specific code, error messages, TODO comments,
        function definitions, import statements, or any text pattern across
        your codebase.  This is the primary tool for understanding *where*
        something is used or defined.

        Returns matches as ``file:line: content`` lines (up to 200 results).

        Args:
            pattern: Python regex to search for.
            path: File or directory to search (default: working directory).
            file_pattern: Glob filter applied to file names, e.g. ``*.py``.
            case_sensitive: Use case-sensitive matching (default True).

        Examples:
            Find all usages of a function:
                grep_search(pattern="calculate_total")
            Find TODO comments in Python files only:
                grep_search(pattern="TODO|FIXME", file_pattern="*.py")
            Case-insensitive search for an error message:
                grep_search(pattern="connection refused", case_sensitive=False)
            Search within a single file:
                grep_search(pattern="def test_", path="tests/test_api.py")
            Find class definitions:
                grep_search(pattern="^class\\s+\\w+", file_pattern="*.py")

        Combine with:
            - ``read_file`` → grep_search to find the line, then read_file
              with a line range to see the full context around the match.
            - ``edit_file`` → find all occurrences of a pattern, then
              edit_file to update each one (useful for refactoring).
            - ``find_files`` → use find_files to locate files by name, then
              grep_search to search within those specific files.
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
    def run_python(
        code: Annotated[str, Field(description="Python source code to execute, e.g. 'print(2 + 2)'")],
        timeout: Annotated[int, Field(description="Max execution time in seconds, e.g. 30")] = 60,
    ) -> str:
        """Execute Python code using the **current** Python interpreter.

        Use this tool to run data transformations, test snippets, perform
        calculations, parse/generate files programmatically, or validate
        logic.  The script runs in a subprocess with access to all installed
        packages and the working directory set to *cwd*.

        Prefer this over ``run_command`` when the task is inherently
        Python-based (e.g., JSON/CSV processing, regex testing, math).

        Args:
            code: Python source code to execute.
            timeout: Maximum execution time in seconds (default 60).

        Examples:
            Quick calculation:
                run_python(code="print(2 ** 100)")
            Parse and inspect a JSON file:
                run_python(code="import json\\ndata = json.load(open('config.json'))\\nprint(json.dumps(data, indent=2))")
            List installed packages:
                run_python(code="import pkg_resources\\nfor p in sorted(pkg_resources.working_set, key=lambda x: x.key):\\n    print(p)")

        Combine with:
            - ``read_file`` → read a data file, then run_python to analyse or
              transform it.
            - ``write_file`` → generate content with run_python, then
              write_file to save the output (or write directly in the script).
            - ``run_command`` → use run_python for Python-specific logic and
              run_command for shell operations like installing packages or
              running build tools.
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
    def run_command(
        command: Annotated[str, Field(description="Shell command to execute, e.g. 'pytest -v tests/'")],
        timeout: Annotated[int, Field(description="Max execution time in seconds, e.g. 120")] = 60,
    ) -> str:
        """Execute a shell command in the working directory.

        Use this tool to run build tools, linters, test suites, git commands,
        package managers, or any CLI program.  Returns combined stdout/stderr.

        Prefer ``run_python`` for Python-specific tasks.  Use this tool for
        everything else: ``npm``, ``make``, ``cargo``, ``pytest``, ``git``,
        ``curl``, ``docker``, etc.

        Args:
            command: Shell command string to execute.
            timeout: Maximum execution time in seconds (default 60).

        Examples:
            Run tests:
                run_command(command="python -m pytest -v tests/")
            Install a dependency:
                run_command(command="pip install requests")
            Check git status:
                run_command(command="git status")
            Run a linter:
                run_command(command="flake8 src/ --max-line-length=120")

        Combine with:
            - ``edit_file`` → make code changes, then run_command to run tests
              and verify correctness.
            - ``write_file`` → create a Makefile, Dockerfile, or script, then
              run_command to execute it.
            - ``git_diff`` / ``git_log`` → use git_diff for simple diffs but
              run_command for complex git operations (rebase, cherry-pick, etc.).
            - ``grep_search`` → search for failing test names, then
              run_command to re-run only those tests.
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
    def create_directory(
        path: Annotated[str, Field(description="Directory path to create, e.g. 'src/components/auth'")],
    ) -> str:
        """Create a directory and all necessary parent directories.

        Use this tool to set up project scaffolding or ensure a directory
        exists before writing files into it.  Silently succeeds if the
        directory already exists (``exist_ok=True``).

        Note: ``write_file`` also auto-creates parent directories, so you
        only need this tool when creating empty directories or setting up
        a directory tree before populating it.

        Args:
            path: Directory path to create.

        Examples:
            Create a nested directory structure:
                create_directory(path="src/components/auth")
            Ensure an output directory exists:
                create_directory(path="build/output")

        Combine with:
            - ``write_file`` → create the directory structure first, then
              write files into it.
            - ``list_dir`` → verify the directory was created successfully.
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
    def delete_file(
        path: Annotated[str, Field(description="Path to file or directory to delete, e.g. 'build/cache'")],
    ) -> str:
        """Delete a file or an entire directory tree.

        Use this tool to remove obsolete files, clean up generated artifacts,
        or delete directories that are no longer needed.  For directories,
        this removes the entire tree recursively — use with care.

        Args:
            path: Path to the file or directory to delete.

        Examples:
            Delete a single file:
                delete_file(path="temp_output.txt")
            Delete an entire directory:
                delete_file(path="build/cache")

        Combine with:
            - ``find_files`` → locate files matching a pattern (e.g., ``*.pyc``,
              ``*.log``), then delete_file to remove each one.
            - ``get_file_info`` → check file age or size before deciding
              whether to delete.
            - ``list_dir`` → verify the file/directory is gone after deletion.
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
    def move_file(
        source: Annotated[str, Field(description="Current path, e.g. 'old_name.py'")],
        destination: Annotated[str, Field(description="Target path, e.g. 'src/new_name.py'")],
    ) -> str:
        """Move or rename a file or directory.

        Use this tool to reorganise project structure, rename files for
        clarity, or relocate resources.  Parent directories at the
        destination are created automatically.

        Args:
            source: Current path of the file or directory.
            destination: Target path (the new name / location).

        Examples:
            Rename a file:
                move_file(source="old_name.py", destination="new_name.py")
            Move a file into a subdirectory:
                move_file(source="utils.py", destination="src/utils/utils.py")

        Combine with:
            - ``grep_search`` → after moving/renaming a file, search for
              import statements or references that need updating.
            - ``edit_file`` → update imports and references in other files
              after the move.
            - ``find_files`` → locate files to reorganise, then move_file
              each one.
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
    def get_file_info(
        path: Annotated[str, Field(description="File or directory path, e.g. 'data/report.csv'")],
    ) -> str:
        """Return metadata for a file or directory as JSON.

        Use this tool to check file size, modification time, permissions, or
        whether a path is a file vs. directory.  Useful for verifying file
        state before or after operations.

        Fields returned: path, type, size (bytes), modified, created,
        permissions (octal), and num_entries (for directories).

        Args:
            path: File or directory path.

        Examples:
            Check a file's size and modification time:
                get_file_info(path="data/large_dataset.csv")
            Inspect a directory:
                get_file_info(path="src/")

        Combine with:
            - ``read_file`` → check file size with get_file_info first; if
              the file is very large, use read_file with a line range.
            - ``find_files`` → locate files by pattern, then get_file_info
              to inspect metadata of each match.
            - ``delete_file`` → check file age/size before deciding to clean
              up old or oversized files.
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
    def read_url(
        url: Annotated[str, Field(description="URL to fetch, e.g. 'https://example.com/api/docs'")],
    ) -> str:
        """Fetch a URL and return its text content (first 5 000 characters).

        Use this tool to retrieve API documentation, reference material,
        package READMEs, or any web content needed to inform coding decisions.
        HTML pages are automatically cleaned (script/style tags stripped,
        entities decoded) for readability.

        Args:
            url: The URL to fetch (http:// or https://).

        Examples:
            Fetch API documentation:
                read_url(url="https://docs.python.org/3/library/pathlib.html")
            Retrieve a raw file from GitHub:
                read_url(url="https://raw.githubusercontent.com/owner/repo/main/README.md")

        Combine with:
            - ``write_file`` / ``edit_file`` → fetch documentation or
              examples from the web, then incorporate them into your code.
            - ``run_python`` → fetch data from an API with read_url, then
              process it with run_python.
            - ``grep_search`` → read_url to understand a library's API, then
              grep_search to find how it's currently used in the codebase.
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
    def git_diff(
        path: Annotated[str, Field(description="File or directory to diff, e.g. 'src/main.py'")] = ".",
        cached: Annotated[bool, Field(description="Show staged changes instead of working tree")] = False,
    ) -> str:
        """Show the git diff for the working tree or staged changes.

        Use this tool to review uncommitted modifications, verify that edits
        look correct before committing, or inspect what has been staged.
        For complex git operations (e.g., diff between branches), use
        ``run_command`` with the full git command instead.

        Args:
            path: File or directory to diff (default: entire repo).
            cached: If True, show staged (cached) changes.

        Examples:
            See all unstaged changes:
                git_diff()
            See staged changes only:
                git_diff(cached=True)
            Diff a specific file:
                git_diff(path="src/main.py")

        Combine with:
            - ``edit_file`` → after editing, use git_diff to verify the
              change is correct before committing.
            - ``read_file`` → git_diff shows the change; read_file shows the
              full context around it.
            - ``run_command`` → use git_diff for quick checks, then
              run_command("git add . && git commit -m '...'") to commit.
            - ``git_log`` → use git_log to review history and git_diff to
              inspect current uncommitted work.
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
    def git_log(
        max_entries: Annotated[int, Field(description="Number of commits to show, e.g. 20")] = 10,
        path: Annotated[str, Field(description="Limit to commits touching this path, e.g. 'src/'")] = ".",
    ) -> str:
        """Show the recent git commit history.

        Use this tool to understand what changes have been made recently,
        identify when a bug was introduced, or review the project's
        development timeline.  Commits are shown in reverse chronological
        order with hash, date, author, and message.

        Args:
            max_entries: Number of commits to show (default 10).
            path: Limit history to commits that touch this path.

        Examples:
            Show last 10 commits:
                git_log()
            Show last 5 commits for a specific file:
                git_log(max_entries=5, path="src/main.py")
            Show extended history:
                git_log(max_entries=50)

        Combine with:
            - ``git_diff`` → use git_log to find when a change happened, then
              git_diff to see current uncommitted work.
            - ``run_command`` → for advanced history inspection, use
              run_command("git show <hash>") to see a specific commit's diff.
            - ``read_file`` → identify a commit that changed a file via
              git_log, then read_file to see its current state.
            - ``grep_search`` → use git_log to find a commit message
              mentioning a feature, then grep_search to find related code.
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
