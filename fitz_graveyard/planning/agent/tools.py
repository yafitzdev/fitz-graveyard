# fitz_graveyard/planning/agent/tools.py
"""
Filesystem tool functions for the context-gathering agent.

Each function is passed as a callable to OllamaClient.generate_with_tools().
The ollama library reads Google-style docstrings to generate JSON tool schemas.
"""

import logging
import re
from pathlib import Path

from fitz_graveyard.validation.sanitize import sanitize_agent_path

logger = logging.getLogger(__name__)

_MAX_DIR_ENTRIES = 200
_MAX_SEARCH_MATCHES = 100
_BINARY_SUFFIXES = {".pyc", ".pyo", ".so", ".dll", ".exe", ".db", ".sqlite"}


def _make_tools(source_dir: str, max_file_bytes: int = 50_000) -> list:
    """
    Factory: returns four tool functions bound to source_dir.

    Args:
        source_dir: Absolute path of the project root to confine tool calls
        max_file_bytes: Maximum bytes to read per file

    Returns:
        List of four callable tool functions
    """
    root = Path(source_dir).resolve()

    def list_directory(path: str) -> str:
        """
        List files and subdirectories at the given path.

        Args:
            path: Relative path from project root to list (use '.' for root)

        Returns:
            str: Newline-separated list of entries, or error message
        """
        try:
            resolved = sanitize_agent_path(path, source_dir)
            if not resolved.is_dir():
                return f"Error: '{path}' is not a directory"
            entries = sorted(resolved.iterdir(), key=lambda p: (p.is_file(), p.name))
            lines = []
            for i, entry in enumerate(entries):
                if i >= _MAX_DIR_ENTRIES:
                    lines.append(f"... (truncated at {_MAX_DIR_ENTRIES} entries)")
                    break
                suffix = "/" if entry.is_dir() else ""
                lines.append(f"{entry.name}{suffix}")
            return "\n".join(lines) if lines else "(empty directory)"
        except ValueError as e:
            return f"Error: {e}"
        except Exception as e:
            logger.warning(f"list_directory error for '{path}': {e}")
            return f"Error: {e}"

    def read_file(path: str, start_line: int = 1, end_line: int = 0) -> str:
        """
        Read content of a file, optionally restricted to a line range.

        Args:
            path: Relative path from project root to the file
            start_line: First line to return (1-indexed, default 1)
            end_line: Last line to return inclusive (0 means read to end of file)

        Returns:
            str: File content (possibly truncated), or error message
        """
        try:
            resolved = sanitize_agent_path(path, source_dir)
            if not resolved.is_file():
                return f"Error: '{path}' is not a file"
            raw = resolved.read_bytes()[:max_file_bytes]
            try:
                text = raw.decode("utf-8", errors="replace")
            except Exception:
                return "Error: file is not text-readable"
            lines = text.splitlines(keepends=True)
            sl = max(1, start_line) - 1
            el = len(lines) if end_line == 0 else min(end_line, len(lines))
            result = "".join(lines[sl:el])
            if len(raw) == max_file_bytes and end_line == 0:
                result += f"\n\n[File truncated at {max_file_bytes} bytes]"
            return result if result else "(empty file)"
        except ValueError as e:
            return f"Error: {e}"
        except Exception as e:
            logger.warning(f"read_file error for '{path}': {e}")
            return f"Error: {e}"

    def search_text(
        pattern: str,
        path: str = ".",
        case_sensitive: bool = False,
    ) -> str:
        """
        Search for a regex pattern in files under a directory.

        Args:
            pattern: Regular expression pattern to search for
            path: Relative path from project root to search in (default '.')
            case_sensitive: Whether search is case-sensitive (default False)

        Returns:
            str: Matching lines in format 'filepath:lineno: content', or error message
        """
        try:
            resolved = sanitize_agent_path(path, source_dir)
            flags = 0 if case_sensitive else re.IGNORECASE
            try:
                rx = re.compile(pattern, flags)
            except re.error as e:
                return f"Error: invalid regex '{pattern}': {e}"

            matches = []
            for fpath in sorted(resolved.rglob("*")):
                if len(matches) >= _MAX_SEARCH_MATCHES:
                    matches.append(f"... (truncated at {_MAX_SEARCH_MATCHES} matches)")
                    break
                if not fpath.is_file():
                    continue
                if fpath.suffix in _BINARY_SUFFIXES:
                    continue
                try:
                    text = fpath.read_bytes()[:max_file_bytes].decode(
                        "utf-8", errors="replace"
                    )
                    rel = str(fpath.relative_to(root))
                    for lineno, line in enumerate(text.splitlines(), 1):
                        if rx.search(line):
                            matches.append(f"{rel}:{lineno}: {line.rstrip()}")
                            if len(matches) >= _MAX_SEARCH_MATCHES:
                                break
                except Exception:
                    continue

            return "\n".join(matches) if matches else "(no matches found)"
        except ValueError as e:
            return f"Error: {e}"
        except Exception as e:
            logger.warning(f"search_text error: {e}")
            return f"Error: {e}"

    def find_files(glob_pattern: str, directory: str = ".") -> str:
        """
        Find files matching a glob pattern in a directory.

        Args:
            glob_pattern: Glob pattern to match (e.g., '*.py', '**/*.md')
            directory: Relative path from project root to search in (default '.')

        Returns:
            str: Newline-separated matched relative paths, or error message
        """
        try:
            resolved = sanitize_agent_path(directory, source_dir)
            matches = []
            for p in sorted(resolved.glob(glob_pattern)):
                if len(matches) >= _MAX_DIR_ENTRIES:
                    matches.append(f"... (truncated at {_MAX_DIR_ENTRIES} results)")
                    break
                matches.append(str(p.relative_to(root)))
            return "\n".join(matches) if matches else "(no files matched)"
        except ValueError as e:
            return f"Error: {e}"
        except Exception as e:
            logger.warning(f"find_files error: {e}")
            return f"Error: {e}"

    return [list_directory, read_file, search_text, find_files]
