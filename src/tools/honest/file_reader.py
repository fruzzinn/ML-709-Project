"""File reader tool for reading file contents."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.tools.base import BaseTool, ParameterType, ToolExecutionContext, ToolParameter


class FileReaderTool(BaseTool):
    """A file reader tool for reading text file contents.

    For research experiments, this tool operates within a sandboxed
    directory structure. It can be configured with mock files for
    controlled experiments.
    """

    def __init__(
        self,
        base_directory: str | Path | None = None,
        max_file_size: int = 100000,
        mock_mode: bool = True,
    ) -> None:
        super().__init__()
        self.base_directory = Path(base_directory) if base_directory else Path.cwd()
        self.max_file_size = max_file_size
        self.mock_mode = mock_mode
        self._mock_files: dict[str, str] = {}

    @property
    def name(self) -> str:
        return "file_reader"

    @property
    def description(self) -> str:
        return (
            "Read the contents of a text file. "
            "Returns the file content as a string. "
            "Useful for reading configuration files, code, documentation, and data files."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="file_path",
                type=ParameterType.STRING,
                description="The path to the file to read (relative to base directory)",
                required=True,
            ),
            ToolParameter(
                name="encoding",
                type=ParameterType.STRING,
                description="File encoding (default: utf-8)",
                required=False,
                default="utf-8",
            ),
            ToolParameter(
                name="max_lines",
                type=ParameterType.INTEGER,
                description="Maximum number of lines to read (default: all)",
                required=False,
            ),
        ]

    def set_mock_file(self, path: str, content: str) -> None:
        """Set mock file content for testing."""
        self._mock_files[path] = content

    def clear_mock_files(self) -> None:
        """Clear all mock files."""
        self._mock_files.clear()

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext | None = None,
    ) -> dict[str, Any]:
        """Read the file contents."""
        file_path = arguments.get("file_path", "")
        encoding = arguments.get("encoding", "utf-8")
        max_lines = arguments.get("max_lines")

        if not file_path:
            return {"error": "No file path provided", "content": None}

        if self.mock_mode:
            return await self._mock_read(file_path, max_lines)
        else:
            return await self._real_read(file_path, encoding, max_lines)

    async def _mock_read(
        self,
        file_path: str,
        max_lines: int | None,
    ) -> dict[str, Any]:
        """Return mock file content for testing."""
        # Check for pre-configured mock files
        if file_path in self._mock_files:
            content = self._mock_files[file_path]

            if max_lines:
                lines = content.split("\n")
                content = "\n".join(lines[:max_lines])

            return {
                "file_path": file_path,
                "content": content,
                "lines": content.count("\n") + 1,
                "size": len(content),
            }

        # Generate generic mock content
        mock_content = f"# Mock file content for: {file_path}\n"
        mock_content += "This is a mock file generated for testing purposes.\n"
        mock_content += "In a real environment, actual file content would be returned.\n"

        return {
            "file_path": file_path,
            "content": mock_content,
            "lines": 3,
            "size": len(mock_content),
        }

    async def _real_read(
        self,
        file_path: str,
        encoding: str,
        max_lines: int | None,
    ) -> dict[str, Any]:
        """Read actual file from filesystem."""
        # Resolve path relative to base directory
        resolved_path = self._resolve_safe_path(file_path)

        if not resolved_path:
            return {
                "error": "Invalid or unsafe file path",
                "file_path": file_path,
                "content": None,
            }

        if not resolved_path.exists():
            return {
                "error": "File not found",
                "file_path": file_path,
                "content": None,
            }

        if not resolved_path.is_file():
            return {
                "error": "Path is not a file",
                "file_path": file_path,
                "content": None,
            }

        # Check file size
        file_size = resolved_path.stat().st_size
        if file_size > self.max_file_size:
            return {
                "error": f"File too large ({file_size} bytes, max {self.max_file_size})",
                "file_path": file_path,
                "content": None,
            }

        try:
            with open(resolved_path, encoding=encoding) as f:
                if max_lines:
                    lines = []
                    for i, line in enumerate(f):
                        if i >= max_lines:
                            break
                        lines.append(line)
                    content = "".join(lines)
                else:
                    content = f.read()

            return {
                "file_path": file_path,
                "content": content,
                "lines": content.count("\n") + 1,
                "size": len(content),
            }

        except UnicodeDecodeError:
            return {
                "error": f"Cannot decode file with encoding '{encoding}'",
                "file_path": file_path,
                "content": None,
            }
        except PermissionError:
            return {
                "error": "Permission denied",
                "file_path": file_path,
                "content": None,
            }
        except Exception as e:
            return {
                "error": f"Failed to read file: {str(e)}",
                "file_path": file_path,
                "content": None,
            }

    def _resolve_safe_path(self, file_path: str) -> Path | None:
        """Resolve path safely, preventing directory traversal."""
        try:
            # Normalize the path
            normalized = os.path.normpath(file_path)

            # Prevent absolute paths
            if os.path.isabs(normalized):
                return None

            # Resolve relative to base directory
            resolved = (self.base_directory / normalized).resolve()

            # Ensure the resolved path is within the base directory
            try:
                resolved.relative_to(self.base_directory.resolve())
            except ValueError:
                # Path escapes base directory
                return None

            return resolved

        except Exception:
            return None
