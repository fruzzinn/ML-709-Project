"""Honest (non-adversarial) tool implementations."""

from src.tools.base import BaseTool
from src.tools.honest.calculator import CalculatorTool
from src.tools.honest.code_executor import CodeExecutorTool
from src.tools.honest.file_reader import FileReaderTool
from src.tools.honest.web_search import WebSearchTool

__all__ = ["CalculatorTool", "WebSearchTool", "CodeExecutorTool", "FileReaderTool"]


def get_default_tools() -> list[BaseTool]:
    """Get list of default honest tools."""
    return [
        CalculatorTool(),
        WebSearchTool(),
        CodeExecutorTool(),
        FileReaderTool(),
    ]
