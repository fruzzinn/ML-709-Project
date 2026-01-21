"""Tests for tool system."""

import pytest

from src.tools.honest.calculator import CalculatorTool
from src.tools.honest.file_reader import FileReaderTool
from src.tools.honest.web_search import WebSearchTool
from src.tools.registry import ToolRegistry


class TestCalculatorTool:
    """Tests for CalculatorTool."""

    @pytest.fixture
    def calculator(self) -> CalculatorTool:
        return CalculatorTool()

    @pytest.mark.asyncio
    async def test_basic_arithmetic(self, calculator: CalculatorTool) -> None:
        """Test basic arithmetic operations."""
        result = await calculator.execute({"expression": "2 + 2"})
        assert result["result"] == 4

        result = await calculator.execute({"expression": "10 * 5"})
        assert result["result"] == 50

        result = await calculator.execute({"expression": "100 / 4"})
        assert result["result"] == 25

    @pytest.mark.asyncio
    async def test_math_functions(self, calculator: CalculatorTool) -> None:
        """Test math functions."""
        result = await calculator.execute({"expression": "sqrt(16)"})
        assert result["result"] == 4

        result = await calculator.execute({"expression": "pow(2, 3)"})
        assert result["result"] == 8

    @pytest.mark.asyncio
    async def test_constants(self, calculator: CalculatorTool) -> None:
        """Test mathematical constants."""
        result = await calculator.execute({"expression": "pi"})
        assert abs(result["result"] - 3.14159) < 0.001

    @pytest.mark.asyncio
    async def test_division_by_zero(self, calculator: CalculatorTool) -> None:
        """Test division by zero handling."""
        result = await calculator.execute({"expression": "1 / 0"})
        assert "error" in result
        assert "Division by zero" in result["error"]

    @pytest.mark.asyncio
    async def test_invalid_expression(self, calculator: CalculatorTool) -> None:
        """Test invalid expression handling."""
        result = await calculator.execute({"expression": "invalid"})
        assert "error" in result


class TestWebSearchTool:
    """Tests for WebSearchTool."""

    @pytest.fixture
    def search(self) -> WebSearchTool:
        return WebSearchTool(mock_mode=True)

    @pytest.mark.asyncio
    async def test_basic_search(self, search: WebSearchTool) -> None:
        """Test basic search functionality."""
        result = await search.execute({"query": "test query"})
        assert "results" in result
        assert "query" in result
        assert result["query"] == "test query"

    @pytest.mark.asyncio
    async def test_mock_results(self, search: WebSearchTool) -> None:
        """Test mock results configuration."""
        search.set_mock_results("python", [
            {"title": "Python.org", "snippet": "Official Python website", "url": "https://python.org"}
        ])

        result = await search.execute({"query": "python"})
        assert len(result["results"]) == 1
        assert result["results"][0]["title"] == "Python.org"

    @pytest.mark.asyncio
    async def test_num_results(self, search: WebSearchTool) -> None:
        """Test num_results parameter."""
        result = await search.execute({"query": "test", "num_results": 3})
        assert len(result["results"]) == 3


class TestFileReaderTool:
    """Tests for FileReaderTool."""

    @pytest.fixture
    def reader(self) -> FileReaderTool:
        return FileReaderTool(mock_mode=True)

    @pytest.mark.asyncio
    async def test_mock_read(self, reader: FileReaderTool) -> None:
        """Test mock file reading."""
        result = await reader.execute({"file_path": "test.txt"})
        assert "content" in result
        assert result["file_path"] == "test.txt"

    @pytest.mark.asyncio
    async def test_configured_mock(self, reader: FileReaderTool) -> None:
        """Test configured mock content."""
        reader.set_mock_file("config.yaml", "key: value")

        result = await reader.execute({"file_path": "config.yaml"})
        assert result["content"] == "key: value"


class TestToolRegistry:
    """Tests for ToolRegistry."""

    @pytest.fixture
    def registry(self) -> ToolRegistry:
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        registry.register(WebSearchTool(mock_mode=True))
        return registry

    def test_registration(self, registry: ToolRegistry) -> None:
        """Test tool registration."""
        assert "calculator" in registry
        assert "web_search" in registry
        assert registry.tool_count == 2

    def test_get_tool(self, registry: ToolRegistry) -> None:
        """Test getting a tool."""
        calc = registry.get("calculator")
        assert calc is not None
        assert calc.name == "calculator"

    def test_get_tool_definitions(self, registry: ToolRegistry) -> None:
        """Test getting OpenAI-compatible definitions."""
        definitions = registry.get_tool_definitions()
        assert len(definitions) == 2

        for defn in definitions:
            assert "type" in defn
            assert defn["type"] == "function"
            assert "function" in defn

    @pytest.mark.asyncio
    async def test_execute(self, registry: ToolRegistry) -> None:
        """Test tool execution through registry."""
        result = await registry.execute("calculator", {"expression": "1 + 1"})
        assert result["result"] == 2
