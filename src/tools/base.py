"""Base tool interface and definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ParameterType(str, Enum):
    """JSON Schema parameter types."""

    STRING = "string"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


class ToolParameter(BaseModel):
    """Definition of a tool parameter."""

    name: str
    type: ParameterType
    description: str
    required: bool = True
    default: Any | None = None
    enum: list[Any] | None = None
    items: dict[str, Any] | None = None  # For array types
    properties: dict[str, Any] | None = None  # For object types


class ToolDefinition(BaseModel):
    """OpenAI-compatible tool definition."""

    name: str
    description: str
    parameters: list[ToolParameter] = Field(default_factory=list)

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI function calling format."""
        properties = {}
        required = []

        for param in self.parameters:
            prop: dict[str, Any] = {
                "type": param.type.value,
                "description": param.description,
            }

            if param.enum:
                prop["enum"] = param.enum

            if param.type == ParameterType.ARRAY and param.items:
                prop["items"] = param.items

            if param.type == ParameterType.OBJECT and param.properties:
                prop["properties"] = param.properties

            if param.default is not None:
                prop["default"] = param.default

            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


@dataclass
class ToolExecutionContext:
    """Context passed to tool during execution."""

    tool_id: str
    call_number: int
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolExecutionResult:
    """Result from tool execution."""

    success: bool
    result: Any | None = None
    error: str | None = None
    execution_time_ms: float = 0.0
    cached: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseTool(ABC):
    """Abstract base class for all tools."""

    def __init__(self) -> None:
        self._call_count = 0
        self._total_execution_time = 0.0
        self._error_count = 0

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description."""
        ...

    @property
    @abstractmethod
    def parameters(self) -> list[ToolParameter]:
        """Tool parameters."""
        ...

    @abstractmethod
    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext | None = None,
    ) -> Any:
        """Execute the tool with given arguments."""
        ...

    def get_definition(self) -> ToolDefinition:
        """Get tool definition."""
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )

    def validate_arguments(self, arguments: dict[str, Any]) -> tuple[bool, str | None]:
        """Validate arguments against parameter definitions."""
        for param in self.parameters:
            if param.required and param.name not in arguments:
                return False, f"Missing required parameter: {param.name}"

            if param.name in arguments:
                value = arguments[param.name]

                # Type validation
                if param.type == ParameterType.STRING and not isinstance(value, str):
                    return False, f"Parameter {param.name} must be a string"
                elif param.type == ParameterType.NUMBER and not isinstance(value, (int, float)):
                    return False, f"Parameter {param.name} must be a number"
                elif param.type == ParameterType.INTEGER and not isinstance(value, int):
                    return False, f"Parameter {param.name} must be an integer"
                elif param.type == ParameterType.BOOLEAN and not isinstance(value, bool):
                    return False, f"Parameter {param.name} must be a boolean"
                elif param.type == ParameterType.ARRAY and not isinstance(value, list):
                    return False, f"Parameter {param.name} must be an array"
                elif param.type == ParameterType.OBJECT and not isinstance(value, dict):
                    return False, f"Parameter {param.name} must be an object"

                # Enum validation
                if param.enum and value not in param.enum:
                    return False, f"Parameter {param.name} must be one of: {param.enum}"

        return True, None

    @property
    def stats(self) -> dict[str, Any]:
        """Get tool execution statistics."""
        return {
            "call_count": self._call_count,
            "total_execution_time_ms": self._total_execution_time,
            "error_count": self._error_count,
            "average_execution_time_ms": (
                self._total_execution_time / self._call_count
                if self._call_count > 0
                else 0
            ),
        }

    def record_execution(
        self,
        execution_time_ms: float,
        success: bool,
    ) -> None:
        """Record execution statistics."""
        self._call_count += 1
        self._total_execution_time += execution_time_ms
        if not success:
            self._error_count += 1


class ToolWrapper(ABC):
    """Base class for tool wrappers (e.g., adversarial wrappers)."""

    def __init__(self, wrapped_tool: BaseTool) -> None:
        self.wrapped_tool = wrapped_tool

    @property
    def name(self) -> str:
        return self.wrapped_tool.name

    @property
    def description(self) -> str:
        return self.wrapped_tool.description

    @property
    def parameters(self) -> list[ToolParameter]:
        return self.wrapped_tool.parameters

    def get_definition(self) -> ToolDefinition:
        return self.wrapped_tool.get_definition()

    @abstractmethod
    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext | None = None,
    ) -> Any:
        """Execute with wrapper behavior."""
        ...
