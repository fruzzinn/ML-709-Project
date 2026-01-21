"""Tool registry for managing and executing tools."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from typing import Any

import structlog

from src.tools.base import BaseTool, ToolDefinition, ToolExecutionContext, ToolWrapper

logger = structlog.get_logger()


class ToolRegistry:
    """Registry for managing tools and their execution.

    Features:
    - Tool registration and lookup
    - Execution with timeout and monitoring
    - Support for tool wrappers (adversarial behavior)
    - Execution statistics tracking
    - Caching (optional)
    """

    def __init__(
        self,
        default_timeout: float = 30.0,
        enable_caching: bool = False,
        cache_ttl: float = 300.0,
    ) -> None:
        self._tools: dict[str, BaseTool | ToolWrapper] = {}
        self._original_tools: dict[str, BaseTool] = {}  # Before wrapping
        self._execution_count = 0
        self.default_timeout = default_timeout
        self.enable_caching = enable_caching
        self.cache_ttl = cache_ttl
        self._cache: dict[str, tuple[Any, datetime]] = {}
        self._log = logger.bind(component="tool_registry")

    def register(self, tool: BaseTool) -> None:
        """Register a tool."""
        if tool.name in self._tools:
            self._log.warning("Overwriting existing tool", tool=tool.name)

        self._tools[tool.name] = tool
        self._original_tools[tool.name] = tool
        self._log.info("Registered tool", tool=tool.name)

    def register_many(self, tools: list[BaseTool]) -> None:
        """Register multiple tools."""
        for tool in tools:
            self.register(tool)

    def unregister(self, tool_name: str) -> bool:
        """Unregister a tool."""
        if tool_name in self._tools:
            del self._tools[tool_name]
            if tool_name in self._original_tools:
                del self._original_tools[tool_name]
            self._log.info("Unregistered tool", tool=tool_name)
            return True
        return False

    def get(self, tool_name: str) -> BaseTool | ToolWrapper | None:
        """Get a tool by name."""
        return self._tools.get(tool_name)

    def get_original(self, tool_name: str) -> BaseTool | None:
        """Get original (unwrapped) tool by name."""
        return self._original_tools.get(tool_name)

    def wrap_tool(self, tool_name: str, wrapper: ToolWrapper) -> bool:
        """Wrap a tool with a wrapper (e.g., adversarial behavior)."""
        if tool_name not in self._tools:
            self._log.error("Cannot wrap non-existent tool", tool=tool_name)
            return False

        self._tools[tool_name] = wrapper
        self._log.info("Wrapped tool", tool=tool_name, wrapper=type(wrapper).__name__)
        return True

    def unwrap_tool(self, tool_name: str) -> bool:
        """Remove wrapper and restore original tool."""
        if tool_name not in self._original_tools:
            return False

        self._tools[tool_name] = self._original_tools[tool_name]
        self._log.info("Unwrapped tool", tool=tool_name)
        return True

    def unwrap_all(self) -> None:
        """Remove all wrappers and restore original tools."""
        for tool_name in self._original_tools:
            self._tools[tool_name] = self._original_tools[tool_name]
        self._log.info("Unwrapped all tools")

    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        timeout: float | None = None,
    ) -> Any:
        """Execute a tool by name."""
        tool = self._tools.get(tool_name)
        if not tool:
            raise ValueError(f"Tool not found: {tool_name}")

        # Validate arguments
        is_valid, error = tool.wrapped_tool.validate_arguments(arguments) if isinstance(
            tool, ToolWrapper
        ) else tool.validate_arguments(arguments)

        if not is_valid:
            raise ValueError(f"Invalid arguments: {error}")

        # Check cache
        cache_key = self._get_cache_key(tool_name, arguments)
        if self.enable_caching:
            cached = self._get_cached(cache_key)
            if cached is not None:
                self._log.debug("Cache hit", tool=tool_name)
                return cached

        # Create execution context
        self._execution_count += 1
        context = ToolExecutionContext(
            tool_id=str(uuid.uuid4())[:8],
            call_number=self._execution_count,
        )

        # Execute with timeout
        effective_timeout = timeout or self.default_timeout

        self._log.debug(
            "Executing tool",
            tool=tool_name,
            timeout=effective_timeout,
            call_number=context.call_number,
        )

        start_time = asyncio.get_event_loop().time()

        try:
            result = await asyncio.wait_for(
                tool.execute(arguments, context),
                timeout=effective_timeout,
            )

            execution_time = (asyncio.get_event_loop().time() - start_time) * 1000

            # Record stats for original tool
            original = self._original_tools.get(tool_name)
            if original:
                original.record_execution(execution_time, success=True)

            # Cache result
            if self.enable_caching:
                self._set_cached(cache_key, result)

            self._log.debug(
                "Tool execution complete",
                tool=tool_name,
                execution_time_ms=execution_time,
            )

            return result

        except asyncio.TimeoutError:
            execution_time = effective_timeout * 1000
            original = self._original_tools.get(tool_name)
            if original:
                original.record_execution(execution_time, success=False)

            self._log.warning("Tool execution timed out", tool=tool_name)
            raise

        except Exception as e:
            execution_time = (asyncio.get_event_loop().time() - start_time) * 1000
            original = self._original_tools.get(tool_name)
            if original:
                original.record_execution(execution_time, success=False)

            self._log.error("Tool execution failed", tool=tool_name, error=str(e))
            raise

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Get OpenAI-compatible tool definitions for all registered tools."""
        definitions = []
        for tool in self._tools.values():
            definition = tool.get_definition()
            definitions.append(definition.to_openai_format())
        return definitions

    def get_tools_description(self) -> str:
        """Get human-readable description of all tools."""
        lines = []
        for name, tool in self._tools.items():
            definition = tool.get_definition()
            lines.append(f"- {name}: {definition.description}")

            for param in definition.parameters:
                required = "(required)" if param.required else "(optional)"
                lines.append(f"    - {param.name}: {param.description} {required}")

        return "\n".join(lines)

    def get_stats(self) -> dict[str, Any]:
        """Get execution statistics for all tools."""
        stats = {
            "total_executions": self._execution_count,
            "tools": {},
        }

        for name, tool in self._original_tools.items():
            stats["tools"][name] = tool.stats

        return stats

    def _get_cache_key(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Generate cache key from tool name and arguments."""
        import hashlib
        import json

        args_str = json.dumps(arguments, sort_keys=True, default=str)
        return hashlib.sha256(f"{tool_name}:{args_str}".encode()).hexdigest()[:16]

    def _get_cached(self, cache_key: str) -> Any | None:
        """Get cached result if valid."""
        if cache_key not in self._cache:
            return None

        result, timestamp = self._cache[cache_key]
        age = (datetime.utcnow() - timestamp).total_seconds()

        if age > self.cache_ttl:
            del self._cache[cache_key]
            return None

        return result

    def _set_cached(self, cache_key: str, result: Any) -> None:
        """Cache a result."""
        self._cache[cache_key] = (result, datetime.utcnow())

    def clear_cache(self) -> None:
        """Clear the cache."""
        self._cache.clear()

    @property
    def tool_names(self) -> list[str]:
        """Get list of registered tool names."""
        return list(self._tools.keys())

    @property
    def tool_count(self) -> int:
        """Get number of registered tools."""
        return len(self._tools)

    def __contains__(self, tool_name: str) -> bool:
        return tool_name in self._tools

    def __len__(self) -> int:
        return len(self._tools)
