"""Anthropic provider implementation."""

from __future__ import annotations

import asyncio
from typing import Any

import structlog

from src.llm.providers.base import BaseProvider, ProviderConfig, ProviderResponse

logger = structlog.get_logger()


class AnthropicProvider(BaseProvider):
    """Anthropic Claude API provider.

    Supports:
    - Claude Opus 4.5, Claude Sonnet 4
    - Claude 3.5 (Sonnet, Haiku)
    - Tool use
    - Streaming (not yet implemented)

    For ADRS: Uses Opus 4.5 for complex defense generation and analysis.
    """

    DEFAULT_MODEL = "claude-opus-4-5-20250514"

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._client: Any = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the Anthropic client."""
        try:
            from anthropic import AsyncAnthropic

            self._client = AsyncAnthropic(
                api_key=self.config.api_key,
                timeout=self.config.timeout_seconds,
            )
            self._log.info("Anthropic client initialized")
        except ImportError as e:
            self._log.error("anthropic package not installed")
            raise ImportError("anthropic package required: pip install anthropic") from e

    async def chat(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> ProviderResponse:
        """Send a chat request to Anthropic."""
        start_time = asyncio.get_event_loop().time()

        # Convert tools to Anthropic format
        anthropic_tools = None
        if tools:
            anthropic_tools = self._convert_tools(tools)

        # Build request kwargs
        kwargs: dict[str, Any] = {
            "model": self.config.model or self.DEFAULT_MODEL,
            "messages": messages,
            "max_tokens": max_tokens or self.config.max_tokens,
            "temperature": temperature,
        }

        if system:
            kwargs["system"] = system

        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        try:
            response = await self._client.messages.create(**kwargs)

            latency_ms = (asyncio.get_event_loop().time() - start_time) * 1000

            # Extract response
            content = None
            tool_calls = None

            for block in response.content:
                if block.type == "text":
                    content = block.text
                elif block.type == "tool_use":
                    if tool_calls is None:
                        tool_calls = []
                    tool_calls.append({
                        "id": block.id,
                        "name": block.name,
                        "arguments": block.input,
                    })

            # Track metrics
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            total_tokens = input_tokens + output_tokens
            self._track_request(latency_ms, total_tokens)

            return ProviderResponse(
                content=content,
                tool_calls=tool_calls,
                finish_reason=response.stop_reason,
                usage={
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": total_tokens,
                },
                model=response.model,
                latency_ms=latency_ms,
                raw_response=response,
            )

        except Exception as e:
            latency_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            self._log.error("Anthropic chat request failed", error=str(e))
            raise

    async def complete(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> ProviderResponse:
        """Send a completion request to Anthropic (using messages endpoint)."""
        return await self.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def health_check(self) -> bool:
        """Check if Anthropic is available."""
        try:
            # Simple request to check connectivity
            response = await self._client.messages.create(
                model=self.config.model or self.DEFAULT_MODEL,
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}],
            )
            return response is not None
        except Exception as e:
            self._log.warning("Anthropic health check failed", error=str(e))
            return False

    def _convert_tools(
        self,
        openai_tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Convert OpenAI tool format to Anthropic format."""
        anthropic_tools = []

        for tool in openai_tools:
            if tool.get("type") == "function":
                func = tool["function"]
                anthropic_tools.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
                })

        return anthropic_tools
