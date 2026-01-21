"""OpenAI provider implementation."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import structlog

from src.llm.providers.base import BaseProvider, ProviderConfig, ProviderResponse

logger = structlog.get_logger()


class OpenAIProvider(BaseProvider):
    """OpenAI API provider.

    Supports:
    - GPT-4, GPT-4 Turbo, GPT-3.5 Turbo
    - Function/tool calling
    - Streaming (not yet implemented)
    """

    DEFAULT_MODEL = "gpt-4-turbo-preview"

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._client: Any = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the OpenAI client."""
        try:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout_seconds,
            )
            self._log.info("OpenAI client initialized")
        except ImportError as e:
            self._log.error("openai package not installed")
            raise ImportError("openai package required: pip install openai") from e

    async def chat(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> ProviderResponse:
        """Send a chat request to OpenAI."""
        start_time = asyncio.get_event_loop().time()

        # Build messages list
        api_messages = []
        if system:
            api_messages.append({"role": "system", "content": system})
        api_messages.extend(messages)

        # Build request kwargs
        kwargs: dict[str, Any] = {
            "model": self.config.model or self.DEFAULT_MODEL,
            "messages": api_messages,
            "temperature": temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
        }

        # Add tools if provided
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        try:
            response = await self._client.chat.completions.create(**kwargs)

            latency_ms = (asyncio.get_event_loop().time() - start_time) * 1000

            # Extract response
            choice = response.choices[0]
            content = choice.message.content
            tool_calls = None

            if choice.message.tool_calls:
                tool_calls = [
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": json.loads(tc.function.arguments),
                    }
                    for tc in choice.message.tool_calls
                ]

            # Track metrics
            tokens = response.usage.total_tokens if response.usage else 0
            self._track_request(latency_ms, tokens)

            return ProviderResponse(
                content=content,
                tool_calls=tool_calls,
                finish_reason=choice.finish_reason,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": tokens,
                },
                model=response.model,
                latency_ms=latency_ms,
                raw_response=response,
            )

        except Exception as e:
            latency_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            self._log.error("OpenAI chat request failed", error=str(e))
            raise

    async def complete(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> ProviderResponse:
        """Send a completion request to OpenAI (using chat endpoint)."""
        return await self.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def health_check(self) -> bool:
        """Check if OpenAI is available."""
        try:
            response = await self._client.models.list()
            return len(response.data) > 0
        except Exception as e:
            self._log.warning("OpenAI health check failed", error=str(e))
            return False
