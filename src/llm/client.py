"""LLM client abstraction for multiple providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

logger = structlog.get_logger()


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    VLLM = "vllm"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


@dataclass
class LLMConfig:
    """Configuration for LLM client."""

    provider: LLMProvider = LLMProvider.VLLM
    model: str = "mistralai/Mistral-Small-Instruct-2409"
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "not-needed"  # vLLM doesn't require API key
    max_tokens: int = 4096
    temperature: float = 0.7
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.0


@dataclass
class LLMResponse:
    """Response from LLM."""

    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    model: str = ""
    finish_reason: str = ""


class LLMClient:
    """Unified client for LLM providers.

    Primary provider: vLLM (local models)
    Fallback: OpenAI, Anthropic (optional)
    """

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()
        self._client: AsyncOpenAI | None = None
        self._log = logger.bind(
            component="llm_client",
            provider=self.config.provider.value,
            model=self.config.model,
        )

    async def _get_client(self) -> AsyncOpenAI:
        """Get or create the OpenAI-compatible client."""
        if self._client is None:
            self._client = AsyncOpenAI(
                base_url=self.config.base_url,
                api_key=self.config.api_key,
            )
        return self._client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def chat(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Send a chat completion request."""
        client = await self._get_client()

        # Prepare messages
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        # Prepare request kwargs
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": full_messages,
            "temperature": temperature or self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        self._log.debug("Sending chat request", num_messages=len(full_messages))

        try:
            response = await client.chat.completions.create(**kwargs)

            # Parse response
            choice = response.choices[0]
            message = choice.message

            # Extract tool calls if present
            tool_calls = []
            if message.tool_calls:
                for tc in message.tool_calls:
                    tool_calls.append(
                        {
                            "id": tc.id,
                            "name": tc.function.name,
                            "arguments": self._parse_arguments(tc.function.arguments),
                        }
                    )

            return LLMResponse(
                content=message.content or "",
                tool_calls=tool_calls,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                },
                model=response.model,
                finish_reason=choice.finish_reason or "",
            )

        except Exception as e:
            self._log.error("Chat request failed", error=str(e))
            raise

    def _parse_arguments(self, arguments: str) -> dict[str, Any]:
        """Parse tool call arguments from JSON string."""
        import json

        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            return {"raw": arguments}

    async def close(self) -> None:
        """Close the client."""
        if self._client:
            await self._client.close()
            self._client = None
