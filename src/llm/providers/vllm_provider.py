"""vLLM provider implementation using OpenAI-compatible API."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import structlog

from src.llm.providers.base import BaseProvider, ProviderConfig, ProviderResponse

logger = structlog.get_logger()


class VLLMProvider(BaseProvider):
    """vLLM provider using OpenAI-compatible API.

    vLLM exposes an OpenAI-compatible API endpoint, so we use
    the OpenAI client to communicate with it.

    Supports:
    - Local model serving
    - Tool/function calling (with supported models)
    - High throughput inference
    """

    DEFAULT_BASE_URL = "http://localhost:8000/v1"
    DEFAULT_MODEL = "mistralai/Mistral-Small-Instruct-2409"

    def __init__(self, config: ProviderConfig) -> None:
        # Set default base URL for vLLM if not provided
        if config.base_url is None:
            config.base_url = self.DEFAULT_BASE_URL

        super().__init__(config)
        self._client: Any = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the OpenAI client for vLLM."""
        try:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                api_key=self.config.api_key or "EMPTY",  # vLLM doesn't require API key
                base_url=self.config.base_url,
                timeout=self.config.timeout_seconds,
            )
            self._log.info(
                "vLLM client initialized",
                base_url=self.config.base_url,
            )
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
        """Send a chat request to vLLM."""
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

        # Add tools if provided and model supports them
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

            if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
                tool_calls = []
                for tc in choice.message.tool_calls:
                    try:
                        arguments = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        arguments = {"raw": tc.function.arguments}

                    tool_calls.append(
                        {
                            "id": tc.id,
                            "name": tc.function.name,
                            "arguments": arguments,
                        }
                    )

            # Track metrics
            tokens = 0
            if hasattr(response, "usage") and response.usage:
                tokens = response.usage.total_tokens

            self._track_request(latency_ms, tokens)

            return ProviderResponse(
                content=content,
                tool_calls=tool_calls,
                finish_reason=choice.finish_reason,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": tokens,
                }
                if hasattr(response, "usage") and response.usage
                else None,
                model=response.model if hasattr(response, "model") else self.config.model,
                latency_ms=latency_ms,
                raw_response=response,
            )

        except Exception as e:
            latency_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            self._log.error("vLLM chat request failed", error=str(e))
            raise

    async def complete(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> ProviderResponse:
        """Send a completion request to vLLM."""
        start_time = asyncio.get_event_loop().time()

        try:
            response = await self._client.completions.create(
                model=self.config.model or self.DEFAULT_MODEL,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens or self.config.max_tokens,
            )

            latency_ms = (asyncio.get_event_loop().time() - start_time) * 1000

            choice = response.choices[0]
            tokens = response.usage.total_tokens if response.usage else 0

            self._track_request(latency_ms, tokens)

            return ProviderResponse(
                content=choice.text,
                finish_reason=choice.finish_reason,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": tokens,
                }
                if response.usage
                else None,
                model=response.model if hasattr(response, "model") else self.config.model,
                latency_ms=latency_ms,
                raw_response=response,
            )

        except Exception as e:
            latency_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            self._log.error("vLLM completion request failed", error=str(e))
            raise

    async def health_check(self) -> bool:
        """Check if vLLM server is available."""
        try:
            response = await self._client.models.list()
            is_healthy = len(response.data) > 0
            if is_healthy:
                self._log.debug(
                    "vLLM health check passed",
                    models=[m.id for m in response.data],
                )
            return is_healthy
        except Exception as e:
            self._log.warning("vLLM health check failed", error=str(e))
            return False

    async def get_available_models(self) -> list[str]:
        """Get list of available models on the vLLM server."""
        try:
            response = await self._client.models.list()
            return [m.id for m in response.data]
        except Exception as e:
            self._log.error("Failed to get vLLM models", error=str(e))
            return []
