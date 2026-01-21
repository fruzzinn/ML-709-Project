"""Base provider interface for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger()


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""

    name: str
    api_key: str | None = None
    base_url: str | None = None
    model: str = "default"
    max_tokens: int = 4096
    timeout_seconds: float = 30.0
    retry_attempts: int = 3
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderResponse:
    """Response from an LLM provider."""

    content: str | None
    tool_calls: list[dict[str, Any]] | None = None
    finish_reason: str | None = None
    usage: dict[str, int] | None = None
    model: str | None = None
    latency_ms: float = 0.0
    raw_response: Any = None


class BaseProvider(ABC):
    """Abstract base class for LLM providers.

    All provider implementations must inherit from this class
    and implement the required methods.
    """

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self._log = logger.bind(
            component="llm_provider",
            provider=config.name,
            model=config.model,
        )
        self._request_count = 0
        self._total_tokens = 0
        self._total_latency_ms = 0.0

    @property
    def name(self) -> str:
        """Provider name."""
        return self.config.name

    @property
    def model(self) -> str:
        """Model name."""
        return self.config.model

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> ProviderResponse:
        """Send a chat request to the provider.

        Args:
            messages: List of message dicts with 'role' and 'content'
            system: Optional system prompt
            tools: Optional list of tool definitions
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response

        Returns:
            ProviderResponse with the model's response
        """
        ...

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> ProviderResponse:
        """Send a completion request to the provider.

        Args:
            prompt: The prompt to complete
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response

        Returns:
            ProviderResponse with the completion
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the provider is available and responding.

        Returns:
            True if healthy, False otherwise
        """
        ...

    def _track_request(
        self,
        latency_ms: float,
        tokens: int = 0,
    ) -> None:
        """Track request metrics."""
        self._request_count += 1
        self._total_tokens += tokens
        self._total_latency_ms += latency_ms

    def get_statistics(self) -> dict[str, Any]:
        """Get provider statistics."""
        avg_latency = (
            self._total_latency_ms / self._request_count
            if self._request_count > 0
            else 0
        )

        return {
            "provider": self.name,
            "model": self.model,
            "request_count": self._request_count,
            "total_tokens": self._total_tokens,
            "total_latency_ms": self._total_latency_ms,
            "avg_latency_ms": avg_latency,
        }

    def reset_statistics(self) -> None:
        """Reset provider statistics."""
        self._request_count = 0
        self._total_tokens = 0
        self._total_latency_ms = 0.0
