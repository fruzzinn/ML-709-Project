"""Claude Code Bridge - Uses Claude CLI authentication for API calls.

This provider bridges the ADRS system to Claude Code's CLI authentication,
allowing use of your Claude subscription without a separate API key.

Security Note: Uses asyncio.create_subprocess_exec (not shell) which is safe
against command injection as arguments are passed as a list, not a string.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from typing import Any

import structlog

from src.llm.providers.base import BaseProvider, ProviderConfig, ProviderResponse

logger = structlog.get_logger()


class ClaudeCodeBridgeProvider(BaseProvider):
    """Provider that uses Claude Code CLI for inference.

    Uses the `claude` CLI with -p flag to make requests using
    your existing Claude Code authentication (OAuth).

    Benefits:
    - No separate API key needed
    - Uses your Claude subscription credits
    - Supports Opus 4.5 and other models
    """

    DEFAULT_MODEL = "opus"  # Claude Code uses aliases

    def __init__(self, config: ProviderConfig | None = None) -> None:
        config = config or ProviderConfig(
            name="claude-code-bridge",
            model="opus",
        )
        super().__init__(config)
        self._claude_path = self._find_claude_cli()
        self._log.info("Claude Code Bridge initialized", claude_path=self._claude_path)

    def _find_claude_cli(self) -> str:
        """Find the claude CLI executable."""
        claude_path = shutil.which("claude")
        if not claude_path:
            raise RuntimeError(
                "Claude CLI not found. Please install Claude Code: https://claude.ai/code"
            )
        return claude_path

    async def chat(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        _tools: list[dict[str, Any]] | None = None,
        _temperature: float = 0.7,
        _max_tokens: int | None = None,
    ) -> ProviderResponse:
        """Send a chat request via Claude CLI.

        Uses asyncio.create_subprocess_exec for safe subprocess execution
        (arguments passed as list, no shell interpretation).
        """
        start_time = asyncio.get_event_loop().time()

        # Build the prompt from messages
        prompt = self._build_prompt(messages, system)

        # Build CLI arguments as a list (safe against injection)
        args = [
            "-p",  # Print mode (non-interactive)
            "--model",
            self.config.model or self.DEFAULT_MODEL,
            "--output-format",
            "json",
            "--dangerously-skip-permissions",  # We're just doing inference
            prompt,
        ]

        self._log.debug("Executing Claude CLI", model=self.config.model)

        try:
            # Run claude CLI asynchronously using exec (not shell)
            process = await asyncio.create_subprocess_exec(
                self._claude_path,
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.config.timeout_seconds or 120.0,
            )

            latency_ms = (asyncio.get_event_loop().time() - start_time) * 1000

            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                self._log.error("Claude CLI failed", error=error_msg)
                raise RuntimeError(f"Claude CLI error: {error_msg}")

            # Parse JSON response
            response_text = stdout.decode().strip()

            # Handle streaming JSON output (multiple JSON objects)
            content = self._parse_cli_output(response_text)

            self._track_request(latency_ms, len(content.split()))

            return ProviderResponse(
                content=content,
                tool_calls=None,  # CLI doesn't return tool calls in same format
                finish_reason="stop",
                usage={
                    "prompt_tokens": len(prompt.split()),  # Estimate
                    "completion_tokens": len(content.split()),  # Estimate
                    "total_tokens": len(prompt.split()) + len(content.split()),
                },
                model=self.config.model or self.DEFAULT_MODEL,
                latency_ms=latency_ms,
            )

        except TimeoutError as e:
            self._log.error("Claude CLI timed out", timeout=self.config.timeout_seconds)
            raise RuntimeError(f"Claude CLI timed out after {self.config.timeout_seconds}s") from e

        except Exception as e:
            self._log.error("Claude CLI request failed", error=str(e))
            raise

    async def complete(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> ProviderResponse:
        """Send a completion request via Claude CLI."""
        return await self.chat(
            messages=[{"role": "user", "content": prompt}],
            _temperature=temperature,
            _max_tokens=max_tokens,
        )

    async def health_check(self) -> bool:
        """Check if Claude CLI is available and authenticated."""
        try:
            process = await asyncio.create_subprocess_exec(
                self._claude_path,
                "-p",
                "--model",
                "haiku",  # Use cheapest model for health check
                "--output-format",
                "json",
                "Say 'ok' and nothing else.",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, _ = await asyncio.wait_for(
                process.communicate(),
                timeout=30.0,
            )

            return process.returncode == 0 and b"ok" in stdout.lower()

        except Exception as e:
            self._log.warning("Health check failed", error=str(e))
            return False

    def _build_prompt(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
    ) -> str:
        """Build a prompt string from messages."""
        parts = []

        if system:
            parts.append(f"<system>\n{system}\n</system>\n")

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                parts.append(f"<system>\n{content}\n</system>\n")
            elif role == "assistant":
                parts.append(f"<assistant>\n{content}\n</assistant>\n")
            else:
                parts.append(f"<user>\n{content}\n</user>\n")

        return "\n".join(parts)

    def _parse_cli_output(self, output: str) -> str:
        """Parse Claude CLI JSON output.

        Claude CLI with --output-format json returns:
        {"type":"result","subtype":"success","result":"...","usage":{...}}
        """
        content_parts = []

        for line in output.strip().split("\n"):
            if not line.strip():
                continue
            try:
                data = json.loads(line)

                # Primary format: result type with "result" field
                if data.get("type") == "result":
                    if "result" in data:
                        content_parts.append(str(data["result"]))
                        # Store usage info for later
                        self._last_usage = data.get("usage", {})
                        self._last_cost = data.get("total_cost_usd", 0)

                # Handle streaming assistant messages
                elif data.get("type") == "assistant":
                    if "message" in data:
                        msg = data["message"]
                        if isinstance(msg, dict) and "content" in msg:
                            for block in msg["content"]:
                                if block.get("type") == "text":
                                    content_parts.append(block.get("text", ""))
                        elif isinstance(msg, str):
                            content_parts.append(msg)

                # Handle streaming deltas
                elif data.get("type") == "content_block_delta":
                    delta = data.get("delta", {})
                    if delta.get("type") == "text_delta":
                        content_parts.append(delta.get("text", ""))

            except json.JSONDecodeError:
                # Plain text output
                content_parts.append(line)

        return "".join(content_parts) or output
