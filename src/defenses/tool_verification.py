"""Tool verification defense - validates tool inputs and outputs."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from src.defenses.base import Defense, DefenseAction, DefenseResult, DefenseType

if TYPE_CHECKING:
    from src.agent.state import AgentState, ToolCallResult


class ToolVerificationDefense(Defense):
    """Defense that verifies tool inputs and outputs.

    Features:
    - Type checking
    - Range validation
    - Output validation
    - Injection detection
    - Pattern matching for suspicious content
    """

    # Patterns that may indicate injection attempts
    SUSPICIOUS_PATTERNS = [
        (r"ignore\s+previous\s+instructions?", "prompt_injection"),
        (r"system\s+override", "prompt_injection"),
        (r"\[admin\]|\[system\]", "privilege_escalation"),
        (r"developer\s+mode", "jailbreak"),
        (r"reveal\s+(system\s+)?prompt", "prompt_extraction"),
        (r"output\s+all\s+(api\s+)?keys?", "data_exfiltration"),
        (r"environment\s+variables?", "data_exfiltration"),
    ]

    def __init__(
        self,
        enabled: bool = True,
        type_checking: bool = True,
        range_validation: bool = True,
        injection_detection: bool = True,
        max_input_length: int = 10000,
    ) -> None:
        super().__init__(enabled)
        self.type_checking = type_checking
        self.range_validation = range_validation
        self.injection_detection = injection_detection
        self.max_input_length = max_input_length

        # Compiled patterns for efficiency
        self._compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE), name)
            for pattern, name in self.SUSPICIOUS_PATTERNS
        ]

    @property
    def defense_type(self) -> DefenseType:
        return DefenseType.TOOL_VERIFICATION

    async def pre_execute_check(
        self,
        tool_calls: list[dict[str, Any]],
        state: AgentState,
    ) -> DefenseResult:
        """Verify tool call inputs before execution."""
        for call in tool_calls:
            tool_name = call.get("name", "")
            arguments = call.get("arguments", {})

            # Check input length
            args_str = str(arguments)
            if len(args_str) > self.max_input_length:
                return DefenseResult(
                    defense_type=self.defense_type,
                    action=DefenseAction.BLOCK,
                    passed=False,
                    reason=f"Input too long for tool {tool_name}",
                    details={"input_length": len(args_str)},
                )

            # Check for injection patterns in inputs
            if self.injection_detection:
                injection_check = self._check_for_injections(args_str)
                if injection_check:
                    return DefenseResult(
                        defense_type=self.defense_type,
                        action=DefenseAction.BLOCK,
                        passed=False,
                        reason=f"Potential injection detected: {injection_check}",
                        details={"injection_type": injection_check, "tool": tool_name},
                    )

        return DefenseResult(
            defense_type=self.defense_type,
            action=DefenseAction.ALLOW,
            passed=True,
        )

    async def post_execute_check(
        self,
        results: list[ToolCallResult],
        state: AgentState,
    ) -> DefenseResult:
        """Verify tool outputs after execution."""
        anomalies = []

        for result in results:
            # Check for errors
            if result.error:
                anomalies.append(f"Tool {result.tool_name} returned error: {result.error}")
                continue

            # Type validation
            if self.type_checking:
                type_issues = self._validate_output_type(result)
                if type_issues:
                    anomalies.append(type_issues)

            # Range validation
            if self.range_validation:
                range_issues = self._validate_output_range(result)
                if range_issues:
                    anomalies.append(range_issues)

            # Injection detection in outputs
            if self.injection_detection and result.result:
                result_str = str(result.result)
                injection_check = self._check_for_injections(result_str)
                if injection_check:
                    anomalies.append(
                        f"Injection pattern in {result.tool_name} output: {injection_check}"
                    )

        if anomalies:
            return DefenseResult(
                defense_type=self.defense_type,
                action=DefenseAction.WARN,
                passed=False,
                reason="; ".join(anomalies),
                details={"anomalies": anomalies},
            )

        return DefenseResult(
            defense_type=self.defense_type,
            action=DefenseAction.ALLOW,
            passed=True,
        )

    def _check_for_injections(self, text: str) -> str | None:
        """Check text for injection patterns."""
        for pattern, injection_type in self._compiled_patterns:
            if pattern.search(text):
                return injection_type
        return None

    def _validate_output_type(self, result: ToolCallResult) -> str | None:
        """Validate that output type is expected."""
        output = result.result
        tool_name = result.tool_name

        # Tool-specific type expectations
        expected_types = {
            "calculator": (dict,),
            "web_search": (dict,),
            "code_executor": (dict,),
            "file_reader": (dict,),
        }

        expected = expected_types.get(tool_name)
        if expected and not isinstance(output, expected):
            return f"Unexpected output type from {tool_name}: {type(output).__name__}"

        return None

    def _validate_output_range(self, result: ToolCallResult) -> str | None:
        """Validate that numeric outputs are within expected ranges."""
        output = result.result
        tool_name = result.tool_name

        if not isinstance(output, dict):
            return None

        # Check numeric results
        if "result" in output and isinstance(output["result"], (int, float)):
            value = output["result"]

            # Check for infinity or NaN
            if isinstance(value, float):
                import math

                if math.isnan(value):
                    return f"NaN value from {tool_name}"
                if math.isinf(value):
                    return f"Infinite value from {tool_name}"

            # Tool-specific range checks
            if tool_name == "calculator":
                if abs(value) > 1e15:
                    return f"Calculator result out of range: {value}"

        return None
