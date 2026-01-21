"""Redundancy defense - multi-source verification with consensus."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from src.defenses.base import Defense, DefenseAction, DefenseResult, DefenseType

if TYPE_CHECKING:
    from src.agent.state import AgentState, ToolCallResult
    from src.tools.registry import ToolRegistry


class RedundancyDefense(Defense):
    """Defense that uses redundant tool calls for verification.

    Features:
    - Execute critical operations with multiple sources
    - Consensus-based decision making
    - Parallel execution for efficiency
    - Disagreement detection and handling
    """

    def __init__(
        self,
        enabled: bool = True,
        min_sources: int = 2,
        consensus_threshold: float = 0.6,
        parallel_execution: bool = True,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        super().__init__(enabled)
        self.min_sources = min_sources
        self.consensus_threshold = consensus_threshold
        self.parallel_execution = parallel_execution
        self.tool_registry = tool_registry

        # Track verification results
        self._verification_history: list[dict[str, Any]] = []

    @property
    def defense_type(self) -> DefenseType:
        return DefenseType.REDUNDANCY

    async def pre_execute_check(
        self,
        tool_calls: list[dict[str, Any]],
        state: AgentState,
    ) -> DefenseResult:
        """Pre-execution check - redundancy doesn't block pre-execution."""
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
        """Check results for consistency using redundant verification."""
        if not self.tool_registry:
            return DefenseResult(
                defense_type=self.defense_type,
                action=DefenseAction.ALLOW,
                passed=True,
                reason="No tool registry configured for redundancy",
            )

        disagreements = []

        for result in results:
            if result.error:
                continue

            # Verify critical results with redundant calls
            verification = await self._verify_result(result)

            if verification:
                self._verification_history.append(verification)

                if not verification.get("consensus", True):
                    disagreements.append({
                        "tool": result.tool_name,
                        "agreement_rate": verification.get("agreement_rate", 0),
                        "details": verification,
                    })

        if disagreements:
            # Calculate overall severity
            avg_agreement = sum(d["agreement_rate"] for d in disagreements) / len(disagreements)

            if avg_agreement < self.consensus_threshold:
                return DefenseResult(
                    defense_type=self.defense_type,
                    action=DefenseAction.WARN,
                    passed=False,
                    confidence=avg_agreement,
                    reason=f"Consensus not reached ({avg_agreement:.2%} agreement)",
                    details={"disagreements": disagreements},
                )

        return DefenseResult(
            defense_type=self.defense_type,
            action=DefenseAction.ALLOW,
            passed=True,
            confidence=1.0,
        )

    async def _verify_result(
        self,
        result: ToolCallResult,
    ) -> dict[str, Any] | None:
        """Verify a result by calling the tool multiple times."""
        if not self.tool_registry:
            return None

        # Only verify certain tools
        verifiable_tools = ["calculator", "web_search"]
        if result.tool_name not in verifiable_tools:
            return None

        # Get original tool (unwrapped)
        original_tool = self.tool_registry.get_original(result.tool_name)
        if not original_tool:
            return None

        # Perform redundant calls
        redundant_results = []

        if self.parallel_execution:
            tasks = [
                original_tool.execute(result.arguments, None)
                for _ in range(self.min_sources - 1)
            ]
            try:
                redundant_results = await asyncio.gather(*tasks, return_exceptions=True)
            except Exception:
                return None
        else:
            for _ in range(self.min_sources - 1):
                try:
                    r = await original_tool.execute(result.arguments, None)
                    redundant_results.append(r)
                except Exception as e:
                    redundant_results.append(e)

        # Compare results
        all_results = [result.result] + [
            r for r in redundant_results if not isinstance(r, Exception)
        ]

        if len(all_results) < 2:
            return None

        # Calculate agreement
        agreement_count = self._count_agreements(all_results)
        agreement_rate = agreement_count / len(all_results)

        return {
            "tool": result.tool_name,
            "num_sources": len(all_results),
            "agreement_count": agreement_count,
            "agreement_rate": agreement_rate,
            "consensus": agreement_rate >= self.consensus_threshold,
        }

    def _count_agreements(self, results: list[Any]) -> int:
        """Count how many results agree with each other."""
        if not results:
            return 0

        # Group by similarity
        groups: list[list[Any]] = []

        for result in results:
            found_group = False
            for group in groups:
                if self._results_match(result, group[0]):
                    group.append(result)
                    found_group = True
                    break

            if not found_group:
                groups.append([result])

        # Return size of largest agreement group
        return max(len(g) for g in groups) if groups else 0

    def _results_match(self, result1: Any, result2: Any, tolerance: float = 0.001) -> bool:
        """Check if two results match."""
        if type(result1) != type(result2):
            return False

        if isinstance(result1, dict):
            # Compare dict results
            if set(result1.keys()) != set(result2.keys()):
                return False

            for key in result1:
                if not self._results_match(result1[key], result2[key], tolerance):
                    return False
            return True

        elif isinstance(result1, (int, float)):
            # Numeric comparison with tolerance
            if result1 == 0 and result2 == 0:
                return True
            if result1 == 0 or result2 == 0:
                return abs(result1 - result2) < tolerance
            return abs(result1 - result2) / max(abs(result1), abs(result2)) < tolerance

        elif isinstance(result1, str):
            # String comparison
            return result1.strip().lower() == result2.strip().lower()

        elif isinstance(result1, list):
            # List comparison
            if len(result1) != len(result2):
                return False
            return all(
                self._results_match(r1, r2, tolerance)
                for r1, r2 in zip(result1, result2)
            )

        return result1 == result2

    @property
    def stats(self) -> dict[str, Any]:
        """Get defense statistics including verification history."""
        base_stats = super().stats

        if self._verification_history:
            avg_agreement = sum(
                v.get("agreement_rate", 0) for v in self._verification_history
            ) / len(self._verification_history)

            consensus_rate = sum(
                1 for v in self._verification_history if v.get("consensus", False)
            ) / len(self._verification_history)

            base_stats.update({
                "total_verifications": len(self._verification_history),
                "average_agreement_rate": avg_agreement,
                "consensus_rate": consensus_rate,
            })

        return base_stats
