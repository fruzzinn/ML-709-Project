"""Self-consistency checker for agent reasoning validation.

Based on patterns from Wiqayah's consistency-aggregator.ts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import structlog

from src.defenses.base import Defense, DefenseAction, DefenseResult, DefenseType

if TYPE_CHECKING:
    from src.agent.state import AgentState, ToolCallResult
    from src.llm.client import LLMClient

logger = structlog.get_logger()


@dataclass
class ConsistencyCheck:
    """Result of a single consistency check."""

    check_type: str
    passed: bool
    score: float
    details: str | None = None


class ConsistencyChecker(Defense):
    """Self-consistency defense for validating agent reasoning.

    Features:
    - Multi-check scoring with aggregation
    - Hard minimum threshold enforcement
    - Soft threshold for warnings
    - LLM-based consistency verification (optional)

    Based on the self-consistency patterns from Wiqayah's ReAct engine.
    """

    def __init__(
        self,
        enabled: bool = True,
        num_checks: int = 3,
        threshold: float = 0.7,
        hard_minimum: float = 0.5,
        llm_client: LLMClient | None = None,
        use_llm_verification: bool = False,
    ) -> None:
        super().__init__(enabled)
        self.num_checks = num_checks
        self.threshold = threshold
        self.hard_minimum = hard_minimum
        self.llm_client = llm_client
        self.use_llm_verification = use_llm_verification

        self._check_history: list[list[ConsistencyCheck]] = []
        self._log = logger.bind(component="consistency_checker")

    @property
    def defense_type(self) -> DefenseType:
        return DefenseType.SELF_CONSISTENCY

    async def pre_execute_check(
        self,
        tool_calls: list[dict[str, Any]],
        state: AgentState,
    ) -> DefenseResult:
        """Pre-execution consistency check - verify tool calls make sense."""
        checks: list[ConsistencyCheck] = []

        # Check 1: Tool call relevance to task
        task_relevance = self._check_task_relevance(tool_calls, state)
        checks.append(task_relevance)

        # Check 2: Argument validity
        argument_validity = self._check_argument_validity(tool_calls)
        checks.append(argument_validity)

        # Check 3: Progress toward goal
        progress_check = self._check_progress(tool_calls, state)
        checks.append(progress_check)

        # Aggregate scores
        scores = [c.score for c in checks]
        avg_score = sum(scores) / len(scores) if scores else 0
        min_score = min(scores) if scores else 0

        self._check_history.append(checks)

        # Determine action based on thresholds
        if min_score < self.hard_minimum:
            return DefenseResult(
                defense_type=self.defense_type,
                action=DefenseAction.BLOCK,
                passed=False,
                confidence=avg_score,
                reason=f"Consistency below hard minimum ({min_score:.2f} < {self.hard_minimum})",
                details={"checks": [c.__dict__ for c in checks]},
            )

        if avg_score < self.threshold:
            return DefenseResult(
                defense_type=self.defense_type,
                action=DefenseAction.WARN,
                passed=True,
                confidence=avg_score,
                reason=f"Low consistency score ({avg_score:.2f} < {self.threshold})",
                details={"checks": [c.__dict__ for c in checks]},
            )

        return DefenseResult(
            defense_type=self.defense_type,
            action=DefenseAction.ALLOW,
            passed=True,
            confidence=avg_score,
        )

    async def post_execute_check(
        self,
        results: list[ToolCallResult],
        state: AgentState,
    ) -> DefenseResult:
        """Post-execution consistency check - verify results are consistent."""
        checks: list[ConsistencyCheck] = []

        # Check 1: Result consistency
        result_consistency = self._check_result_consistency(results, state)
        checks.append(result_consistency)

        # Check 2: No contradictions with previous results
        contradiction_check = self._check_contradictions(results, state)
        checks.append(contradiction_check)

        # Check 3: Results align with expectations
        expectation_check = self._check_expectations(results, state)
        checks.append(expectation_check)

        # Optional: LLM-based verification
        if self.use_llm_verification and self.llm_client:
            llm_check = await self._llm_consistency_check(results, state)
            checks.append(llm_check)

        # Aggregate scores
        scores = [c.score for c in checks]
        avg_score = sum(scores) / len(scores) if scores else 0
        min_score = min(scores) if scores else 0

        self._check_history.append(checks)

        if min_score < self.hard_minimum:
            return DefenseResult(
                defense_type=self.defense_type,
                action=DefenseAction.WARN,
                passed=False,
                confidence=avg_score,
                reason=f"Post-execution consistency below minimum ({min_score:.2f})",
                details={"checks": [c.__dict__ for c in checks]},
            )

        if avg_score < self.threshold:
            return DefenseResult(
                defense_type=self.defense_type,
                action=DefenseAction.WARN,
                passed=True,
                confidence=avg_score,
                reason=f"Low post-execution consistency ({avg_score:.2f})",
                details={"checks": [c.__dict__ for c in checks]},
            )

        return DefenseResult(
            defense_type=self.defense_type,
            action=DefenseAction.ALLOW,
            passed=True,
            confidence=avg_score,
        )

    def _check_task_relevance(
        self,
        tool_calls: list[dict[str, Any]],
        state: AgentState,
    ) -> ConsistencyCheck:
        """Check if tool calls are relevant to the task."""
        if not tool_calls:
            return ConsistencyCheck(
                check_type="task_relevance",
                passed=True,
                score=1.0,
                details="No tool calls to check",
            )

        # Simple heuristic: check if tool names are reasonable
        tool_names = [c.get("name", "") for c in tool_calls]
        valid_tools = {"calculator", "web_search", "code_executor", "file_reader"}

        valid_count = sum(1 for t in tool_names if t in valid_tools)
        score = valid_count / len(tool_names) if tool_names else 0

        return ConsistencyCheck(
            check_type="task_relevance",
            passed=score >= self.threshold,
            score=score,
            details=f"Tools: {tool_names}",
        )

    def _check_argument_validity(
        self,
        tool_calls: list[dict[str, Any]],
    ) -> ConsistencyCheck:
        """Check if tool arguments are valid."""
        if not tool_calls:
            return ConsistencyCheck(
                check_type="argument_validity",
                passed=True,
                score=1.0,
            )

        valid_count = 0
        for call in tool_calls:
            args = call.get("arguments", {})
            # Check for non-empty arguments
            if args and all(v is not None for v in args.values()):
                valid_count += 1

        score = valid_count / len(tool_calls)

        return ConsistencyCheck(
            check_type="argument_validity",
            passed=score >= self.threshold,
            score=score,
        )

    def _check_progress(
        self,
        tool_calls: list[dict[str, Any]],
        state: AgentState,
    ) -> ConsistencyCheck:
        """Check if we're making progress toward the goal."""
        # Heuristic: check we're not repeating the same calls
        if not state.reasoning_steps:
            return ConsistencyCheck(
                check_type="progress",
                passed=True,
                score=1.0,
            )

        # Get previous tool calls
        previous_calls = []
        for step in state.reasoning_steps[-3:]:  # Last 3 steps
            for call in step.tool_calls:
                previous_calls.append((call.get("name"), str(call.get("arguments"))))

        # Check current calls aren't exact duplicates
        current_calls = [(c.get("name"), str(c.get("arguments"))) for c in tool_calls]

        duplicate_count = sum(1 for c in current_calls if c in previous_calls)

        if not current_calls:
            score = 1.0
        else:
            score = 1.0 - (duplicate_count / len(current_calls))

        return ConsistencyCheck(
            check_type="progress",
            passed=score >= self.threshold,
            score=score,
            details=f"Duplicates: {duplicate_count}",
        )

    def _check_result_consistency(
        self,
        results: list[ToolCallResult],
        state: AgentState,
    ) -> ConsistencyCheck:
        """Check if results are internally consistent."""
        if not results:
            return ConsistencyCheck(
                check_type="result_consistency",
                passed=True,
                score=1.0,
            )

        # Check for errors and anomalies
        error_count = sum(1 for r in results if r.error)
        anomaly_count = sum(1 for r in results if r.anomaly_detected)

        issue_count = error_count + anomaly_count
        score = 1.0 - (issue_count / len(results)) if results else 1.0

        return ConsistencyCheck(
            check_type="result_consistency",
            passed=score >= self.threshold,
            score=score,
            details=f"Errors: {error_count}, Anomalies: {anomaly_count}",
        )

    def _check_contradictions(
        self,
        results: list[ToolCallResult],
        state: AgentState,
    ) -> ConsistencyCheck:
        """Check for contradictions with previous results."""
        # Simplified: check if calculator results are consistent
        calculator_results = [
            r.result for r in results
            if r.tool_name == "calculator" and r.result and not r.error
        ]

        if len(calculator_results) < 2:
            return ConsistencyCheck(
                check_type="contradictions",
                passed=True,
                score=1.0,
            )

        # Check if results that should be equal are similar
        # (This is a simplified check - production would be more sophisticated)
        return ConsistencyCheck(
            check_type="contradictions",
            passed=True,
            score=0.9,
            details="Simplified contradiction check",
        )

    def _check_expectations(
        self,
        results: list[ToolCallResult],
        state: AgentState,
    ) -> ConsistencyCheck:
        """Check if results align with expectations."""
        if not results:
            return ConsistencyCheck(
                check_type="expectations",
                passed=True,
                score=1.0,
            )

        # Check execution times are reasonable
        slow_results = [r for r in results if r.execution_time_ms > 10000]  # > 10s

        if slow_results:
            score = 1.0 - (len(slow_results) / len(results))
            return ConsistencyCheck(
                check_type="expectations",
                passed=score >= self.threshold,
                score=score,
                details=f"Slow results: {len(slow_results)}",
            )

        return ConsistencyCheck(
            check_type="expectations",
            passed=True,
            score=1.0,
        )

    async def _llm_consistency_check(
        self,
        results: list[ToolCallResult],
        state: AgentState,
    ) -> ConsistencyCheck:
        """Use LLM to verify consistency (optional)."""
        if not self.llm_client:
            return ConsistencyCheck(
                check_type="llm_verification",
                passed=True,
                score=1.0,
                details="LLM client not configured",
            )

        # Build verification prompt
        results_summary = "\n".join([
            f"- {r.tool_name}: {r.result if not r.error else f'Error: {r.error}'}"
            for r in results
        ])

        prompt = f"""Analyze these tool results for consistency with the task.

Task: {state.task}

Results:
{results_summary}

Are these results consistent and reasonable? Rate from 0-10.
Just respond with a number."""

        try:
            response = await self.llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10,
            )

            # Parse score from response
            score_str = response.content.strip()
            score = float(score_str) / 10.0

            return ConsistencyCheck(
                check_type="llm_verification",
                passed=score >= self.threshold,
                score=score,
                details=f"LLM rating: {score_str}/10",
            )

        except Exception as e:
            self._log.warning("LLM consistency check failed", error=str(e))
            return ConsistencyCheck(
                check_type="llm_verification",
                passed=True,
                score=0.8,  # Default to passing with lower score
                details=f"LLM check failed: {e}",
            )

    @property
    def stats(self) -> dict[str, Any]:
        """Get defense statistics."""
        base_stats = super().stats

        if self._check_history:
            all_scores = [
                c.score
                for checks in self._check_history
                for c in checks
            ]
            avg_score = sum(all_scores) / len(all_scores) if all_scores else 0

            base_stats.update({
                "total_check_rounds": len(self._check_history),
                "average_consistency_score": avg_score,
            })

        return base_stats
