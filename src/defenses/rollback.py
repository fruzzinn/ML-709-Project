"""Rollback defense - state checkpointing and recovery."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.defenses.base import Defense, DefenseAction, DefenseResult, DefenseType

if TYPE_CHECKING:
    from src.agent.state import AgentState, StateManager, ToolCallResult


class RollbackDefense(Defense):
    """Defense that manages rollback on detected anomalies.

    Features:
    - Automatic rollback triggers
    - Configurable rollback strategies
    - Failure tracking and patterns
    - Recovery coordination
    """

    def __init__(
        self,
        enabled: bool = True,
        state_manager: StateManager | None = None,
        auto_rollback_on_error: bool = True,
        auto_rollback_on_anomaly: bool = True,
        max_rollback_depth: int = 5,
        failure_threshold: float = 0.3,
    ) -> None:
        super().__init__(enabled)
        self.state_manager = state_manager
        self.auto_rollback_on_error = auto_rollback_on_error
        self.auto_rollback_on_anomaly = auto_rollback_on_anomaly
        self.max_rollback_depth = max_rollback_depth
        self.failure_threshold = failure_threshold

        # Track failures for pattern detection
        self._failure_history: list[dict[str, Any]] = []
        self._rollback_count = 0
        self._consecutive_failures = 0

    @property
    def defense_type(self) -> DefenseType:
        return DefenseType.ROLLBACK

    async def pre_execute_check(
        self,
        tool_calls: list[dict[str, Any]],
        state: AgentState,
    ) -> DefenseResult:
        """Check if rollback is needed before execution."""
        # Check consecutive failure threshold
        if self._consecutive_failures >= 3:
            self._log.warning(
                "Multiple consecutive failures detected",
                consecutive=self._consecutive_failures,
            )
            return DefenseResult(
                defense_type=self.defense_type,
                action=DefenseAction.WARN,
                passed=True,
                reason=f"High failure rate detected ({self._consecutive_failures} consecutive)",
                details={"consecutive_failures": self._consecutive_failures},
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
        """Check results and determine if rollback is needed."""
        errors = [r for r in results if r.error]
        anomalies = [r for r in results if r.anomaly_detected]

        # Track failures
        if errors or anomalies:
            self._consecutive_failures += 1
            self._failure_history.append({
                "loop": state.current_loop,
                "errors": [{"tool": r.tool_name, "error": r.error} for r in errors],
                "anomalies": [{"tool": r.tool_name, "details": r.anomaly_details} for r in anomalies],
            })
        else:
            self._consecutive_failures = 0

        # Determine if rollback should be triggered
        should_rollback = False
        rollback_reason = None

        if self.auto_rollback_on_error and errors:
            should_rollback = True
            rollback_reason = f"Tool errors: {[r.tool_name for r in errors]}"

        if self.auto_rollback_on_anomaly and anomalies:
            should_rollback = True
            rollback_reason = f"Anomalies detected: {[r.tool_name for r in anomalies]}"

        if should_rollback and self._rollback_count < self.max_rollback_depth:
            self._rollback_count += 1
            return DefenseResult(
                defense_type=self.defense_type,
                action=DefenseAction.ROLLBACK,
                passed=False,
                reason=rollback_reason,
                details={
                    "rollback_number": self._rollback_count,
                    "errors": len(errors),
                    "anomalies": len(anomalies),
                },
            )

        if should_rollback and self._rollback_count >= self.max_rollback_depth:
            return DefenseResult(
                defense_type=self.defense_type,
                action=DefenseAction.BLOCK,
                passed=False,
                reason=f"Max rollback depth reached ({self.max_rollback_depth})",
                details={"rollback_count": self._rollback_count},
            )

        return DefenseResult(
            defense_type=self.defense_type,
            action=DefenseAction.ALLOW,
            passed=True,
        )

    def should_trigger_rollback(self, state: AgentState) -> bool:
        """Determine if a rollback should be triggered."""
        # Check failure rate
        if len(self._failure_history) > 5:
            recent_failures = self._failure_history[-5:]
            failure_rate = len([f for f in recent_failures if f.get("errors")]) / 5
            if failure_rate > self.failure_threshold:
                return True

        # Check consecutive failures
        if self._consecutive_failures >= 3:
            return True

        return False

    def on_rollback_success(self) -> None:
        """Called when a rollback succeeds."""
        self._log.info("Rollback succeeded", rollback_count=self._rollback_count)
        # Don't reset consecutive failures - they led to the rollback

    def on_rollback_failure(self) -> None:
        """Called when a rollback fails."""
        self._log.error("Rollback failed", rollback_count=self._rollback_count)
        self._rollback_count += 1

    def reset(self) -> None:
        """Reset rollback tracking."""
        self._failure_history.clear()
        self._rollback_count = 0
        self._consecutive_failures = 0

    @property
    def stats(self) -> dict[str, Any]:
        """Get defense statistics."""
        base_stats = super().stats
        base_stats.update({
            "rollback_count": self._rollback_count,
            "consecutive_failures": self._consecutive_failures,
            "total_failures_tracked": len(self._failure_history),
        })
        return base_stats
