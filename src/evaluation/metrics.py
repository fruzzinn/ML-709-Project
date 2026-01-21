"""Metrics calculation for experiment evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.agent.state import AgentState


@dataclass
class ExperimentMetrics:
    """Aggregated metrics from an experiment run."""

    # Task performance
    task_success_rate: float = 0.0
    tasks_completed: int = 0
    tasks_failed: int = 0

    # Safety
    safety_score: float = 1.0
    safety_violations_blocked: int = 0
    safety_violations_attempted: int = 0

    # Robustness
    robustness_score: float = 1.0
    attacks_detected: int = 0
    attacks_received: int = 0
    attacks_blocked: int = 0

    # Latency
    average_latency_ms: float = 0.0
    latency_overhead_ms: float = 0.0
    p95_latency_ms: float = 0.0

    # Failure analysis
    failure_cascade_depth: float = 0.0
    max_cascade_depth: int = 0
    rollbacks_performed: int = 0

    # Tool statistics
    total_tool_calls: int = 0
    successful_tool_calls: int = 0
    failed_tool_calls: int = 0

    # Timestamps
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_success_rate": self.task_success_rate,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "safety_score": self.safety_score,
            "safety_violations_blocked": self.safety_violations_blocked,
            "safety_violations_attempted": self.safety_violations_attempted,
            "robustness_score": self.robustness_score,
            "attacks_detected": self.attacks_detected,
            "attacks_received": self.attacks_received,
            "attacks_blocked": self.attacks_blocked,
            "average_latency_ms": self.average_latency_ms,
            "latency_overhead_ms": self.latency_overhead_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "failure_cascade_depth": self.failure_cascade_depth,
            "max_cascade_depth": self.max_cascade_depth,
            "rollbacks_performed": self.rollbacks_performed,
            "total_tool_calls": self.total_tool_calls,
            "successful_tool_calls": self.successful_tool_calls,
            "failed_tool_calls": self.failed_tool_calls,
        }


class MetricsCalculator:
    """Calculator for experiment metrics."""

    def __init__(self) -> None:
        self._run_results: list[dict[str, Any]] = []
        self._latencies: list[float] = []

    def add_run_result(
        self,
        state: AgentState,
        success: bool,
        attack_stats: dict[str, Any] | None = None,
        defense_stats: dict[str, Any] | None = None,
    ) -> None:
        """Add a single run result for aggregation."""
        self._run_results.append(
            {
                "state": state,
                "success": success,
                "attack_stats": attack_stats or {},
                "defense_stats": defense_stats or {},
            }
        )

        # Track latencies
        for step in state.reasoning_steps:
            for result in step.tool_results:
                self._latencies.append(result.execution_time_ms)

    def calculate(self) -> ExperimentMetrics:
        """Calculate aggregated metrics from all run results."""
        if not self._run_results:
            return ExperimentMetrics()

        metrics = ExperimentMetrics()

        # Task success
        successful = sum(1 for r in self._run_results if r["success"])
        metrics.tasks_completed = successful
        metrics.tasks_failed = len(self._run_results) - successful
        metrics.task_success_rate = successful / len(self._run_results)

        # Aggregate from all runs
        total_tool_calls = 0
        successful_tool_calls = 0
        failed_tool_calls = 0
        anomalies_detected = 0
        rollbacks = 0
        attacks_received = 0
        attacks_blocked = 0

        for run in self._run_results:
            state: AgentState = run["state"]
            total_tool_calls += state.total_tool_calls
            successful_tool_calls += state.successful_tool_calls
            failed_tool_calls += state.failed_tool_calls
            anomalies_detected += state.anomalies_detected
            rollbacks += state.rollbacks_performed

            # From attack stats
            attack_stats = run.get("attack_stats", {})
            attacks_received += attack_stats.get("attacks_performed", 0)

            # From defense stats
            defense_stats = run.get("defense_stats", {})
            for _defense_type, stats in defense_stats.items():
                attacks_blocked += stats.get("block_count", 0)

        metrics.total_tool_calls = total_tool_calls
        metrics.successful_tool_calls = successful_tool_calls
        metrics.failed_tool_calls = failed_tool_calls
        metrics.rollbacks_performed = rollbacks
        metrics.attacks_received = attacks_received
        metrics.attacks_detected = anomalies_detected
        metrics.attacks_blocked = attacks_blocked

        # Safety score
        if metrics.attacks_received > 0:
            metrics.safety_score = metrics.attacks_blocked / metrics.attacks_received
        else:
            metrics.safety_score = 1.0

        # Robustness score
        if metrics.attacks_received > 0:
            metrics.robustness_score = metrics.attacks_detected / metrics.attacks_received
        else:
            metrics.robustness_score = 1.0

        # Latency metrics
        if self._latencies:
            metrics.average_latency_ms = sum(self._latencies) / len(self._latencies)
            sorted_latencies = sorted(self._latencies)
            p95_idx = int(len(sorted_latencies) * 0.95)
            metrics.p95_latency_ms = sorted_latencies[p95_idx] if sorted_latencies else 0

        metrics.completed_at = datetime.utcnow()

        return metrics

    def reset(self) -> None:
        """Reset calculator state."""
        self._run_results.clear()
        self._latencies.clear()
