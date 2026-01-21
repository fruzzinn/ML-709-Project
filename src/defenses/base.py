"""Base classes for defense mechanisms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from src.agent.state import AgentState, ToolCallResult

logger = structlog.get_logger()


class DefenseType(str, Enum):
    """Types of defense mechanisms."""

    TOOL_VERIFICATION = "tool_verification"
    REDUNDANCY = "redundancy"
    ROLLBACK = "rollback"
    ANOMALY_DETECTION = "anomaly_detection"
    SELF_CONSISTENCY = "self_consistency"
    SANITIZATION = "sanitization"


class DefenseAction(str, Enum):
    """Actions taken by defenses."""

    ALLOW = "allow"
    BLOCK = "block"
    WARN = "warn"
    MODIFY = "modify"
    ROLLBACK = "rollback"


@dataclass
class DefenseResult:
    """Result from a defense check."""

    defense_type: DefenseType
    action: DefenseAction
    passed: bool
    confidence: float = 1.0
    reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class Defense(ABC):
    """Abstract base class for defense mechanisms."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._check_count = 0
        self._block_count = 0
        self._log = logger.bind(component="defense", defense_type=self.defense_type.value)

    @property
    @abstractmethod
    def defense_type(self) -> DefenseType:
        """Type of this defense."""
        ...

    @abstractmethod
    async def pre_execute_check(
        self,
        tool_calls: list[dict[str, Any]],
        state: AgentState,
    ) -> DefenseResult:
        """Check before tool execution."""
        ...

    @abstractmethod
    async def post_execute_check(
        self,
        results: list[ToolCallResult],
        state: AgentState,
    ) -> DefenseResult:
        """Check after tool execution."""
        ...

    def record_check(self, result: DefenseResult) -> None:
        """Record a defense check."""
        self._check_count += 1
        if result.action == DefenseAction.BLOCK:
            self._block_count += 1

    @property
    def stats(self) -> dict[str, Any]:
        """Get defense statistics."""
        return {
            "defense_type": self.defense_type.value,
            "enabled": self.enabled,
            "check_count": self._check_count,
            "block_count": self._block_count,
            "block_rate": self._block_count / self._check_count if self._check_count > 0 else 0,
        }


class DefenseManager:
    """Manager for coordinating multiple defense mechanisms."""

    def __init__(self) -> None:
        self._defenses: dict[DefenseType, Defense] = {}
        self._results_history: list[DefenseResult] = []
        self._log = logger.bind(component="defense_manager")

    def register(self, defense: Defense) -> None:
        """Register a defense mechanism."""
        self._defenses[defense.defense_type] = defense
        self._log.info("Registered defense", defense_type=defense.defense_type.value)

    def unregister(self, defense_type: DefenseType) -> bool:
        """Unregister a defense mechanism."""
        if defense_type in self._defenses:
            del self._defenses[defense_type]
            return True
        return False

    def get(self, defense_type: DefenseType) -> Defense | None:
        """Get a defense by type."""
        return self._defenses.get(defense_type)

    async def pre_execute_check(
        self,
        tool_calls: list[dict[str, Any]],
        state: AgentState,
    ) -> tuple[bool, str | None]:
        """Run all pre-execution checks.

        Returns:
            Tuple of (all_passed, rejection_reason)
        """
        for defense in self._defenses.values():
            if not defense.enabled:
                continue

            result = await defense.pre_execute_check(tool_calls, state)
            defense.record_check(result)
            self._results_history.append(result)

            if result.action == DefenseAction.BLOCK:
                self._log.warning(
                    "Pre-execution check blocked",
                    defense=defense.defense_type.value,
                    reason=result.reason,
                )
                return False, result.reason

        return True, None

    async def detect_anomaly(
        self,
        results: list[ToolCallResult],
        state: AgentState,
    ) -> tuple[bool, str | None]:
        """Run all post-execution anomaly detection.

        Returns:
            Tuple of (anomaly_detected, anomaly_details)
        """
        anomaly_detected = False
        anomaly_details: list[str] = []

        for defense in self._defenses.values():
            if not defense.enabled:
                continue

            result = await defense.post_execute_check(results, state)
            defense.record_check(result)
            self._results_history.append(result)

            if result.action in [DefenseAction.BLOCK, DefenseAction.WARN, DefenseAction.ROLLBACK]:
                anomaly_detected = True
                if result.reason:
                    anomaly_details.append(f"{defense.defense_type.value}: {result.reason}")

        if anomaly_detected:
            self._log.warning(
                "Anomaly detected",
                details=anomaly_details,
            )
            return True, "; ".join(anomaly_details)

        return False, None

    def get_all_stats(self) -> dict[str, Any]:
        """Get statistics from all defenses."""
        return {
            defense_type.value: defense.stats
            for defense_type, defense in self._defenses.items()
        }

    def get_results_history(self) -> list[DefenseResult]:
        """Get history of all defense results."""
        return self._results_history.copy()

    def clear_history(self) -> None:
        """Clear results history."""
        self._results_history.clear()

    @property
    def defense_count(self) -> int:
        """Get number of registered defenses."""
        return len(self._defenses)

    @property
    def enabled_defenses(self) -> list[DefenseType]:
        """Get list of enabled defense types."""
        return [dt for dt, d in self._defenses.items() if d.enabled]
