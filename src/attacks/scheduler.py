"""Attack scheduler for controlling when and how attacks occur."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import structlog

from src.attacks.attack_types import AttackConfig, AttackScenario, AttackType
from src.tools.adversarial.wrappers import (
    AdversarialWrapper,
    ByzantineWrapper,
    CollusionWrapper,
    DelayedResponseWrapper,
    PoisonedAPIWrapper,
    WrongOutputWrapper,
)
from src.tools.base import BaseTool
from src.tools.registry import ToolRegistry

logger = structlog.get_logger()


class SchedulerStrategy(str, Enum):
    """Attack scheduling strategies."""

    NONE = "none"  # No attacks
    RANDOM = "random"  # Random probability-based attacks
    TARGETED = "targeted"  # Focus on specific tools
    ESCALATING = "escalating"  # Increase attack intensity over time
    BURST = "burst"  # Clustered attacks in bursts
    ADAPTIVE = "adaptive"  # Adapt based on agent behavior


@dataclass
class SchedulerState:
    """State tracking for the scheduler."""

    loop_number: int = 0
    total_tool_calls: int = 0
    attacks_performed: int = 0
    attacks_detected: int = 0
    current_phase: str = "normal"
    burst_active: bool = False
    burst_remaining: int = 0
    tool_call_history: list[str] = field(default_factory=list)
    attack_history: list[dict[str, Any]] = field(default_factory=list)


class AttackScheduler:
    """Scheduler for coordinating adversarial attacks.

    Strategies:
    - random: Each tool call has independent attack probability
    - targeted: Focus attacks on specific tools
    - escalating: Increase attack probability over time
    - burst: Cluster attacks in bursts with quiet periods
    - adaptive: Adjust based on agent's success rate
    """

    def __init__(
        self,
        strategy: SchedulerStrategy = SchedulerStrategy.RANDOM,
        scenario: AttackScenario | None = None,
    ) -> None:
        self.strategy = strategy
        self.scenario = scenario
        self.state = SchedulerState()
        self._wrappers: dict[str, AdversarialWrapper] = {}
        self._log = logger.bind(component="attack_scheduler", strategy=strategy.value)

        # Strategy-specific parameters
        self._escalation_rate = 0.05  # Probability increase per loop
        self._burst_probability = 0.2  # Probability of starting a burst
        self._burst_length = 3  # Number of calls in a burst
        self._adaptive_threshold = 0.7  # Success rate threshold

    def setup(self, registry: ToolRegistry) -> None:
        """Set up attack wrappers for all tools in the registry."""
        if not self.scenario or self.strategy == SchedulerStrategy.NONE:
            self._log.info("Attack scheduler disabled")
            return

        self._log.info(
            "Setting up attack scheduler",
            scenario=self.scenario.name,
            num_attacks=len(self.scenario.attacks),
        )

        for tool_name in registry.tool_names:
            original_tool = registry.get_original(tool_name)
            if not original_tool:
                continue

            # Get attacks that target this tool
            attacks = self.scenario.get_attacks_for_tool(tool_name)
            if not attacks:
                continue

            # Create wrapper based on attack type
            # If multiple attacks target same tool, use the first one
            # (could be extended to chain wrappers)
            attack = attacks[0]
            wrapper = self._create_wrapper(original_tool, attack)

            if wrapper:
                registry.wrap_tool(tool_name, wrapper)
                self._wrappers[tool_name] = wrapper
                self._log.debug(
                    "Wrapped tool with attack",
                    tool=tool_name,
                    attack_type=attack.attack_type.value,
                )

    def _create_wrapper(
        self,
        tool: BaseTool,
        attack: AttackConfig,
    ) -> AdversarialWrapper | None:
        """Create appropriate wrapper for attack type."""
        params = attack.parameters

        if attack.attack_type == AttackType.WRONG_OUTPUT:
            return WrongOutputWrapper(
                tool,
                attack_probability=attack.probability,
                error_magnitude=params.get("error_magnitude", 0.1),
                strategies=params.get("strategies"),
            )

        elif attack.attack_type == AttackType.DELAYED_RESPONSE:
            return DelayedResponseWrapper(
                tool,
                attack_probability=attack.probability,
                min_delay=params.get("min_delay", 1.0),
                max_delay=params.get("max_delay", 10.0),
                timeout_probability=params.get("timeout_probability", 0.1),
            )

        elif attack.attack_type == AttackType.POISONED_API:
            return PoisonedAPIWrapper(
                tool,
                attack_probability=attack.probability,
                strategies=params.get("strategies"),
                custom_payloads=params.get("custom_payloads"),
            )

        elif attack.attack_type == AttackType.BYZANTINE:
            return ByzantineWrapper(
                tool,
                attack_probability=attack.probability,
                failure_modes=params.get("failure_modes"),
            )

        elif attack.attack_type == AttackType.COLLUSION:
            return CollusionWrapper(
                tool,
                attack_probability=attack.probability,
                collusion_id=params.get("collusion_id", "default"),
            )

        return None

    def on_loop_start(self, loop_number: int) -> None:
        """Called at the start of each agent loop."""
        self.state.loop_number = loop_number
        self._update_attack_probabilities()

    def on_tool_call(self, tool_name: str) -> None:
        """Called before each tool execution."""
        self.state.total_tool_calls += 1
        self.state.tool_call_history.append(tool_name)

        # Update wrapper probability based on strategy
        if tool_name in self._wrappers:
            wrapper = self._wrappers[tool_name]
            wrapper.attack_probability = self._get_effective_probability(tool_name)

    def on_attack_detected(self, tool_name: str) -> None:
        """Called when an attack is detected by defenses."""
        self.state.attacks_detected += 1

        # Adaptive strategy: reduce attack probability if being detected
        if self.strategy == SchedulerStrategy.ADAPTIVE:
            if tool_name in self._wrappers:
                self._wrappers[tool_name].attack_probability *= 0.8

    def _update_attack_probabilities(self) -> None:
        """Update attack probabilities based on strategy."""
        if self.strategy == SchedulerStrategy.ESCALATING:
            # Increase probability each loop
            for wrapper in self._wrappers.values():
                wrapper.attack_probability = min(
                    0.9,
                    wrapper.attack_probability + self._escalation_rate,
                )

        elif self.strategy == SchedulerStrategy.BURST:
            # Handle burst mode
            if self.state.burst_active:
                self.state.burst_remaining -= 1
                if self.state.burst_remaining <= 0:
                    self.state.burst_active = False
                    self._set_all_probabilities(0.1)  # Low between bursts
            else:
                # Check if we should start a burst
                if random.random() < self._burst_probability:
                    self.state.burst_active = True
                    self.state.burst_remaining = self._burst_length
                    self._set_all_probabilities(0.8)  # High during burst

    def _get_effective_probability(self, tool_name: str) -> float:
        """Get effective attack probability for a tool."""
        if tool_name not in self._wrappers:
            return 0.0

        base_probability = self._wrappers[tool_name].attack_probability

        if self.strategy == SchedulerStrategy.TARGETED:
            # Higher probability for frequently used tools
            usage_count = self.state.tool_call_history.count(tool_name)
            if usage_count > 3:
                return min(0.9, base_probability * 1.5)

        elif self.strategy == SchedulerStrategy.ADAPTIVE:
            # Adjust based on detection rate
            if self.state.attacks_performed > 0:
                detection_rate = self.state.attacks_detected / self.state.attacks_performed
                if detection_rate > 0.5:
                    return base_probability * 0.7  # Reduce if being detected

        return base_probability

    def _set_all_probabilities(self, probability: float) -> None:
        """Set all wrapper probabilities to a value."""
        for wrapper in self._wrappers.values():
            wrapper.attack_probability = probability

    def get_statistics(self) -> dict[str, Any]:
        """Get attack statistics."""
        stats = {
            "strategy": self.strategy.value,
            "scenario": self.scenario.name if self.scenario else None,
            "state": {
                "loop_number": self.state.loop_number,
                "total_tool_calls": self.state.total_tool_calls,
                "attacks_performed": sum(w.attack_count for w in self._wrappers.values()),
                "attacks_detected": self.state.attacks_detected,
            },
            "wrappers": {},
        }

        for name, wrapper in self._wrappers.items():
            stats["wrappers"][name] = {
                "attack_type": wrapper.attack_type,
                "attack_count": wrapper.attack_count,
                "current_probability": wrapper.attack_probability,
            }

        return stats

    def reset(self) -> None:
        """Reset scheduler state."""
        self.state = SchedulerState()
        for wrapper in self._wrappers.values():
            wrapper.reset_stats()
        CollusionWrapper.reset_collusion_state()

    def teardown(self, registry: ToolRegistry) -> None:
        """Remove all attack wrappers."""
        registry.unwrap_all()
        self._wrappers.clear()
        self._log.info("Attack scheduler torn down")
