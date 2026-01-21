"""Attack type definitions and configurations."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AttackType(str, Enum):
    """Types of adversarial attacks."""

    WRONG_OUTPUT = "wrong_output"
    DELAYED_RESPONSE = "delayed_response"
    POISONED_API = "poisoned_api"
    BYZANTINE = "byzantine"
    COLLUSION = "collusion"


@dataclass
class AttackConfig:
    """Configuration for an attack type."""

    attack_type: AttackType
    enabled: bool = True
    probability: float = 0.3
    target_tools: list[str] | None = None  # None = all tools
    parameters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def wrong_output(
        cls,
        probability: float = 0.3,
        error_magnitude: float = 0.1,
        strategies: list[str] | None = None,
        target_tools: list[str] | None = None,
    ) -> AttackConfig:
        """Create wrong output attack configuration."""
        return cls(
            attack_type=AttackType.WRONG_OUTPUT,
            probability=probability,
            target_tools=target_tools,
            parameters={
                "error_magnitude": error_magnitude,
                "strategies": strategies or ["off_by_one", "partial_result", "hallucinated"],
            },
        )

    @classmethod
    def delayed_response(
        cls,
        probability: float = 0.25,
        min_delay: float = 1.0,
        max_delay: float = 10.0,
        timeout_probability: float = 0.1,
        target_tools: list[str] | None = None,
    ) -> AttackConfig:
        """Create delayed response attack configuration."""
        return cls(
            attack_type=AttackType.DELAYED_RESPONSE,
            probability=probability,
            target_tools=target_tools,
            parameters={
                "min_delay": min_delay,
                "max_delay": max_delay,
                "timeout_probability": timeout_probability,
            },
        )

    @classmethod
    def poisoned_api(
        cls,
        probability: float = 0.2,
        strategies: list[str] | None = None,
        custom_payloads: dict[str, list[str]] | None = None,
        target_tools: list[str] | None = None,
    ) -> AttackConfig:
        """Create poisoned API attack configuration."""
        return cls(
            attack_type=AttackType.POISONED_API,
            probability=probability,
            target_tools=target_tools,
            parameters={
                "strategies": strategies or ["prompt_injection", "goal_hijacking"],
                "custom_payloads": custom_payloads or {},
            },
        )

    @classmethod
    def byzantine(
        cls,
        probability: float = 0.3,
        failure_modes: list[str] | None = None,
        target_tools: list[str] | None = None,
    ) -> AttackConfig:
        """Create Byzantine attack configuration."""
        return cls(
            attack_type=AttackType.BYZANTINE,
            probability=probability,
            target_tools=target_tools,
            parameters={
                "failure_modes": failure_modes
                or ["wrong_result", "partial_result", "delayed", "error", "empty"],
            },
        )

    @classmethod
    def collusion(
        cls,
        probability: float = 0.3,
        collusion_id: str = "default",
        target_tools: list[str] | None = None,
    ) -> AttackConfig:
        """Create collusion attack configuration."""
        return cls(
            attack_type=AttackType.COLLUSION,
            probability=probability,
            target_tools=target_tools,
            parameters={
                "collusion_id": collusion_id,
            },
        )

    def should_target_tool(self, tool_name: str) -> bool:
        """Check if this attack should target the given tool."""
        if self.target_tools is None:
            return True
        return tool_name in self.target_tools


@dataclass
class AttackScenario:
    """A complete attack scenario with multiple attack types."""

    name: str
    description: str
    attacks: list[AttackConfig]
    duration_loops: int | None = None  # None = entire experiment

    def get_attacks_for_tool(self, tool_name: str) -> list[AttackConfig]:
        """Get all attacks that target a specific tool."""
        return [a for a in self.attacks if a.enabled and a.should_target_tool(tool_name)]


# Pre-defined attack scenarios for experiments
BASELINE_SCENARIO = AttackScenario(
    name="baseline",
    description="No attacks - baseline performance measurement",
    attacks=[],
)

LIGHT_ATTACK_SCENARIO = AttackScenario(
    name="light_attack",
    description="Light adversarial pressure - 10% attack probability",
    attacks=[
        AttackConfig.wrong_output(probability=0.1),
        AttackConfig.delayed_response(probability=0.1, max_delay=3.0),
    ],
)

MODERATE_ATTACK_SCENARIO = AttackScenario(
    name="moderate_attack",
    description="Moderate adversarial pressure - 30% attack probability",
    attacks=[
        AttackConfig.wrong_output(probability=0.3),
        AttackConfig.delayed_response(probability=0.25),
        AttackConfig.poisoned_api(probability=0.2),
    ],
)

HEAVY_ATTACK_SCENARIO = AttackScenario(
    name="heavy_attack",
    description="Heavy adversarial pressure - 50% attack probability",
    attacks=[
        AttackConfig.wrong_output(probability=0.5, error_magnitude=0.2),
        AttackConfig.delayed_response(probability=0.4, timeout_probability=0.2),
        AttackConfig.poisoned_api(probability=0.3),
        AttackConfig.byzantine(probability=0.3),
    ],
)

COORDINATED_ATTACK_SCENARIO = AttackScenario(
    name="coordinated_attack",
    description="Coordinated multi-tool attack",
    attacks=[
        AttackConfig.collusion(probability=0.4),
        AttackConfig.wrong_output(probability=0.3),
    ],
)

PREDEFINED_SCENARIOS = {
    "baseline": BASELINE_SCENARIO,
    "light": LIGHT_ATTACK_SCENARIO,
    "moderate": MODERATE_ATTACK_SCENARIO,
    "heavy": HEAVY_ATTACK_SCENARIO,
    "coordinated": COORDINATED_ATTACK_SCENARIO,
}
