"""Prompt generator for ADRS solution generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class PromptTemplate(str, Enum):
    """Available prompt templates for solution generation."""

    DEFENSE_DESIGN = "defense_design"
    DEFENSE_IMPROVEMENT = "defense_improvement"
    ATTACK_ANALYSIS = "attack_analysis"
    FAILURE_MITIGATION = "failure_mitigation"
    CONSISTENCY_CHECK = "consistency_check"


@dataclass
class PromptContext:
    """Context for prompt generation."""

    attack_types: list[str] = field(default_factory=list)
    current_defenses: list[str] = field(default_factory=list)
    failure_patterns: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)
    previous_solutions: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class GeneratedPrompt:
    """A generated prompt ready for LLM."""

    system_prompt: str
    user_prompt: str
    template: PromptTemplate
    context: PromptContext
    metadata: dict[str, Any] = field(default_factory=dict)


class PromptGenerator:
    """Generates prompts for ADRS solution generation.

    Creates structured prompts that guide LLMs to generate
    defense mechanism solutions based on attack scenarios
    and evaluation results.
    """

    SYSTEM_PROMPTS = {
        PromptTemplate.DEFENSE_DESIGN: """You are an expert in designing robust defense mechanisms for AI agent systems.
Your task is to design novel defense strategies that can protect against adversarial tool attacks.

Key considerations:
- Defense must be computationally efficient
- Should not significantly increase latency
- Must maintain agent task success rate
- Should detect and mitigate various attack types

Output your defense design as a structured specification with:
1. Defense name and description
2. Detection mechanism
3. Mitigation strategy
4. Integration points with agent loop
5. Expected overhead and trade-offs""",
        PromptTemplate.DEFENSE_IMPROVEMENT: """You are an expert in improving existing defense mechanisms.
Analyze the current defense performance and suggest specific improvements.

Focus on:
- Addressing detected weaknesses
- Reducing false positive rates
- Improving detection accuracy
- Minimizing performance overhead

Provide concrete, implementable improvements with expected impact.""",
        PromptTemplate.ATTACK_ANALYSIS: """You are a security researcher analyzing adversarial attacks on AI agents.
Your task is to understand attack patterns and identify vulnerabilities.

Analyze:
- Attack success patterns
- Conditions that enable attacks
- Agent behaviors that increase vulnerability
- Potential detection signals

Provide insights that can inform defense design.""",
        PromptTemplate.FAILURE_MITIGATION: """You are an expert in failure analysis and mitigation for distributed systems.
Analyze failure propagation patterns and design mitigation strategies.

Consider:
- Failure cascade paths
- Critical failure points
- Recovery mechanisms
- State preservation strategies

Design mitigations that minimize failure impact.""",
        PromptTemplate.CONSISTENCY_CHECK: """You are an expert in consistency verification for AI agent reasoning.
Design consistency checking mechanisms that can detect compromised reasoning.

Focus on:
- Cross-validation techniques
- Redundant verification
- Anomaly detection in reasoning chains
- Recovery from inconsistent states

Provide specific verification strategies.""",
    }

    def __init__(self) -> None:
        self._log = logger.bind(component="prompt_generator")
        self._generation_count = 0

    def generate(
        self,
        template: PromptTemplate,
        context: PromptContext,
    ) -> GeneratedPrompt:
        """Generate a prompt for the given template and context."""
        self._generation_count += 1

        system_prompt = self.SYSTEM_PROMPTS.get(
            template, self.SYSTEM_PROMPTS[PromptTemplate.DEFENSE_DESIGN]
        )
        user_prompt = self._build_user_prompt(template, context)

        self._log.debug(
            "Generated prompt",
            template=template.value,
            prompt_length=len(user_prompt),
        )

        return GeneratedPrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            template=template,
            context=context,
            metadata={"generation_number": self._generation_count},
        )

    def _build_user_prompt(
        self,
        template: PromptTemplate,
        context: PromptContext,
    ) -> str:
        """Build the user prompt based on template and context."""
        if template == PromptTemplate.DEFENSE_DESIGN:
            return self._build_defense_design_prompt(context)
        elif template == PromptTemplate.DEFENSE_IMPROVEMENT:
            return self._build_defense_improvement_prompt(context)
        elif template == PromptTemplate.ATTACK_ANALYSIS:
            return self._build_attack_analysis_prompt(context)
        elif template == PromptTemplate.FAILURE_MITIGATION:
            return self._build_failure_mitigation_prompt(context)
        elif template == PromptTemplate.CONSISTENCY_CHECK:
            return self._build_consistency_check_prompt(context)
        return self._build_defense_design_prompt(context)

    def _build_defense_design_prompt(self, context: PromptContext) -> str:
        """Build prompt for defense design."""
        sections = ["## Task: Design a Novel Defense Mechanism\n"]

        # Attack context
        if context.attack_types:
            sections.append("### Target Attack Types")
            for attack in context.attack_types:
                sections.append(f"- {attack}")
            sections.append("")

        # Current defenses
        if context.current_defenses:
            sections.append("### Existing Defenses (to complement, not duplicate)")
            for defense in context.current_defenses:
                sections.append(f"- {defense}")
            sections.append("")

        # Performance constraints
        if context.constraints:
            sections.append("### Constraints")
            for key, value in context.constraints.items():
                sections.append(f"- {key}: {value}")
            sections.append("")

        # Current metrics
        if context.metrics:
            sections.append("### Current Performance Metrics")
            for metric, value in context.metrics.items():
                sections.append(f"- {metric}: {value:.3f}")
            sections.append("")

        # Previous solutions to avoid
        if context.previous_solutions:
            sections.append("### Previously Tried Approaches (avoid similar)")
            for i, sol in enumerate(context.previous_solutions[-3:], 1):
                name = sol.get("name", f"Solution {i}")
                score = sol.get("fitness", 0)
                sections.append(f"- {name} (fitness: {score:.3f})")
            sections.append("")

        sections.append("### Requirements")
        sections.append("1. Provide a unique defense mechanism name")
        sections.append("2. Describe the detection algorithm")
        sections.append("3. Specify the mitigation action")
        sections.append("4. Estimate computational overhead")
        sections.append("5. List potential weaknesses")

        return "\n".join(sections)

    def _build_defense_improvement_prompt(self, context: PromptContext) -> str:
        """Build prompt for defense improvement."""
        sections = ["## Task: Improve Existing Defense\n"]

        if context.current_defenses:
            sections.append("### Current Defense to Improve")
            sections.append(context.current_defenses[0] if context.current_defenses else "Unknown")
            sections.append("")

        if context.metrics:
            sections.append("### Current Performance")
            for metric, value in context.metrics.items():
                sections.append(f"- {metric}: {value:.3f}")
            sections.append("")

        if context.failure_patterns:
            sections.append("### Observed Failure Patterns")
            for pattern in context.failure_patterns[:5]:
                sections.append(f"- {pattern.get('description', 'Unknown pattern')}")
            sections.append("")

        sections.append("### Requirements")
        sections.append("1. Identify specific weaknesses")
        sections.append("2. Propose concrete improvements")
        sections.append("3. Estimate improvement impact")
        sections.append("4. Consider implementation complexity")

        return "\n".join(sections)

    def _build_attack_analysis_prompt(self, context: PromptContext) -> str:
        """Build prompt for attack analysis."""
        sections = ["## Task: Analyze Attack Patterns\n"]

        if context.attack_types:
            sections.append("### Attack Types to Analyze")
            for attack in context.attack_types:
                sections.append(f"- {attack}")
            sections.append("")

        if context.failure_patterns:
            sections.append("### Observed Attack Successes")
            for pattern in context.failure_patterns[:10]:
                sections.append(f"- {pattern}")
            sections.append("")

        sections.append("### Analysis Requirements")
        sections.append("1. Identify common attack success conditions")
        sections.append("2. Find detection opportunities")
        sections.append("3. Suggest preventive measures")
        sections.append("4. Rank attacks by severity")

        return "\n".join(sections)

    def _build_failure_mitigation_prompt(self, context: PromptContext) -> str:
        """Build prompt for failure mitigation."""
        sections = ["## Task: Design Failure Mitigation Strategy\n"]

        if context.failure_patterns:
            sections.append("### Failure Patterns to Address")
            for pattern in context.failure_patterns:
                desc = pattern.get("description", "Unknown")
                severity = pattern.get("severity", "medium")
                sections.append(f"- [{severity.upper()}] {desc}")
            sections.append("")

        if context.metrics:
            sections.append("### Current Reliability Metrics")
            for metric, value in context.metrics.items():
                sections.append(f"- {metric}: {value:.3f}")
            sections.append("")

        sections.append("### Requirements")
        sections.append("1. Design recovery mechanisms")
        sections.append("2. Minimize failure cascade depth")
        sections.append("3. Preserve critical state")
        sections.append("4. Enable graceful degradation")

        return "\n".join(sections)

    def _build_consistency_check_prompt(self, context: PromptContext) -> str:
        """Build prompt for consistency checking."""
        sections = ["## Task: Design Consistency Verification\n"]

        if context.current_defenses:
            sections.append("### Existing Verification Methods")
            for defense in context.current_defenses:
                sections.append(f"- {defense}")
            sections.append("")

        sections.append("### Requirements")
        sections.append("1. Cross-validate reasoning steps")
        sections.append("2. Detect compromised tool outputs")
        sections.append("3. Verify goal alignment")
        sections.append("4. Minimize verification overhead")
        sections.append("5. Handle partial information")

        return "\n".join(sections)

    def generate_batch(
        self,
        templates: list[PromptTemplate],
        context: PromptContext,
    ) -> list[GeneratedPrompt]:
        """Generate multiple prompts for different templates."""
        return [self.generate(template, context) for template in templates]

    @property
    def generation_count(self) -> int:
        """Total number of prompts generated."""
        return self._generation_count
