"""Solution generator using LLM for defense generation.

Supports both LLMClient (OpenAI-compatible) and BaseProvider (Anthropic, etc.)
for flexibility in model selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable
from datetime import datetime


@runtime_checkable
class LLMInterface(Protocol):
    """Protocol for LLM clients - supports both LLMClient and BaseProvider."""

    async def chat(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> Any:
        """Chat completion interface."""
        ...


@dataclass
class DefenseSolution:
    """A generated defense solution."""

    solution_id: str
    name: str
    description: str
    defense_type: str
    parameters: dict[str, Any] = field(default_factory=dict)
    code_snippet: str | None = None
    fitness_score: float = 0.0
    generation: int = 0
    parent_id: str | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)


class SolutionGenerator:
    """Generator for defense solutions using LLM.

    Uses prompt engineering to generate novel defense mechanisms
    based on attack patterns and failure analysis.
    """

    GENERATION_PROMPT = """You are an AI security researcher designing defenses for agentic AI systems.

Given the following attack scenario and failure patterns, generate a novel defense mechanism.

Attack Scenario:
{attack_scenario}

Observed Failures:
{failure_patterns}

Current Defenses (to improve upon):
{current_defenses}

Generate a defense solution with:
1. Name: A descriptive name for the defense
2. Description: What it does and why
3. Defense Type: One of [verification, redundancy, rollback, anomaly_detection, consistency]
4. Parameters: Key configuration parameters
5. Implementation Hint: How it should be implemented

Format your response as JSON."""

    def __init__(self, llm_client: LLMInterface) -> None:
        """Initialize the solution generator.

        Args:
            llm_client: Any LLM client implementing the LLMInterface protocol.
                       Supports LLMClient (vLLM/OpenAI) or BaseProvider (Anthropic).
        """
        self.llm = llm_client
        self._solution_counter = 0

    async def generate(
        self,
        attack_scenario: str,
        failure_patterns: list[str],
        current_defenses: list[str],
        num_solutions: int = 3,
    ) -> list[DefenseSolution]:
        """Generate defense solutions."""
        solutions = []

        for _ in range(num_solutions):
            prompt = self.GENERATION_PROMPT.format(
                attack_scenario=attack_scenario,
                failure_patterns="\n".join(f"- {f}" for f in failure_patterns),
                current_defenses="\n".join(f"- {d}" for d in current_defenses),
            )

            response = await self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,  # Higher for creativity
            )

            solution = self._parse_solution(response.content)
            if solution:
                solutions.append(solution)

        return solutions

    def _parse_solution(self, response: str) -> DefenseSolution | None:
        """Parse LLM response into a DefenseSolution."""
        import json

        try:
            # Try to extract JSON from response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(response[start:end])

                self._solution_counter += 1
                return DefenseSolution(
                    solution_id=f"sol_{self._solution_counter}",
                    name=data.get("name", "Unnamed Defense"),
                    description=data.get("description", ""),
                    defense_type=data.get("defense_type", "verification"),
                    parameters=data.get("parameters", {}),
                    code_snippet=data.get("implementation_hint"),
                )
        except json.JSONDecodeError:
            pass

        return None

    async def mutate(
        self,
        solution: DefenseSolution,
        mutation_strength: float = 0.3,
    ) -> DefenseSolution:
        """Mutate an existing solution to create a variant."""
        prompt = f"""Modify this defense mechanism to improve its effectiveness.

Current Defense:
- Name: {solution.name}
- Description: {solution.description}
- Type: {solution.defense_type}
- Parameters: {solution.parameters}

Mutation strength: {mutation_strength} (0=minor tweaks, 1=major changes)

Generate a modified version that:
1. Keeps the core concept but improves it
2. Adjusts parameters for better performance
3. Addresses potential weaknesses

Format your response as JSON."""

        response = await self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6 + mutation_strength * 0.3,
        )

        mutated = self._parse_solution(response.content)
        if mutated:
            mutated.parent_id = solution.solution_id
            mutated.generation = solution.generation + 1
        return mutated or solution
