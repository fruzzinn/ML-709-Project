"""MAP-Elites quality-diversity selector for solution selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.adrs.inner_loop.evaluator import EvaluationResult
from src.adrs.inner_loop.solution_generator import DefenseSolution


@dataclass
class EliteCell:
    """A cell in the MAP-Elites archive."""

    solution: DefenseSolution
    evaluation: EvaluationResult
    behavior_descriptor: tuple[float, ...]


class MAPElitesSelector:
    """MAP-Elites algorithm for quality-diversity selection.

    Based on OpenEvolve patterns - maintains a diverse archive
    of high-performing solutions across different behavioral niches.
    """

    def __init__(
        self,
        dimensions: list[str],
        resolution: int = 10,
        max_archive_size: int = 100,
    ) -> None:
        self.dimensions = dimensions
        self.resolution = resolution
        self.max_archive_size = max_archive_size
        self._archive: dict[tuple[int, ...], EliteCell] = {}

    def add_solution(
        self,
        solution: DefenseSolution,
        evaluation: EvaluationResult,
    ) -> bool:
        """Add a solution to the archive if it's an elite."""
        # Calculate behavior descriptor
        descriptor = self._calculate_behavior(evaluation)
        cell_key = self._discretize(descriptor)

        # Check if this cell is empty or if new solution is better
        if (
            cell_key not in self._archive
            or evaluation.fitness_score > self._archive[cell_key].evaluation.fitness_score
        ):
            self._archive[cell_key] = EliteCell(
                solution=solution,
                evaluation=evaluation,
                behavior_descriptor=descriptor,
            )
            return True

        return False

    def _calculate_behavior(self, evaluation: EvaluationResult) -> tuple[float, ...]:
        """Calculate behavior descriptor from evaluation results."""
        # Default dimensions: safety vs latency trade-off
        return (
            evaluation.safety_score,
            1 - min(evaluation.latency_overhead_ms / 200, 1.0),
        )

    def _discretize(self, descriptor: tuple[float, ...]) -> tuple[int, ...]:
        """Discretize continuous behavior descriptor to grid cell."""
        return tuple(min(int(d * self.resolution), self.resolution - 1) for d in descriptor)

    def select_for_mutation(self, n: int = 1) -> list[DefenseSolution]:
        """Select solutions for mutation (uniform random from archive)."""
        import random

        if not self._archive:
            return []

        cells = list(self._archive.values())
        selected = random.sample(cells, min(n, len(cells)))
        return [c.solution for c in selected]

    def get_best(self, n: int = 5) -> list[tuple[DefenseSolution, EvaluationResult]]:
        """Get top N solutions by fitness."""
        sorted_cells = sorted(
            self._archive.values(),
            key=lambda c: c.evaluation.fitness_score,
            reverse=True,
        )
        return [(c.solution, c.evaluation) for c in sorted_cells[:n]]

    def get_archive_stats(self) -> dict[str, Any]:
        """Get statistics about the archive."""
        if not self._archive:
            return {"size": 0, "coverage": 0}

        fitness_scores = [c.evaluation.fitness_score for c in self._archive.values()]
        total_cells = self.resolution ** len(self.dimensions)

        return {
            "size": len(self._archive),
            "coverage": len(self._archive) / total_cells,
            "max_fitness": max(fitness_scores),
            "avg_fitness": sum(fitness_scores) / len(fitness_scores),
            "min_fitness": min(fitness_scores),
        }

    def clear(self) -> None:
        """Clear the archive."""
        self._archive.clear()
