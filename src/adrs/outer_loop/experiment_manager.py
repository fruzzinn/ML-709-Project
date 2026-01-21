"""Experiment manager using Best-First Tree Search."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
import heapq

from src.adrs.inner_loop.solution_generator import DefenseSolution
from src.adrs.inner_loop.evaluator import EvaluationResult


@dataclass(order=True)
class ExperimentNode:
    """A node in the experiment tree."""

    priority: float  # Negative fitness for min-heap (we want max)
    node_id: str = field(compare=False)
    solution: DefenseSolution = field(compare=False)
    evaluation: EvaluationResult | None = field(compare=False, default=None)
    parent_id: str | None = field(compare=False, default=None)
    depth: int = field(compare=False, default=0)
    children: list[str] = field(compare=False, default_factory=list)
    created_at: datetime = field(compare=False, default_factory=datetime.utcnow)


class ExperimentManager:
    """Experiment manager using Best-First Tree Search (BFTS).

    Based on AI-Scientist-v2 patterns - explores the solution space
    by expanding the most promising nodes first.
    """

    def __init__(
        self,
        beam_width: int = 5,
        max_depth: int = 10,
    ) -> None:
        self.beam_width = beam_width
        self.max_depth = max_depth

        self._nodes: dict[str, ExperimentNode] = {}
        self._frontier: list[ExperimentNode] = []  # Min-heap (priority = -fitness)
        self._node_counter = 0
        self._best_solution: tuple[DefenseSolution, EvaluationResult] | None = None

    def add_root(
        self,
        solution: DefenseSolution,
        evaluation: EvaluationResult,
    ) -> str:
        """Add a root node to the experiment tree."""
        self._node_counter += 1
        node_id = f"exp_{self._node_counter}"

        node = ExperimentNode(
            priority=-evaluation.fitness_score,
            node_id=node_id,
            solution=solution,
            evaluation=evaluation,
            depth=0,
        )

        self._nodes[node_id] = node
        heapq.heappush(self._frontier, node)

        self._update_best(solution, evaluation)

        return node_id

    def add_child(
        self,
        parent_id: str,
        solution: DefenseSolution,
        evaluation: EvaluationResult,
    ) -> str | None:
        """Add a child node to an existing node."""
        if parent_id not in self._nodes:
            return None

        parent = self._nodes[parent_id]

        # Check depth limit
        if parent.depth >= self.max_depth:
            return None

        self._node_counter += 1
        node_id = f"exp_{self._node_counter}"

        node = ExperimentNode(
            priority=-evaluation.fitness_score,
            node_id=node_id,
            solution=solution,
            evaluation=evaluation,
            parent_id=parent_id,
            depth=parent.depth + 1,
        )

        self._nodes[node_id] = node
        parent.children.append(node_id)
        heapq.heappush(self._frontier, node)

        self._update_best(solution, evaluation)

        return node_id

    def select_for_expansion(self, n: int = 1) -> list[ExperimentNode]:
        """Select top N nodes for expansion (beam search)."""
        selected = []

        # Get top nodes from frontier
        candidates = []
        while self._frontier and len(candidates) < n * 2:
            node = heapq.heappop(self._frontier)
            if node.depth < self.max_depth:
                candidates.append(node)

        # Take top N by fitness
        candidates.sort(key=lambda x: x.priority)
        selected = candidates[:n]

        # Put remaining back
        for node in candidates[n:]:
            heapq.heappush(self._frontier, node)

        return selected

    def _update_best(
        self,
        solution: DefenseSolution,
        evaluation: EvaluationResult,
    ) -> None:
        """Update best solution if this one is better."""
        if self._best_solution is None:
            self._best_solution = (solution, evaluation)
        elif evaluation.fitness_score > self._best_solution[1].fitness_score:
            self._best_solution = (solution, evaluation)

    def get_best(self) -> tuple[DefenseSolution, EvaluationResult] | None:
        """Get the best solution found so far."""
        return self._best_solution

    def get_tree_stats(self) -> dict[str, Any]:
        """Get statistics about the experiment tree."""
        if not self._nodes:
            return {"size": 0}

        depths = [n.depth for n in self._nodes.values()]
        fitness_scores = [
            -n.priority for n in self._nodes.values()
            if n.evaluation is not None
        ]

        return {
            "size": len(self._nodes),
            "frontier_size": len(self._frontier),
            "max_depth_reached": max(depths),
            "best_fitness": max(fitness_scores) if fitness_scores else 0,
            "avg_fitness": sum(fitness_scores) / len(fitness_scores) if fitness_scores else 0,
        }

    def get_path_to_best(self) -> list[ExperimentNode]:
        """Get the path from root to best solution."""
        if not self._best_solution:
            return []

        # Find node with best solution
        best_node = None
        for node in self._nodes.values():
            if node.solution.solution_id == self._best_solution[0].solution_id:
                best_node = node
                break

        if not best_node:
            return []

        # Trace path to root
        path = [best_node]
        current = best_node
        while current.parent_id:
            current = self._nodes[current.parent_id]
            path.append(current)

        return list(reversed(path))
