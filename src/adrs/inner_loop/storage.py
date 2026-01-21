"""Storage system for ADRS solutions using SQLite."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import structlog

logger = structlog.get_logger()


@dataclass
class StoredSolution:
    """A solution stored in the database."""

    id: int
    name: str
    description: str
    code: str
    fitness: float
    metrics: dict[str, float]
    behavior_descriptor: tuple[float, ...]
    generation: int
    parent_id: int | None
    created_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


class SolutionStorage:
    """SQLite-based storage for ADRS solutions.

    Provides persistent storage for:
    - Generated defense solutions
    - Fitness scores and metrics
    - Behavior descriptors for MAP-Elites
    - Solution lineage tracking
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS solutions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        description TEXT,
        code TEXT NOT NULL,
        fitness REAL NOT NULL,
        metrics TEXT NOT NULL,
        behavior_descriptor TEXT NOT NULL,
        generation INTEGER NOT NULL,
        parent_id INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        metadata TEXT,
        FOREIGN KEY (parent_id) REFERENCES solutions(id)
    );

    CREATE INDEX IF NOT EXISTS idx_fitness ON solutions(fitness DESC);
    CREATE INDEX IF NOT EXISTS idx_generation ON solutions(generation);
    CREATE INDEX IF NOT EXISTS idx_behavior ON solutions(behavior_descriptor);

    CREATE TABLE IF NOT EXISTS experiments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        config TEXT NOT NULL,
        started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        completed_at TIMESTAMP,
        status TEXT DEFAULT 'running',
        results TEXT
    );

    CREATE TABLE IF NOT EXISTS evaluations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        solution_id INTEGER NOT NULL,
        experiment_id INTEGER,
        attack_type TEXT NOT NULL,
        success_rate REAL,
        detection_rate REAL,
        latency_ms REAL,
        evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        details TEXT,
        FOREIGN KEY (solution_id) REFERENCES solutions(id),
        FOREIGN KEY (experiment_id) REFERENCES experiments(id)
    );

    CREATE INDEX IF NOT EXISTS idx_eval_solution ON evaluations(solution_id);
    """

    def __init__(self, db_path: str | Path = "adrs_solutions.db") -> None:
        self.db_path = Path(db_path)
        self._log = logger.bind(component="solution_storage", db=str(db_path))
        self._init_database()

    def _init_database(self) -> None:
        """Initialize the database schema."""
        with self._get_connection() as conn:
            conn.executescript(self.SCHEMA)
            conn.commit()
        self._log.info("Database initialized")

    @contextmanager
    def _get_connection(self) -> Iterator[sqlite3.Connection]:
        """Get a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def store_solution(
        self,
        name: str,
        description: str,
        code: str,
        fitness: float,
        metrics: dict[str, float],
        behavior_descriptor: tuple[float, ...],
        generation: int,
        parent_id: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Store a new solution and return its ID."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO solutions
                (name, description, code, fitness, metrics, behavior_descriptor,
                 generation, parent_id, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    name,
                    description,
                    code,
                    fitness,
                    json.dumps(metrics),
                    json.dumps(behavior_descriptor),
                    generation,
                    parent_id,
                    json.dumps(metadata or {}),
                ),
            )
            conn.commit()
            solution_id = cursor.lastrowid

        self._log.debug("Stored solution", id=solution_id, name=name, fitness=fitness)
        return solution_id  # type: ignore

    def get_solution(self, solution_id: int) -> StoredSolution | None:
        """Retrieve a solution by ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM solutions WHERE id = ?", (solution_id,)
            ).fetchone()

        if not row:
            return None

        return self._row_to_solution(row)

    def get_top_solutions(
        self,
        limit: int = 10,
        generation: int | None = None,
    ) -> list[StoredSolution]:
        """Get top solutions by fitness."""
        with self._get_connection() as conn:
            if generation is not None:
                rows = conn.execute(
                    """
                    SELECT * FROM solutions
                    WHERE generation = ?
                    ORDER BY fitness DESC
                    LIMIT ?
                    """,
                    (generation, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM solutions ORDER BY fitness DESC LIMIT ?",
                    (limit,),
                ).fetchall()

        return [self._row_to_solution(row) for row in rows]

    def get_solutions_by_behavior(
        self,
        behavior_descriptor: tuple[float, ...],
        tolerance: float = 0.1,
    ) -> list[StoredSolution]:
        """Find solutions with similar behavior descriptors."""
        all_solutions = self.get_all_solutions()

        similar = []
        for sol in all_solutions:
            if self._behavior_distance(sol.behavior_descriptor, behavior_descriptor) <= tolerance:
                similar.append(sol)

        return similar

    def get_all_solutions(self, generation: int | None = None) -> list[StoredSolution]:
        """Get all solutions, optionally filtered by generation."""
        with self._get_connection() as conn:
            if generation is not None:
                rows = conn.execute(
                    "SELECT * FROM solutions WHERE generation = ?", (generation,)
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM solutions").fetchall()

        return [self._row_to_solution(row) for row in rows]

    def get_solution_lineage(self, solution_id: int) -> list[StoredSolution]:
        """Get the ancestry of a solution."""
        lineage = []
        current_id: int | None = solution_id

        while current_id is not None:
            solution = self.get_solution(current_id)
            if solution:
                lineage.append(solution)
                current_id = solution.parent_id
            else:
                break

        return lineage

    def update_fitness(self, solution_id: int, fitness: float, metrics: dict[str, float]) -> None:
        """Update the fitness and metrics of a solution."""
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE solutions
                SET fitness = ?, metrics = ?
                WHERE id = ?
                """,
                (fitness, json.dumps(metrics), solution_id),
            )
            conn.commit()

        self._log.debug("Updated solution fitness", id=solution_id, fitness=fitness)

    def store_evaluation(
        self,
        solution_id: int,
        attack_type: str,
        success_rate: float,
        detection_rate: float,
        latency_ms: float,
        experiment_id: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> int:
        """Store an evaluation result."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO evaluations
                (solution_id, experiment_id, attack_type, success_rate,
                 detection_rate, latency_ms, details)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    solution_id,
                    experiment_id,
                    attack_type,
                    success_rate,
                    detection_rate,
                    latency_ms,
                    json.dumps(details or {}),
                ),
            )
            conn.commit()
            return cursor.lastrowid  # type: ignore

    def get_evaluations(self, solution_id: int) -> list[dict[str, Any]]:
        """Get all evaluations for a solution."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM evaluations WHERE solution_id = ?", (solution_id,)
            ).fetchall()

        return [dict(row) for row in rows]

    def start_experiment(self, name: str, config: dict[str, Any]) -> int:
        """Start a new experiment and return its ID."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "INSERT INTO experiments (name, config) VALUES (?, ?)",
                (name, json.dumps(config)),
            )
            conn.commit()
            return cursor.lastrowid  # type: ignore

    def complete_experiment(
        self,
        experiment_id: int,
        status: str = "completed",
        results: dict[str, Any] | None = None,
    ) -> None:
        """Mark an experiment as completed."""
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE experiments
                SET completed_at = CURRENT_TIMESTAMP, status = ?, results = ?
                WHERE id = ?
                """,
                (status, json.dumps(results or {}), experiment_id),
            )
            conn.commit()

    def get_statistics(self) -> dict[str, Any]:
        """Get storage statistics."""
        with self._get_connection() as conn:
            total_solutions = conn.execute(
                "SELECT COUNT(*) FROM solutions"
            ).fetchone()[0]

            total_evaluations = conn.execute(
                "SELECT COUNT(*) FROM evaluations"
            ).fetchone()[0]

            total_experiments = conn.execute(
                "SELECT COUNT(*) FROM experiments"
            ).fetchone()[0]

            max_generation = conn.execute(
                "SELECT MAX(generation) FROM solutions"
            ).fetchone()[0] or 0

            avg_fitness = conn.execute(
                "SELECT AVG(fitness) FROM solutions"
            ).fetchone()[0] or 0

            best_fitness = conn.execute(
                "SELECT MAX(fitness) FROM solutions"
            ).fetchone()[0] or 0

        return {
            "total_solutions": total_solutions,
            "total_evaluations": total_evaluations,
            "total_experiments": total_experiments,
            "max_generation": max_generation,
            "average_fitness": avg_fitness,
            "best_fitness": best_fitness,
        }

    def _row_to_solution(self, row: sqlite3.Row) -> StoredSolution:
        """Convert a database row to a StoredSolution."""
        return StoredSolution(
            id=row["id"],
            name=row["name"],
            description=row["description"] or "",
            code=row["code"],
            fitness=row["fitness"],
            metrics=json.loads(row["metrics"]),
            behavior_descriptor=tuple(json.loads(row["behavior_descriptor"])),
            generation=row["generation"],
            parent_id=row["parent_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    @staticmethod
    def _behavior_distance(
        b1: tuple[float, ...],
        b2: tuple[float, ...],
    ) -> float:
        """Calculate Euclidean distance between behavior descriptors."""
        if len(b1) != len(b2):
            return float("inf")
        return sum((a - b) ** 2 for a, b in zip(b1, b2)) ** 0.5

    def clear(self) -> None:
        """Clear all data (use with caution)."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM evaluations")
            conn.execute("DELETE FROM solutions")
            conn.execute("DELETE FROM experiments")
            conn.commit()
        self._log.warning("Storage cleared")
