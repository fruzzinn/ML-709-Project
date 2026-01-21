"""Analyzer for ADRS experiment results."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog

from src.adrs.inner_loop.storage import SolutionStorage, StoredSolution

logger = structlog.get_logger()


@dataclass
class AnalysisResult:
    """Result of an analysis."""

    name: str
    timestamp: datetime
    summary: dict[str, Any]
    details: dict[str, Any]
    recommendations: list[str]
    visualizations: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrendAnalysis:
    """Analysis of trends over generations."""

    metric: str
    values: list[float]
    generations: list[int]
    trend: str  # "improving", "declining", "stable"
    slope: float
    significance: float


class ExperimentAnalyzer:
    """Analyzer for ADRS experiment results.

    Provides:
    - Statistical analysis of solution performance
    - Trend analysis across generations
    - Comparative analysis between solutions
    - Recommendations for next experiments
    """

    def __init__(self, storage: SolutionStorage | None = None) -> None:
        self.storage = storage
        self._log = logger.bind(component="experiment_analyzer")
        self._analyses: list[AnalysisResult] = []

    def analyze_generation(
        self,
        generation: int,
        solutions: list[StoredSolution] | None = None,
    ) -> AnalysisResult:
        """Analyze a specific generation's results."""
        if solutions is None and self.storage:
            solutions = self.storage.get_all_solutions(generation=generation)

        if not solutions:
            return AnalysisResult(
                name=f"generation_{generation}_analysis",
                timestamp=datetime.utcnow(),
                summary={"error": "No solutions found"},
                details={},
                recommendations=["Generate more solutions"],
            )

        # Compute statistics
        fitness_values = [s.fitness for s in solutions]
        metrics_data = self._aggregate_metrics(solutions)

        summary = {
            "generation": generation,
            "num_solutions": len(solutions),
            "fitness": {
                "mean": float(np.mean(fitness_values)),
                "std": float(np.std(fitness_values)),
                "min": float(np.min(fitness_values)),
                "max": float(np.max(fitness_values)),
                "median": float(np.median(fitness_values)),
            },
        }

        # Add aggregated metrics
        for metric_name, values in metrics_data.items():
            if values:
                summary[f"metric_{metric_name}"] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                }

        # Analyze behavior diversity
        behavior_analysis = self._analyze_behavior_diversity(solutions)

        details = {
            "solutions": [
                {
                    "id": s.id,
                    "name": s.name,
                    "fitness": s.fitness,
                    "behavior": s.behavior_descriptor,
                }
                for s in sorted(solutions, key=lambda x: x.fitness, reverse=True)[:10]
            ],
            "behavior_diversity": behavior_analysis,
        }

        # Generate recommendations
        recommendations = self._generate_recommendations(summary, behavior_analysis)

        result = AnalysisResult(
            name=f"generation_{generation}_analysis",
            timestamp=datetime.utcnow(),
            summary=summary,
            details=details,
            recommendations=recommendations,
        )

        self._analyses.append(result)
        self._log.info(
            "Generation analysis complete",
            generation=generation,
            num_solutions=len(solutions),
            best_fitness=summary["fitness"]["max"],
        )

        return result

    def analyze_trends(
        self,
        metric: str = "fitness",
        window_size: int = 5,
    ) -> TrendAnalysis:
        """Analyze trends across generations."""
        if not self.storage:
            raise ValueError("Storage required for trend analysis")

        all_solutions = self.storage.get_all_solutions()
        if not all_solutions:
            return TrendAnalysis(
                metric=metric,
                values=[],
                generations=[],
                trend="unknown",
                slope=0.0,
                significance=0.0,
            )

        # Group by generation
        by_generation: dict[int, list[float]] = defaultdict(list)
        for solution in all_solutions:
            if metric == "fitness":
                by_generation[solution.generation].append(solution.fitness)
            elif metric in solution.metrics:
                by_generation[solution.generation].append(solution.metrics[metric])

        # Compute generation averages
        generations = sorted(by_generation.keys())
        values = [np.mean(by_generation[g]) for g in generations]

        # Compute trend
        if len(values) >= 2:
            # Simple linear regression
            x = np.array(generations)
            y = np.array(values)
            slope = np.polyfit(x, y, 1)[0]

            # Determine trend direction
            if slope > 0.01:
                trend = "improving"
            elif slope < -0.01:
                trend = "declining"
            else:
                trend = "stable"

            # Compute significance (R-squared)
            if len(values) > 2:
                correlation = np.corrcoef(x, y)[0, 1]
                significance = correlation ** 2
            else:
                significance = 0.0
        else:
            slope = 0.0
            trend = "insufficient_data"
            significance = 0.0

        return TrendAnalysis(
            metric=metric,
            values=values,
            generations=generations,
            trend=trend,
            slope=float(slope),
            significance=float(significance),
        )

    def compare_solutions(
        self,
        solution_ids: list[int],
    ) -> dict[str, Any]:
        """Compare multiple solutions."""
        if not self.storage:
            raise ValueError("Storage required for solution comparison")

        solutions = [self.storage.get_solution(sid) for sid in solution_ids]
        solutions = [s for s in solutions if s is not None]

        if len(solutions) < 2:
            return {"error": "Need at least 2 solutions to compare"}

        comparison = {
            "solutions": [],
            "fitness_ranking": [],
            "metric_comparison": defaultdict(dict),
            "behavior_distances": [],
        }

        # Basic info and rankings
        for sol in sorted(solutions, key=lambda x: x.fitness, reverse=True):
            comparison["solutions"].append({
                "id": sol.id,
                "name": sol.name,
                "fitness": sol.fitness,
                "generation": sol.generation,
            })
            comparison["fitness_ranking"].append(sol.id)

        # Compare metrics
        all_metrics: set[str] = set()
        for sol in solutions:
            all_metrics.update(sol.metrics.keys())

        for metric in all_metrics:
            for sol in solutions:
                comparison["metric_comparison"][metric][sol.id] = sol.metrics.get(metric, None)

        # Compute behavior distances
        for i, sol1 in enumerate(solutions):
            for sol2 in solutions[i + 1:]:
                distance = self._behavior_distance(
                    sol1.behavior_descriptor,
                    sol2.behavior_descriptor,
                )
                comparison["behavior_distances"].append({
                    "solution_1": sol1.id,
                    "solution_2": sol2.id,
                    "distance": distance,
                })

        return dict(comparison)

    def analyze_attack_effectiveness(self) -> dict[str, Any]:
        """Analyze effectiveness against different attack types."""
        if not self.storage:
            raise ValueError("Storage required for attack analysis")

        # Get all evaluations
        all_solutions = self.storage.get_all_solutions()
        attack_data: dict[str, list[dict[str, float]]] = defaultdict(list)

        for solution in all_solutions:
            evaluations = self.storage.get_evaluations(solution.id)
            for eval_data in evaluations:
                attack_type = eval_data.get("attack_type", "unknown")
                attack_data[attack_type].append({
                    "success_rate": eval_data.get("success_rate", 0),
                    "detection_rate": eval_data.get("detection_rate", 0),
                    "latency_ms": eval_data.get("latency_ms", 0),
                })

        analysis = {}
        for attack_type, data in attack_data.items():
            if data:
                analysis[attack_type] = {
                    "num_evaluations": len(data),
                    "avg_success_rate": np.mean([d["success_rate"] for d in data]),
                    "avg_detection_rate": np.mean([d["detection_rate"] for d in data]),
                    "avg_latency_ms": np.mean([d["latency_ms"] for d in data]),
                }

        return analysis

    def generate_report(
        self,
        output_path: Path | None = None,
        format: str = "json",
    ) -> str:
        """Generate a comprehensive analysis report."""
        report = {
            "generated_at": datetime.utcnow().isoformat(),
            "summary": {},
            "trends": {},
            "attack_analysis": {},
            "recommendations": [],
        }

        if self.storage:
            stats = self.storage.get_statistics()
            report["summary"] = {
                "total_solutions": stats["total_solutions"],
                "total_evaluations": stats["total_evaluations"],
                "max_generation": stats["max_generation"],
                "best_fitness": stats["best_fitness"],
                "average_fitness": stats["average_fitness"],
            }

            # Add trend analysis
            fitness_trend = self.analyze_trends("fitness")
            report["trends"]["fitness"] = {
                "direction": fitness_trend.trend,
                "slope": fitness_trend.slope,
                "significance": fitness_trend.significance,
            }

            # Add attack analysis
            report["attack_analysis"] = self.analyze_attack_effectiveness()

            # Top solutions
            top_solutions = self.storage.get_top_solutions(limit=5)
            report["top_solutions"] = [
                {
                    "id": s.id,
                    "name": s.name,
                    "fitness": s.fitness,
                    "generation": s.generation,
                }
                for s in top_solutions
            ]

        # Aggregate recommendations from analyses
        all_recommendations: set[str] = set()
        for analysis in self._analyses[-10:]:  # Last 10 analyses
            all_recommendations.update(analysis.recommendations)
        report["recommendations"] = list(all_recommendations)

        # Format output
        if format == "json":
            report_str = json.dumps(report, indent=2, default=str)
        else:
            report_str = self._format_text_report(report)

        # Save if path provided
        if output_path:
            output_path.write_text(report_str)
            self._log.info("Report saved", path=str(output_path))

        return report_str

    def _aggregate_metrics(
        self,
        solutions: list[StoredSolution],
    ) -> dict[str, list[float]]:
        """Aggregate metrics across solutions."""
        metrics_data: dict[str, list[float]] = defaultdict(list)
        for solution in solutions:
            for metric_name, value in solution.metrics.items():
                if isinstance(value, (int, float)):
                    metrics_data[metric_name].append(value)
        return dict(metrics_data)

    def _analyze_behavior_diversity(
        self,
        solutions: list[StoredSolution],
    ) -> dict[str, Any]:
        """Analyze diversity of behavior descriptors."""
        if not solutions:
            return {"diversity_score": 0, "num_clusters": 0}

        behaviors = [s.behavior_descriptor for s in solutions]

        # Compute pairwise distances
        distances = []
        for i, b1 in enumerate(behaviors):
            for b2 in behaviors[i + 1:]:
                distances.append(self._behavior_distance(b1, b2))

        if distances:
            diversity_score = np.mean(distances)
            max_distance = np.max(distances)
            min_distance = np.min(distances)
        else:
            diversity_score = 0
            max_distance = 0
            min_distance = 0

        return {
            "diversity_score": float(diversity_score),
            "max_distance": float(max_distance),
            "min_distance": float(min_distance),
            "num_solutions": len(solutions),
        }

    def _generate_recommendations(
        self,
        summary: dict[str, Any],
        behavior_analysis: dict[str, Any],
    ) -> list[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        fitness_data = summary.get("fitness", {})

        # Check fitness variance
        if fitness_data.get("std", 0) < 0.05:
            recommendations.append(
                "Low fitness variance - consider increasing mutation rate or exploring new regions"
            )

        # Check diversity
        diversity = behavior_analysis.get("diversity_score", 0)
        if diversity < 0.3:
            recommendations.append(
                "Low behavior diversity - prioritize exploration over exploitation"
            )

        # Check for improvement potential
        if fitness_data.get("max", 0) < 0.7:
            recommendations.append(
                "Best fitness below 0.7 - consider new defense architectures"
            )

        # Check sample size
        if summary.get("num_solutions", 0) < 10:
            recommendations.append(
                "Small sample size - generate more solutions for reliable statistics"
            )

        return recommendations

    @staticmethod
    def _behavior_distance(
        b1: tuple[float, ...],
        b2: tuple[float, ...],
    ) -> float:
        """Calculate Euclidean distance between behavior descriptors."""
        if len(b1) != len(b2):
            return float("inf")
        return float(np.sqrt(sum((a - b) ** 2 for a, b in zip(b1, b2))))

    def _format_text_report(self, report: dict[str, Any]) -> str:
        """Format report as readable text."""
        lines = [
            "=" * 60,
            "ADRS EXPERIMENT ANALYSIS REPORT",
            f"Generated: {report['generated_at']}",
            "=" * 60,
            "",
            "SUMMARY",
            "-" * 40,
        ]

        for key, value in report.get("summary", {}).items():
            lines.append(f"  {key}: {value}")

        lines.extend([
            "",
            "TRENDS",
            "-" * 40,
        ])

        for metric, data in report.get("trends", {}).items():
            lines.append(f"  {metric}:")
            for key, value in data.items():
                lines.append(f"    {key}: {value}")

        lines.extend([
            "",
            "TOP SOLUTIONS",
            "-" * 40,
        ])

        for sol in report.get("top_solutions", []):
            lines.append(f"  #{sol['id']} {sol['name']}: fitness={sol['fitness']:.4f}")

        lines.extend([
            "",
            "RECOMMENDATIONS",
            "-" * 40,
        ])

        for rec in report.get("recommendations", []):
            lines.append(f"  - {rec}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)
