"""Experiment result reporting."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.evaluation.metrics import ExperimentMetrics


class ExperimentReporter:
    """Reporter for generating experiment results and artifacts."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_metrics(
        self,
        metrics: ExperimentMetrics,
        experiment_name: str,
    ) -> Path:
        """Save metrics to JSON file."""
        filepath = self.output_dir / f"{experiment_name}_metrics.json"

        data = {
            "experiment_name": experiment_name,
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": metrics.to_dict(),
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

        return filepath

    def save_failure_analysis(
        self,
        analysis: dict[str, Any],
        experiment_name: str,
    ) -> Path:
        """Save failure analysis to JSON file."""
        filepath = self.output_dir / f"{experiment_name}_failure_analysis.json"

        with open(filepath, "w") as f:
            json.dump(analysis, f, indent=2, default=str)

        return filepath

    def generate_summary(
        self,
        metrics: ExperimentMetrics,
        config: dict[str, Any],
        failure_analysis: dict[str, Any] | None = None,
    ) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "EXPERIMENT SUMMARY",
            "=" * 60,
            "",
            f"Task Success Rate: {metrics.task_success_rate:.1%}",
            f"  - Completed: {metrics.tasks_completed}",
            f"  - Failed: {metrics.tasks_failed}",
            "",
            f"Safety Score: {metrics.safety_score:.1%}",
            f"  - Violations Blocked: {metrics.safety_violations_blocked}",
            f"  - Violations Attempted: {metrics.safety_violations_attempted}",
            "",
            f"Robustness Score: {metrics.robustness_score:.1%}",
            f"  - Attacks Detected: {metrics.attacks_detected}",
            f"  - Attacks Received: {metrics.attacks_received}",
            f"  - Attacks Blocked: {metrics.attacks_blocked}",
            "",
            f"Latency:",
            f"  - Average: {metrics.average_latency_ms:.1f}ms",
            f"  - P95: {metrics.p95_latency_ms:.1f}ms",
            f"  - Overhead: {metrics.latency_overhead_ms:.1f}ms",
            "",
            f"Tool Calls:",
            f"  - Total: {metrics.total_tool_calls}",
            f"  - Successful: {metrics.successful_tool_calls}",
            f"  - Failed: {metrics.failed_tool_calls}",
            "",
            f"Rollbacks Performed: {metrics.rollbacks_performed}",
        ]

        if failure_analysis:
            lines.extend([
                "",
                "Failure Analysis:",
                f"  - Total Failures: {failure_analysis.get('total_failures', 0)}",
                f"  - Max Cascade Depth: {failure_analysis.get('cascade_depth', {}).get('max', 0)}",
                f"  - Avg Cascade Depth: {failure_analysis.get('cascade_depth', {}).get('average', 0):.2f}",
            ])

            vulnerable = failure_analysis.get("vulnerable_components", {})
            if vulnerable:
                lines.append("  - Most Vulnerable Components:")
                for comp, count in list(vulnerable.items())[:3]:
                    lines.append(f"      {comp}: {count} failures")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    def save_summary(
        self,
        summary: str,
        experiment_name: str,
    ) -> Path:
        """Save summary to text file."""
        filepath = self.output_dir / f"{experiment_name}_summary.txt"

        with open(filepath, "w") as f:
            f.write(summary)

        return filepath
