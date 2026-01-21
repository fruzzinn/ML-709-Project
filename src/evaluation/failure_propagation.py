"""Failure propagation analysis using graph-based tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import networkx as nx


@dataclass
class FailureNode:
    """A node in the failure propagation graph."""

    node_id: str
    node_type: str  # tool_error, anomaly, rollback, cascade
    component: str  # tool name or component
    timestamp: datetime = field(default_factory=datetime.utcnow)
    severity: float = 1.0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class FailureEdge:
    """An edge representing failure propagation."""

    source_id: str
    target_id: str
    propagation_type: str  # direct, indirect, cascade
    delay_ms: float = 0.0


class FailurePropagationAnalyzer:
    """Analyzer for tracking and understanding failure cascades.

    Uses NetworkX to build and analyze failure propagation graphs.
    """

    def __init__(self) -> None:
        self.graph = nx.DiGraph()
        self._node_counter = 0
        self._failure_nodes: dict[str, FailureNode] = {}

    def add_failure(
        self,
        node_type: str,
        component: str,
        severity: float = 1.0,
        details: dict[str, Any] | None = None,
        caused_by: str | None = None,
    ) -> str:
        """Add a failure node to the graph."""
        self._node_counter += 1
        node_id = f"f_{self._node_counter}"

        node = FailureNode(
            node_id=node_id,
            node_type=node_type,
            component=component,
            severity=severity,
            details=details or {},
        )

        self._failure_nodes[node_id] = node
        self.graph.add_node(
            node_id,
            node_type=node_type,
            component=component,
            severity=severity,
        )

        # Add edge from cause if specified
        if caused_by and caused_by in self._failure_nodes:
            self.graph.add_edge(
                caused_by,
                node_id,
                propagation_type="direct",
            )

        return node_id

    def add_cascade(
        self,
        source_id: str,
        target_id: str,
        propagation_type: str = "cascade",
        delay_ms: float = 0.0,
    ) -> None:
        """Add a cascade edge between failures."""
        if source_id in self.graph and target_id in self.graph:
            self.graph.add_edge(
                source_id,
                target_id,
                propagation_type=propagation_type,
                delay_ms=delay_ms,
            )

    def get_cascade_depth(self, node_id: str) -> int:
        """Get the cascade depth from a failure node."""
        if node_id not in self.graph:
            return 0

        # Find all paths from this node
        descendants = nx.descendants(self.graph, node_id)
        if not descendants:
            return 0

        # Calculate longest path
        max_depth = 0
        for target in descendants:
            try:
                paths = list(nx.all_simple_paths(self.graph, node_id, target))
                for path in paths:
                    max_depth = max(max_depth, len(path) - 1)
            except nx.NetworkXNoPath:
                continue

        return max_depth

    def get_root_causes(self) -> list[FailureNode]:
        """Get all root cause failures (nodes with no incoming edges)."""
        roots = [n for n in self.graph.nodes() if self.graph.in_degree(n) == 0]
        return [self._failure_nodes[r] for r in roots if r in self._failure_nodes]

    def get_most_impactful(self, top_n: int = 5) -> list[tuple[FailureNode, int]]:
        """Get the most impactful failures by cascade size."""
        impact_scores = []

        for node_id in self.graph.nodes():
            descendants = len(nx.descendants(self.graph, node_id))
            if node_id in self._failure_nodes:
                impact_scores.append((self._failure_nodes[node_id], descendants))

        impact_scores.sort(key=lambda x: x[1], reverse=True)
        return impact_scores[:top_n]

    def get_vulnerable_components(self) -> dict[str, int]:
        """Identify which components are most vulnerable."""
        component_failures: dict[str, int] = {}

        for node_id, data in self.graph.nodes(data=True):
            component = data.get("component", "unknown")
            component_failures[component] = component_failures.get(component, 0) + 1

        return dict(sorted(component_failures.items(), key=lambda x: x[1], reverse=True))

    def analyze(self) -> dict[str, Any]:
        """Perform comprehensive failure analysis."""
        if not self.graph.nodes():
            return {
                "total_failures": 0,
                "cascade_depth": {"max": 0, "average": 0},
                "root_causes": [],
                "vulnerable_components": {},
                "most_impactful": [],
            }

        # Calculate cascade depths
        cascade_depths = [self.get_cascade_depth(n) for n in self.graph.nodes()]
        max_depth = max(cascade_depths) if cascade_depths else 0
        avg_depth = sum(cascade_depths) / len(cascade_depths) if cascade_depths else 0

        # Get root causes
        root_causes = self.get_root_causes()

        # Get vulnerable components
        vulnerable = self.get_vulnerable_components()

        # Get most impactful
        impactful = self.get_most_impactful()

        return {
            "total_failures": len(self.graph.nodes()),
            "total_cascades": len(self.graph.edges()),
            "cascade_depth": {
                "max": max_depth,
                "average": avg_depth,
            },
            "root_causes": [
                {"id": r.node_id, "type": r.node_type, "component": r.component}
                for r in root_causes
            ],
            "vulnerable_components": vulnerable,
            "most_impactful": [
                {"id": n.node_id, "component": n.component, "impact": i}
                for n, i in impactful
            ],
        }

    def to_graph_dict(self) -> dict[str, Any]:
        """Export graph as dictionary for visualization."""
        return {
            "nodes": [
                {
                    "id": n,
                    **self.graph.nodes[n],
                }
                for n in self.graph.nodes()
            ],
            "edges": [
                {
                    "source": u,
                    "target": v,
                    **self.graph.edges[u, v],
                }
                for u, v in self.graph.edges()
            ],
        }

    def reset(self) -> None:
        """Reset the analyzer."""
        self.graph.clear()
        self._node_counter = 0
        self._failure_nodes.clear()
