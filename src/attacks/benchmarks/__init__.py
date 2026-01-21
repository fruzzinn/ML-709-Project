"""Benchmark integrations for adversarial agent evaluation.

Supported benchmarks:
- BAD-ACTS: Harmful action detection
- TAMAS: Multi-agent adversarial security
- AgentHarm: Comprehensive safety evaluation
"""

from src.attacks.benchmarks.base import Benchmark, BenchmarkInstance, BenchmarkLoader

__all__ = ["Benchmark", "BenchmarkInstance", "BenchmarkLoader"]
