"""LLM ensemble for multi-model inference."""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

from src.llm.client import LLMClient, LLMResponse

logger = structlog.get_logger()


class AggregationStrategy(str, Enum):
    """Strategy for aggregating ensemble responses."""

    MAJORITY_VOTE = "majority_vote"  # Most common response
    BEST_CONFIDENCE = "best_confidence"  # Highest confidence response
    WEIGHTED_AVERAGE = "weighted_average"  # Weighted by model performance
    RANDOM = "random"  # Random selection
    FIRST_SUCCESS = "first_success"  # First non-error response
    CONSENSUS = "consensus"  # Only if all agree


@dataclass
class EnsembleMember:
    """A member of the LLM ensemble."""

    client: LLMClient
    name: str
    weight: float = 1.0
    enabled: bool = True
    success_count: int = 0
    failure_count: int = 0
    total_latency_ms: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.5

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency."""
        total = self.success_count + self.failure_count
        return self.total_latency_ms / total if total > 0 else 0.0


@dataclass
class EnsembleResponse:
    """Response from the ensemble."""

    content: str
    tool_calls: list[dict[str, Any]] | None = None
    aggregation_strategy: AggregationStrategy = AggregationStrategy.MAJORITY_VOTE
    num_responses: int = 0
    agreement_score: float = 0.0
    individual_responses: list[LLMResponse] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class LLMEnsemble:
    """Ensemble of LLM clients for robust inference.

    Provides:
    - Multiple model queries for redundancy
    - Various aggregation strategies
    - Automatic fallback on failures
    - Performance tracking per model
    """

    def __init__(
        self,
        members: list[EnsembleMember] | None = None,
        default_strategy: AggregationStrategy = AggregationStrategy.MAJORITY_VOTE,
        min_responses: int = 1,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.members = members or []
        self.default_strategy = default_strategy
        self.min_responses = min_responses
        self.timeout_seconds = timeout_seconds
        self._log = logger.bind(component="llm_ensemble")

    def add_member(
        self,
        client: LLMClient,
        name: str,
        weight: float = 1.0,
    ) -> None:
        """Add a new member to the ensemble."""
        member = EnsembleMember(client=client, name=name, weight=weight)
        self.members.append(member)
        self._log.info("Added ensemble member", name=name, weight=weight)

    def remove_member(self, name: str) -> bool:
        """Remove a member from the ensemble."""
        for i, member in enumerate(self.members):
            if member.name == name:
                self.members.pop(i)
                self._log.info("Removed ensemble member", name=name)
                return True
        return False

    async def chat(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        strategy: AggregationStrategy | None = None,
    ) -> EnsembleResponse:
        """Query all ensemble members and aggregate responses."""
        strategy = strategy or self.default_strategy
        enabled_members = [m for m in self.members if m.enabled]

        if not enabled_members:
            self._log.error("No enabled ensemble members")
            return EnsembleResponse(
                content="Error: No ensemble members available",
                aggregation_strategy=strategy,
            )

        self._log.debug(
            "Querying ensemble",
            num_members=len(enabled_members),
            strategy=strategy.value,
        )

        # Query all members concurrently
        tasks = [
            self._query_member(member, messages, system, tools, temperature)
            for member in enabled_members
        ]

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter successful responses
        successful: list[tuple[EnsembleMember, LLMResponse]] = []
        for member, response in zip(enabled_members, responses, strict=False):
            if isinstance(response, Exception):
                member.failure_count += 1
                self._log.warning(
                    "Member query failed",
                    member=member.name,
                    error=str(response),
                )
            elif response is not None:
                member.success_count += 1
                successful.append((member, response))

        if len(successful) < self.min_responses:
            self._log.error(
                "Insufficient successful responses",
                successful=len(successful),
                required=self.min_responses,
            )
            return EnsembleResponse(
                content="Error: Insufficient responses from ensemble",
                aggregation_strategy=strategy,
                num_responses=len(successful),
            )

        # Aggregate responses
        return self._aggregate_responses(successful, strategy)

    async def _query_member(
        self,
        member: EnsembleMember,
        messages: list[dict[str, str]],
        system: str | None,
        tools: list[dict[str, Any]] | None,
        temperature: float,
    ) -> LLMResponse | None:
        """Query a single ensemble member with timeout."""
        try:
            start_time = asyncio.get_event_loop().time()

            response = await asyncio.wait_for(
                member.client.chat(
                    messages=messages,
                    system=system,
                    tools=tools,
                    temperature=temperature,
                ),
                timeout=self.timeout_seconds,
            )

            latency = (asyncio.get_event_loop().time() - start_time) * 1000
            member.total_latency_ms += latency

            return response

        except TimeoutError:
            self._log.warning("Member query timed out", member=member.name)
            return None
        except Exception as e:
            self._log.error("Member query error", member=member.name, error=str(e))
            raise

    def _aggregate_responses(
        self,
        responses: list[tuple[EnsembleMember, LLMResponse]],
        strategy: AggregationStrategy,
    ) -> EnsembleResponse:
        """Aggregate responses based on strategy."""
        if strategy == AggregationStrategy.MAJORITY_VOTE:
            return self._majority_vote(responses)
        elif strategy == AggregationStrategy.BEST_CONFIDENCE:
            return self._best_confidence(responses)
        elif strategy == AggregationStrategy.WEIGHTED_AVERAGE:
            return self._weighted_selection(responses)
        elif strategy == AggregationStrategy.RANDOM:
            return self._random_selection(responses)
        elif strategy == AggregationStrategy.FIRST_SUCCESS:
            return self._first_success(responses)
        elif strategy == AggregationStrategy.CONSENSUS:
            return self._consensus(responses)
        else:
            return self._majority_vote(responses)

    def _majority_vote(
        self,
        responses: list[tuple[EnsembleMember, LLMResponse]],
    ) -> EnsembleResponse:
        """Select response by majority vote on content similarity."""
        # Group similar responses
        groups: dict[str, list[tuple[EnsembleMember, LLMResponse]]] = {}

        for member, response in responses:
            # Normalize content for comparison
            key = self._normalize_content(response.content or "")
            if key not in groups:
                groups[key] = []
            groups[key].append((member, response))

        # Find largest group
        largest_group = max(groups.values(), key=len)
        selected_member, selected_response = largest_group[0]

        agreement = len(largest_group) / len(responses)

        return EnsembleResponse(
            content=selected_response.content or "",
            tool_calls=selected_response.tool_calls,
            aggregation_strategy=AggregationStrategy.MAJORITY_VOTE,
            num_responses=len(responses),
            agreement_score=agreement,
            individual_responses=[r for _, r in responses],
            metadata={"selected_member": selected_member.name},
        )

    def _best_confidence(
        self,
        responses: list[tuple[EnsembleMember, LLMResponse]],
    ) -> EnsembleResponse:
        """Select response with highest confidence (based on member success rate)."""
        best_member, best_response = max(
            responses,
            key=lambda x: x[0].success_rate,
        )

        return EnsembleResponse(
            content=best_response.content or "",
            tool_calls=best_response.tool_calls,
            aggregation_strategy=AggregationStrategy.BEST_CONFIDENCE,
            num_responses=len(responses),
            agreement_score=self._calculate_agreement(responses),
            individual_responses=[r for _, r in responses],
            metadata={
                "selected_member": best_member.name,
                "confidence": best_member.success_rate,
            },
        )

    def _weighted_selection(
        self,
        responses: list[tuple[EnsembleMember, LLMResponse]],
    ) -> EnsembleResponse:
        """Select response weighted by member weights and performance."""
        total_weight = sum(m.weight * m.success_rate for m, _ in responses)

        if total_weight == 0:
            return self._random_selection(responses)

        # Weighted random selection
        r = random.uniform(0, total_weight)
        cumulative = 0.0

        for member, response in responses:
            cumulative += member.weight * member.success_rate
            if cumulative >= r:
                return EnsembleResponse(
                    content=response.content or "",
                    tool_calls=response.tool_calls,
                    aggregation_strategy=AggregationStrategy.WEIGHTED_AVERAGE,
                    num_responses=len(responses),
                    agreement_score=self._calculate_agreement(responses),
                    individual_responses=[r for _, r in responses],
                    metadata={"selected_member": member.name},
                )

        # Fallback to last
        member, response = responses[-1]
        return EnsembleResponse(
            content=response.content or "",
            tool_calls=response.tool_calls,
            aggregation_strategy=AggregationStrategy.WEIGHTED_AVERAGE,
            num_responses=len(responses),
            agreement_score=self._calculate_agreement(responses),
            individual_responses=[r for _, r in responses],
            metadata={"selected_member": member.name},
        )

    def _random_selection(
        self,
        responses: list[tuple[EnsembleMember, LLMResponse]],
    ) -> EnsembleResponse:
        """Randomly select a response."""
        member, response = random.choice(responses)

        return EnsembleResponse(
            content=response.content or "",
            tool_calls=response.tool_calls,
            aggregation_strategy=AggregationStrategy.RANDOM,
            num_responses=len(responses),
            agreement_score=self._calculate_agreement(responses),
            individual_responses=[r for _, r in responses],
            metadata={"selected_member": member.name},
        )

    def _first_success(
        self,
        responses: list[tuple[EnsembleMember, LLMResponse]],
    ) -> EnsembleResponse:
        """Return first successful response."""
        member, response = responses[0]

        return EnsembleResponse(
            content=response.content or "",
            tool_calls=response.tool_calls,
            aggregation_strategy=AggregationStrategy.FIRST_SUCCESS,
            num_responses=len(responses),
            agreement_score=self._calculate_agreement(responses),
            individual_responses=[r for _, r in responses],
            metadata={"selected_member": member.name},
        )

    def _consensus(
        self,
        responses: list[tuple[EnsembleMember, LLMResponse]],
    ) -> EnsembleResponse:
        """Only return if all responses agree."""
        contents = [self._normalize_content(r.content or "") for _, r in responses]

        if len(set(contents)) == 1:
            # All agree
            member, response = responses[0]
            return EnsembleResponse(
                content=response.content or "",
                tool_calls=response.tool_calls,
                aggregation_strategy=AggregationStrategy.CONSENSUS,
                num_responses=len(responses),
                agreement_score=1.0,
                individual_responses=[r for _, r in responses],
                metadata={"consensus_reached": True},
            )
        else:
            # No consensus - return error
            return EnsembleResponse(
                content="Error: No consensus reached among ensemble members",
                aggregation_strategy=AggregationStrategy.CONSENSUS,
                num_responses=len(responses),
                agreement_score=self._calculate_agreement(responses),
                individual_responses=[r for _, r in responses],
                metadata={"consensus_reached": False},
            )

    def _normalize_content(self, content: str) -> str:
        """Normalize content for comparison."""
        # Simple normalization - lowercase, strip whitespace
        return content.lower().strip()

    def _calculate_agreement(
        self,
        responses: list[tuple[EnsembleMember, LLMResponse]],
    ) -> float:
        """Calculate agreement score among responses."""
        if len(responses) <= 1:
            return 1.0

        contents = [self._normalize_content(r.content or "") for _, r in responses]
        unique_contents = set(contents)

        # Agreement = 1 - (unique / total)
        return 1.0 - (len(unique_contents) - 1) / len(responses)

    def get_statistics(self) -> dict[str, Any]:
        """Get ensemble statistics."""
        return {
            "num_members": len(self.members),
            "enabled_members": sum(1 for m in self.members if m.enabled),
            "members": [
                {
                    "name": m.name,
                    "enabled": m.enabled,
                    "weight": m.weight,
                    "success_rate": m.success_rate,
                    "avg_latency_ms": m.avg_latency_ms,
                    "total_queries": m.success_count + m.failure_count,
                }
                for m in self.members
            ],
        }

    def reset_statistics(self) -> None:
        """Reset all member statistics."""
        for member in self.members:
            member.success_count = 0
            member.failure_count = 0
            member.total_latency_ms = 0.0
