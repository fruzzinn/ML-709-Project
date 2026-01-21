"""Agent state management with checkpointing support."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AgentStatus(str, Enum):
    """Agent execution status."""

    IDLE = "idle"
    REASONING = "reasoning"
    EXECUTING = "executing"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class ToolCallResult(BaseModel):
    """Result from a tool execution."""

    tool_name: str
    tool_id: str
    arguments: dict[str, Any]
    result: Any | None = None
    error: str | None = None
    execution_time_ms: float = 0.0
    cached: bool = False
    anomaly_detected: bool = False
    anomaly_details: str | None = None


class ReasoningStep(BaseModel):
    """A single reasoning step in the agent loop."""

    loop_number: int
    thought: str
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    tool_results: list[ToolCallResult] = Field(default_factory=list)
    observation: str | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AgentState(BaseModel):
    """Complete agent state at a point in time."""

    # Execution tracking
    run_id: str
    task: str
    current_loop: int = 0
    max_loops: int = 10
    status: AgentStatus = AgentStatus.IDLE

    # Reasoning history
    reasoning_steps: list[ReasoningStep] = Field(default_factory=list)

    # Results
    final_answer: str | None = None
    intermediate_results: list[Any] = Field(default_factory=list)

    # Metrics
    total_tool_calls: int = 0
    successful_tool_calls: int = 0
    failed_tool_calls: int = 0
    anomalies_detected: int = 0
    rollbacks_performed: int = 0

    # Timestamps
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None

    # Checkpointing
    checkpoint_id: str | None = None

    def get_conversation_history(self) -> list[dict[str, str]]:
        """Get conversation history for LLM context."""
        history: list[dict[str, str]] = []

        for step in self.reasoning_steps:
            # Add assistant thought
            history.append({"role": "assistant", "content": step.thought})

            # Add tool results as user messages
            if step.tool_results:
                results_text = "\n".join(
                    f"Tool {r.tool_name}: {r.result if r.result else r.error}"
                    for r in step.tool_results
                )
                history.append({"role": "user", "content": f"Tool Results:\n{results_text}"})

        return history

    def compute_hash(self) -> str:
        """Compute hash of current state for change detection."""
        state_dict = self.model_dump(exclude={"checkpoint_id", "started_at", "completed_at"})
        state_json = json.dumps(state_dict, sort_keys=True, default=str)
        return hashlib.sha256(state_json.encode()).hexdigest()[:16]


@dataclass
class Checkpoint:
    """A state checkpoint for rollback support."""

    checkpoint_id: str
    state: AgentState
    memory_snapshot: dict[str, Any]
    created_at: datetime = field(default_factory=datetime.utcnow)
    trigger: str = "manual"  # manual, periodic, pre_tool, post_step

    def to_dict(self) -> dict[str, Any]:
        """Serialize checkpoint to dictionary."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "state": self.state.model_dump(),
            "memory_snapshot": self.memory_snapshot,
            "created_at": self.created_at.isoformat(),
            "trigger": self.trigger,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Checkpoint:
        """Deserialize checkpoint from dictionary."""
        return cls(
            checkpoint_id=data["checkpoint_id"],
            state=AgentState.model_validate(data["state"]),
            memory_snapshot=data["memory_snapshot"],
            created_at=datetime.fromisoformat(data["created_at"]),
            trigger=data["trigger"],
        )


class StateManager:
    """Manages agent state with checkpointing and rollback support."""

    def __init__(self, max_checkpoints: int = 10) -> None:
        self.max_checkpoints = max_checkpoints
        self._checkpoints: list[Checkpoint] = []
        self._checkpoint_counter = 0

    def create_checkpoint(
        self,
        state: AgentState,
        memory: dict[str, Any],
        trigger: str = "manual",
    ) -> Checkpoint:
        """Create a new checkpoint."""
        self._checkpoint_counter += 1
        checkpoint_id = f"cp_{state.run_id}_{self._checkpoint_counter}"

        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            state=state.model_copy(deep=True),
            memory_snapshot=memory.copy(),
            trigger=trigger,
        )

        self._checkpoints.append(checkpoint)

        # Prune old checkpoints if exceeding limit
        if len(self._checkpoints) > self.max_checkpoints:
            self._checkpoints = self._checkpoints[-self.max_checkpoints :]

        return checkpoint

    def get_latest_checkpoint(self) -> Checkpoint | None:
        """Get the most recent checkpoint."""
        return self._checkpoints[-1] if self._checkpoints else None

    def get_checkpoint(self, checkpoint_id: str) -> Checkpoint | None:
        """Get a specific checkpoint by ID."""
        for cp in self._checkpoints:
            if cp.checkpoint_id == checkpoint_id:
                return cp
        return None

    def rollback_to_checkpoint(self, checkpoint: Checkpoint) -> tuple[AgentState, dict[str, Any]]:
        """Rollback to a specific checkpoint."""
        # Remove checkpoints after the rollback point
        idx = self._checkpoints.index(checkpoint)
        self._checkpoints = self._checkpoints[: idx + 1]

        # Return copies to avoid mutation issues
        return (
            checkpoint.state.model_copy(deep=True),
            checkpoint.memory_snapshot.copy(),
        )

    def rollback_to_latest(self) -> tuple[AgentState, dict[str, Any]] | None:
        """Rollback to the most recent checkpoint."""
        checkpoint = self.get_latest_checkpoint()
        if checkpoint:
            return self.rollback_to_checkpoint(checkpoint)
        return None

    def clear_checkpoints(self) -> None:
        """Clear all checkpoints."""
        self._checkpoints.clear()
        self._checkpoint_counter = 0

    @property
    def checkpoint_count(self) -> int:
        """Get number of stored checkpoints."""
        return len(self._checkpoints)

    @property
    def checkpoints(self) -> list[Checkpoint]:
        """Get all checkpoints (read-only copy)."""
        return self._checkpoints.copy()
