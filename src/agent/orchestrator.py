"""ReAct Agent Orchestrator - Main agent execution loop."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

import structlog

from src.agent.memory import WorkingMemory
from src.agent.state import AgentState, AgentStatus, ReasoningStep, StateManager, ToolCallResult

if TYPE_CHECKING:
    from src.defenses.base import DefenseManager
    from src.llm.client import LLMClient
    from src.tools.registry import ToolRegistry

logger = structlog.get_logger()


@dataclass
class AgentConfig:
    """Configuration for agent execution."""

    max_loops: int = 10
    temperature: float = 0.7
    timeout_seconds: float = 30.0
    enable_checkpointing: bool = True
    checkpoint_interval: int = 1
    enable_self_consistency: bool = True
    consistency_threshold: float = 0.7
    consistency_hard_minimum: float = 0.5
    num_consistency_checks: int = 3


@dataclass
class AgentRunInput:
    """Input for an agent run."""

    task: str
    context: dict[str, Any] | None = None
    initial_memory: dict[str, Any] | None = None


@dataclass
class AgentRunResult:
    """Result of an agent run."""

    success: bool
    final_answer: str | None
    state: AgentState
    error: str | None = None
    metrics: dict[str, Any] | None = None


class AgentOrchestrator:
    """ReAct pattern agent orchestrator.

    Implements the core agent loop:
    1. REASON: Get LLM thought + action
    2. VERIFY: Pre-execution defense checks
    3. ACT: Execute tools with monitoring
    4. DETECT: Post-execution anomaly detection
    5. OBSERVE: Update state
    6. CHECK: Self-consistency verification
    """

    def __init__(
        self,
        llm_client: LLMClient,
        tool_registry: ToolRegistry,
        defense_manager: DefenseManager | None = None,
        config: AgentConfig | None = None,
    ) -> None:
        self.llm = llm_client
        self.tools = tool_registry
        self.defenses = defense_manager
        self.config = config or AgentConfig()

        self.state_manager = StateManager(max_checkpoints=10)
        self.memory = WorkingMemory(max_entries=100)

        self._log = logger.bind(component="orchestrator")

    async def run(self, input: AgentRunInput) -> AgentRunResult:
        """Execute the agent loop for a given task."""
        run_id = str(uuid.uuid4())[:8]
        self._log = self._log.bind(run_id=run_id)
        self._log.info("Starting agent run", task=input.task[:100])

        # Initialize state
        state = AgentState(
            run_id=run_id,
            task=input.task,
            max_loops=self.config.max_loops,
            status=AgentStatus.IDLE,
        )

        # Initialize memory
        self.memory.clear()
        if input.initial_memory:
            for key, value in input.initial_memory.items():
                self.memory.store(key, value, entry_type="initial", source="user_input")

        if input.context:
            self.memory.store("context", input.context, entry_type="context", source="user_input")

        try:
            result = await self._execute_loop(state)
            return result
        except Exception as e:
            self._log.error("Agent run failed", error=str(e))
            state.status = AgentStatus.FAILED
            state.completed_at = datetime.utcnow()
            return AgentRunResult(
                success=False,
                final_answer=None,
                state=state,
                error=str(e),
            )

    async def _execute_loop(self, state: AgentState) -> AgentRunResult:
        """Main agent execution loop."""
        state.status = AgentStatus.REASONING

        while state.current_loop < state.max_loops:
            self._log.info("Starting loop iteration", loop=state.current_loop)

            # Create checkpoint before processing
            if self.config.enable_checkpointing and state.current_loop % self.config.checkpoint_interval == 0:
                self.state_manager.create_checkpoint(
                    state,
                    self.memory.snapshot(),
                    trigger="periodic",
                )

            # 1. REASON: Get LLM thought and action
            thought, tool_calls = await self._reason(state)

            if not tool_calls:
                # No tool calls means we have a final answer
                state.final_answer = thought
                state.status = AgentStatus.COMPLETED
                state.completed_at = datetime.utcnow()
                self._log.info("Agent completed with final answer")
                break

            # 2. VERIFY: Pre-execution defense checks
            if self.defenses:
                verified, rejection_reason = await self.defenses.pre_execute_check(
                    tool_calls, state
                )
                if not verified:
                    self._log.warning("Pre-execution check failed", reason=rejection_reason)
                    # Record and continue with caution or abort
                    state.anomalies_detected += 1
                    # For now, skip these tool calls
                    tool_calls = []

            # 3. ACT: Execute tools with monitoring
            checkpoint = None
            if self.config.enable_checkpointing:
                checkpoint = self.state_manager.create_checkpoint(
                    state,
                    self.memory.snapshot(),
                    trigger="pre_tool",
                )

            state.status = AgentStatus.EXECUTING
            results = await self._execute_tools(tool_calls, state)

            # 4. DETECT: Post-execution anomaly detection
            if self.defenses:
                anomaly_detected, anomaly_details = await self.defenses.detect_anomaly(
                    results, state
                )
                if anomaly_detected:
                    self._log.warning("Anomaly detected", details=anomaly_details)
                    state.anomalies_detected += 1

                    # Handle anomaly - attempt rollback
                    if checkpoint:
                        await self._handle_anomaly(results, checkpoint, state)
                        continue

            # 5. OBSERVE: Update state with results
            state = self._update_state(state, thought, tool_calls, results)

            # 6. CHECK: Self-consistency verification
            if self.config.enable_self_consistency:
                consistency_score = await self._check_consistency(state)
                if consistency_score >= self.config.consistency_threshold:
                    self._log.info("Consistency threshold met", score=consistency_score)
                    # Could potentially finish early if highly consistent
                elif consistency_score < self.config.consistency_hard_minimum:
                    self._log.warning(
                        "Consistency below hard minimum",
                        score=consistency_score,
                        minimum=self.config.consistency_hard_minimum,
                    )
                    # May need intervention

            state.current_loop += 1
            state.status = AgentStatus.REASONING

        # Check if we hit max loops without completing
        if state.status != AgentStatus.COMPLETED:
            state.status = AgentStatus.FAILED
            state.completed_at = datetime.utcnow()
            self._log.warning("Max loops reached without completion")

        return AgentRunResult(
            success=state.status == AgentStatus.COMPLETED,
            final_answer=state.final_answer,
            state=state,
            metrics=self._compute_metrics(state),
        )

    async def _reason(
        self, state: AgentState
    ) -> tuple[str, list[dict[str, Any]]]:
        """Generate thought and determine next actions using LLM."""
        # Build prompt with current context
        system_prompt = self._build_system_prompt()
        messages = self._build_messages(state)

        # Get LLM response with tool definitions
        response = await self.llm.chat(
            messages=messages,
            system=system_prompt,
            tools=self.tools.get_tool_definitions(),
            temperature=self.config.temperature,
        )

        thought = response.content or ""
        tool_calls = response.tool_calls or []

        self._log.debug(
            "Reasoning complete",
            thought_length=len(thought),
            num_tool_calls=len(tool_calls),
        )

        return thought, tool_calls

    async def _execute_tools(
        self,
        tool_calls: list[dict[str, Any]],
        state: AgentState,
    ) -> list[ToolCallResult]:
        """Execute tool calls with monitoring."""
        results: list[ToolCallResult] = []

        for call in tool_calls:
            tool_name = call.get("name", "")
            tool_id = call.get("id", str(uuid.uuid4())[:8])
            arguments = call.get("arguments", {})

            self._log.debug("Executing tool", tool=tool_name, tool_id=tool_id)

            start_time = asyncio.get_event_loop().time()

            try:
                # Execute through registry (handles adversarial wrappers)
                result = await self.tools.execute(tool_name, arguments)

                execution_time = (asyncio.get_event_loop().time() - start_time) * 1000

                tool_result = ToolCallResult(
                    tool_name=tool_name,
                    tool_id=tool_id,
                    arguments=arguments,
                    result=result,
                    execution_time_ms=execution_time,
                )

                state.total_tool_calls += 1
                state.successful_tool_calls += 1

                # Store result in memory
                self.memory.store(
                    key=f"tool_result_{tool_id}",
                    value=result,
                    entry_type="tool_output",
                    source=tool_name,
                    loop_number=state.current_loop,
                )

            except TimeoutError:
                tool_result = ToolCallResult(
                    tool_name=tool_name,
                    tool_id=tool_id,
                    arguments=arguments,
                    error="Tool execution timed out",
                    execution_time_ms=self.config.timeout_seconds * 1000,
                )
                state.total_tool_calls += 1
                state.failed_tool_calls += 1

            except Exception as e:
                execution_time = (asyncio.get_event_loop().time() - start_time) * 1000
                tool_result = ToolCallResult(
                    tool_name=tool_name,
                    tool_id=tool_id,
                    arguments=arguments,
                    error=str(e),
                    execution_time_ms=execution_time,
                )
                state.total_tool_calls += 1
                state.failed_tool_calls += 1

            results.append(tool_result)

        return results

    async def _handle_anomaly(
        self,
        _results: list[ToolCallResult],
        checkpoint: Any,
        state: AgentState,
    ) -> None:
        """Handle detected anomaly with rollback."""
        self._log.info("Handling anomaly, attempting rollback")

        # Attempt rollback
        rollback_result = self.state_manager.rollback_to_checkpoint(checkpoint)
        if rollback_result:
            restored_state, restored_memory = rollback_result
            # Update current state from restored
            state.reasoning_steps = restored_state.reasoning_steps
            state.intermediate_results = restored_state.intermediate_results
            self.memory.restore(restored_memory)
            state.rollbacks_performed += 1
            self._log.info("Rollback successful")
        else:
            self._log.error("Rollback failed - no valid checkpoint")

    def _update_state(
        self,
        state: AgentState,
        thought: str,
        tool_calls: list[dict[str, Any]],
        results: list[ToolCallResult],
    ) -> AgentState:
        """Update state with new reasoning step."""
        step = ReasoningStep(
            loop_number=state.current_loop,
            thought=thought,
            tool_calls=tool_calls,
            tool_results=results,
        )
        state.reasoning_steps.append(step)

        # Store in memory
        self.memory.store(
            key=f"step_{state.current_loop}_thought",
            value=thought,
            entry_type="observation",
            source="reasoning",
            loop_number=state.current_loop,
        )

        return state

    async def _check_consistency(self, _state: AgentState) -> float:
        """Check self-consistency of agent reasoning."""
        if not self.config.enable_self_consistency:
            return 1.0

        # Simple implementation - check if recent tool results are consistent
        recent_results = self.memory.get_by_type("tool_output")[-5:]

        if len(recent_results) < 2:
            return 1.0

        # Basic consistency: no errors in recent results
        error_count = sum(
            1 for r in recent_results
            if isinstance(r.value, dict) and r.value.get("error")
        )

        consistency = 1.0 - (error_count / len(recent_results))
        return consistency

    def _build_system_prompt(self) -> str:
        """Build system prompt for the agent."""
        tools_description = self.tools.get_tools_description()

        return f"""You are a helpful AI assistant that can use tools to accomplish tasks.

Available tools:
{tools_description}

Instructions:
1. Think step by step about how to accomplish the task
2. Use tools when needed to gather information or perform actions
3. Provide a final answer when you have completed the task
4. If you encounter errors, try alternative approaches

When you have enough information to answer, provide your final response without calling any tools.
"""

    def _build_messages(self, state: AgentState) -> list[dict[str, str]]:
        """Build message history for LLM."""
        messages = [{"role": "user", "content": f"Task: {state.task}"}]

        # Add conversation history
        history = state.get_conversation_history()
        messages.extend(history)

        # Add memory context if relevant
        if self.memory.size > 0:
            memory_context = self.memory.to_context_string(max_entries=10)
            messages.append({
                "role": "system",
                "content": f"Current context:\n{memory_context}",
            })

        return messages

    def _compute_metrics(self, state: AgentState) -> dict[str, Any]:
        """Compute execution metrics."""
        total_execution_time = sum(
            r.execution_time_ms
            for step in state.reasoning_steps
            for r in step.tool_results
        )

        return {
            "total_loops": state.current_loop,
            "total_tool_calls": state.total_tool_calls,
            "successful_tool_calls": state.successful_tool_calls,
            "failed_tool_calls": state.failed_tool_calls,
            "anomalies_detected": state.anomalies_detected,
            "rollbacks_performed": state.rollbacks_performed,
            "total_execution_time_ms": total_execution_time,
            "checkpoints_created": self.state_manager.checkpoint_count,
        }
