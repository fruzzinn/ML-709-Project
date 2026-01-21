"""Tests for attack system."""

import pytest

from src.attacks.attack_types import AttackType, AttackConfig, PREDEFINED_SCENARIOS
from src.attacks.scheduler import AttackScheduler, SchedulerStrategy
from src.tools.honest.calculator import CalculatorTool
from src.tools.adversarial.wrappers import WrongOutputWrapper, DelayedResponseWrapper


class TestAttackConfig:
    """Tests for AttackConfig."""

    def test_wrong_output_config(self) -> None:
        """Test wrong output attack configuration."""
        config = AttackConfig.wrong_output(probability=0.5, error_magnitude=0.2)
        assert config.attack_type == AttackType.WRONG_OUTPUT
        assert config.probability == 0.5
        assert config.parameters["error_magnitude"] == 0.2

    def test_delayed_response_config(self) -> None:
        """Test delayed response attack configuration."""
        config = AttackConfig.delayed_response(probability=0.3, max_delay=15.0)
        assert config.attack_type == AttackType.DELAYED_RESPONSE
        assert config.parameters["max_delay"] == 15.0

    def test_target_tools(self) -> None:
        """Test target tool filtering."""
        config = AttackConfig.wrong_output(target_tools=["calculator"])
        assert config.should_target_tool("calculator")
        assert not config.should_target_tool("web_search")

    def test_all_tools_targeted(self) -> None:
        """Test targeting all tools."""
        config = AttackConfig.wrong_output()
        assert config.should_target_tool("calculator")
        assert config.should_target_tool("web_search")
        assert config.should_target_tool("any_tool")


class TestPredefinedScenarios:
    """Tests for predefined attack scenarios."""

    def test_baseline_scenario(self) -> None:
        """Test baseline scenario has no attacks."""
        scenario = PREDEFINED_SCENARIOS["baseline"]
        assert len(scenario.attacks) == 0

    def test_light_attack_scenario(self) -> None:
        """Test light attack scenario."""
        scenario = PREDEFINED_SCENARIOS["light"]
        assert len(scenario.attacks) > 0
        for attack in scenario.attacks:
            assert attack.probability <= 0.2

    def test_heavy_attack_scenario(self) -> None:
        """Test heavy attack scenario."""
        scenario = PREDEFINED_SCENARIOS["heavy"]
        assert len(scenario.attacks) > 0


class TestWrongOutputWrapper:
    """Tests for WrongOutputWrapper."""

    @pytest.fixture
    def wrapper(self) -> WrongOutputWrapper:
        tool = CalculatorTool()
        return WrongOutputWrapper(tool, attack_probability=1.0)  # Always attack for testing

    @pytest.mark.asyncio
    async def test_wrapper_modifies_result(self, wrapper: WrongOutputWrapper) -> None:
        """Test that wrapper can modify results."""
        # Run multiple times - at least one should be different
        original = await wrapper.wrapped_tool.execute({"expression": "10"})
        modified = await wrapper.execute({"expression": "10"})

        # Result should be modified (attack probability is 1.0)
        assert wrapper.attack_count > 0

    @pytest.mark.asyncio
    async def test_attack_recorded(self, wrapper: WrongOutputWrapper) -> None:
        """Test that attacks are recorded."""
        await wrapper.execute({"expression": "5 + 5"})
        assert wrapper.attack_count == 1
        assert len(wrapper.attack_history) == 1

    def test_attack_disabled(self) -> None:
        """Test that attacks can be disabled."""
        tool = CalculatorTool()
        wrapper = WrongOutputWrapper(tool, attack_probability=0.0)

        assert not wrapper.should_attack()


class TestAttackScheduler:
    """Tests for AttackScheduler."""

    def test_scheduler_none_strategy(self) -> None:
        """Test scheduler with no attacks."""
        scheduler = AttackScheduler(strategy=SchedulerStrategy.NONE)
        assert scheduler.strategy == SchedulerStrategy.NONE

    def test_scheduler_random_strategy(self) -> None:
        """Test scheduler with random strategy."""
        scheduler = AttackScheduler(
            strategy=SchedulerStrategy.RANDOM,
            scenario=PREDEFINED_SCENARIOS["light"],
        )
        assert scheduler.strategy == SchedulerStrategy.RANDOM

    def test_scheduler_statistics(self) -> None:
        """Test scheduler statistics."""
        scheduler = AttackScheduler(
            strategy=SchedulerStrategy.RANDOM,
            scenario=PREDEFINED_SCENARIOS["light"],
        )
        stats = scheduler.get_statistics()
        assert "strategy" in stats
        assert "state" in stats
        assert stats["strategy"] == "random"

    def test_scheduler_reset(self) -> None:
        """Test scheduler reset."""
        scheduler = AttackScheduler(strategy=SchedulerStrategy.RANDOM)
        scheduler.state.loop_number = 5
        scheduler.state.total_tool_calls = 10

        scheduler.reset()

        assert scheduler.state.loop_number == 0
        assert scheduler.state.total_tool_calls == 0
