# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Tests for shared.guardrails.budget_controller.BudgetController.

Validates that:
- BudgetStatus is returned when within limits
- BudgetExceededError is raised on cost overrun
- BudgetExceededError is raised on token overrun
- Warnings are generated at 80% threshold
- Per-team limits work correctly
"""

import pytest

from shared.guardrails.budget_controller import (
    BudgetController,
    BudgetExceededError,
    BudgetStatus,
)


# ============================================================================
# Mock MetricsCollector
# ============================================================================

class MockCollector:
    """Minimal MetricsCollector mock for budget testing."""

    def __init__(self, total_cost: float = 0.0, total_tokens: int = 0):
        self._cost = total_cost
        self._tokens = total_tokens

    def get_summary(self):
        return {
            "schema_version": "1.0.0",
            "llm": {
                "total_calls": 10,
                "total_prompt_tokens": self._tokens // 2,
                "total_completion_tokens": self._tokens // 2,
                "total_tokens": self._tokens,
                "total_cost_usd": self._cost,
                "avg_latency_seconds": 1.0,
                "error_count": 0,
            },
            "tools": {
                "total_calls": 5,
                "success_count": 5,
                "error_count": 0,
                "success_rate": 1.0,
                "avg_latency_seconds": 0.5,
            },
            "by_agent": {},
            "by_stage": {},
        }


# ============================================================================
# Tests
# ============================================================================

class TestBudgetWithinLimits:
    """Test normal operation within budget."""

    def test_low_cost_passes(self):
        budget = BudgetController(max_cost_usd=5.0, max_tokens=500_000)
        collector = MockCollector(total_cost=0.05, total_tokens=1000)
        status = budget.check(collector)
        assert status.ok
        assert status.total_cost_usd == 0.05
        assert status.cost_remaining_usd > 4.0
        assert len(status.warnings) == 0

    def test_zero_cost_passes(self):
        budget = BudgetController()
        collector = MockCollector(total_cost=0.0, total_tokens=0)
        status = budget.check(collector)
        assert status.ok


class TestCostOverrun:
    """Test cost limit enforcement."""

    def test_cost_exceeded_raises(self):
        budget = BudgetController(max_cost_usd=1.0)
        collector = MockCollector(total_cost=1.50)
        with pytest.raises(BudgetExceededError) as exc_info:
            budget.check(collector)
        assert exc_info.value.budget_type == "total_cost"
        assert exc_info.value.current_value == 1.50
        assert exc_info.value.limit == 1.0

    def test_cost_exactly_at_limit_raises(self):
        budget = BudgetController(max_cost_usd=1.0)
        collector = MockCollector(total_cost=1.001)
        with pytest.raises(BudgetExceededError):
            budget.check(collector)


class TestTokenOverrun:
    """Test token limit enforcement."""

    def test_token_exceeded_raises(self):
        budget = BudgetController(max_tokens=100_000)
        collector = MockCollector(total_tokens=150_000)
        with pytest.raises(BudgetExceededError) as exc_info:
            budget.check(collector)
        assert exc_info.value.budget_type == "total_tokens"


class TestWarnings:
    """Test warning generation at 80% threshold."""

    def test_cost_warning_at_80_percent(self):
        budget = BudgetController(max_cost_usd=1.0)
        collector = MockCollector(total_cost=0.85)
        status = budget.check(collector)
        assert status.ok
        assert len(status.warnings) == 1
        assert "Cost warning" in status.warnings[0]

    def test_token_warning_at_80_percent(self):
        budget = BudgetController(max_tokens=100_000)
        collector = MockCollector(total_tokens=85_000)
        status = budget.check(collector)
        assert status.ok
        assert any("Token warning" in w for w in status.warnings)

    def test_no_warning_below_threshold(self):
        budget = BudgetController(max_cost_usd=10.0, max_tokens=500_000)
        collector = MockCollector(total_cost=0.05, total_tokens=1000)
        status = budget.check(collector)
        assert len(status.warnings) == 0


class TestPerTeamLimits:
    """Test per-team cost limits."""

    def test_per_team_exceeded_raises(self):
        budget = BudgetController(
            max_cost_usd=10.0,
            per_team_limits={"IdeationTeam": 0.50},
        )
        collector = MockCollector(total_cost=0.75)
        with pytest.raises(BudgetExceededError) as exc_info:
            budget.check(collector, current_team="IdeationTeam")
        assert exc_info.value.budget_type == "per_team"

    def test_per_team_within_limit(self):
        budget = BudgetController(
            max_cost_usd=10.0,
            per_team_limits={"IdeationTeam": 2.0},
        )
        collector = MockCollector(total_cost=0.50)
        status = budget.check(collector, current_team="IdeationTeam")
        assert status.ok

    def test_per_team_no_limit_for_team(self):
        """Team without a per-team limit should not be checked."""
        budget = BudgetController(
            max_cost_usd=10.0,
            per_team_limits={"IdeationTeam": 0.01},
        )
        collector = MockCollector(total_cost=5.0)
        # LiteratureTeam has no per-team limit, should pass
        status = budget.check(collector, current_team="LiteratureTeam")
        assert status.ok


class TestGetLimits:
    """Test configuration retrieval."""

    def test_get_limits(self):
        budget = BudgetController(
            max_cost_usd=3.0,
            max_tokens=200_000,
            per_team_limits={"A": 1.0},
        )
        limits = budget.get_limits()
        assert limits["max_cost_usd"] == 3.0
        assert limits["max_tokens"] == 200_000
        assert limits["per_team_limits"]["A"] == 1.0
