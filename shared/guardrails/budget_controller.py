# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Budget Controller — Cost circuit-breaker for AEL workflows.

Monitors MetricsCollector and halts execution if cost/token budgets
are exceeded. Supports total limits, per-team limits, and per-run limits.

Usage:
    from shared.guardrails import BudgetController, BudgetExceededError
    from shared.observability import MetricsCollector

    collector = MetricsCollector()
    budget = BudgetController(
        max_cost_usd=5.0,
        max_tokens=500_000,
        per_team_limits={"IdeationTeam": 2.0}
    )

    # Check before/after each stage
    budget.check(collector)
    budget.check(collector, current_team="IdeationTeam")
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class BudgetExceededError(Exception):
    """Raised when a budget limit is breached."""

    def __init__(self, message: str, budget_type: str = "total", current_value: float = 0.0, limit: float = 0.0):
        super().__init__(message)
        self.budget_type = budget_type
        self.current_value = current_value
        self.limit = limit


class BudgetStatus(BaseModel):
    """Current budget status snapshot."""
    ok: bool = Field(description="Whether all budgets are within limits")
    total_cost_usd: float = Field(default=0.0, description="Current total cost")
    total_tokens: int = Field(default=0, description="Current total tokens")
    cost_remaining_usd: float = Field(default=0.0, description="Remaining cost budget")
    tokens_remaining: int = Field(default=0, description="Remaining token budget")
    warnings: list = Field(default_factory=list, description="Warning messages (>80% threshold)")


class BudgetController:
    """
    Cost circuit-breaker that halts execution if budget is exceeded.

    Monitors:
    - Total cost in USD (across all teams)
    - Total tokens consumed (input + output)
    - Per-team cost allocation
    - Warning threshold at 80% of limit
    """

    WARNING_THRESHOLD = 0.80  # Warn at 80% of budget

    def __init__(
        self,
        max_cost_usd: float = 5.0,
        max_tokens: int = 500_000,
        per_team_limits: Optional[Dict[str, float]] = None,
        max_retries_per_stage: int = 3,
        max_total_retries: int = 10,
    ):
        """
        Args:
            max_cost_usd: Maximum total cost in USD.
            max_tokens: Maximum total tokens (input + output).
            per_team_limits: Optional per-team cost limits in USD.
            max_retries_per_stage: Maximum retries allowed per stage (loop limit).
            max_total_retries: Maximum total retries across all stages.
        """
        self.max_cost = max_cost_usd
        self.max_tokens = max_tokens
        self.per_team_limits = per_team_limits or {}
        self.max_retries_per_stage = max_retries_per_stage
        self.max_total_retries = max_total_retries
        self._stage_retries: Dict[str, int] = {}
        self._total_retries: int = 0

    def check(
        self,
        collector: Any,
        current_team: Optional[str] = None,
        team_collector: Any = None,
    ) -> BudgetStatus:
        """
        Check if budget is within limits.

        Args:
            collector: MetricsCollector instance with get_summary() (global).
            current_team: Optional team name for per-team limit check.
            team_collector: Optional separate MetricsCollector for the current
                team only. If provided, per-team cost is read from this collector
                instead of attempting to extract team cost from the global one.

        Returns:
            BudgetStatus if within limits.

        Raises:
            BudgetExceededError: If any limit is breached.
        """
        summary = collector.get_summary()
        llm_summary = summary.get("llm", {})
        total_cost = llm_summary.get("total_cost_usd", 0.0)
        total_tokens = llm_summary.get("total_tokens", 0)

        warnings = []

        # Check total cost
        if total_cost > self.max_cost:
            raise BudgetExceededError(
                f"Total cost ${total_cost:.4f} exceeds limit ${self.max_cost:.2f}",
                budget_type="total_cost",
                current_value=total_cost,
                limit=self.max_cost,
            )
        if total_cost > self.max_cost * self.WARNING_THRESHOLD:
            warnings.append(
                f"Cost warning: ${total_cost:.4f} / ${self.max_cost:.2f} "
                f"({total_cost / self.max_cost * 100:.0f}%)"
            )

        # Check total tokens
        if total_tokens > self.max_tokens:
            raise BudgetExceededError(
                f"Total tokens {total_tokens:,} exceeds limit {self.max_tokens:,}",
                budget_type="total_tokens",
                current_value=total_tokens,
                limit=self.max_tokens,
            )
        if total_tokens > self.max_tokens * self.WARNING_THRESHOLD:
            warnings.append(
                f"Token warning: {total_tokens:,} / {self.max_tokens:,} "
                f"({total_tokens / self.max_tokens * 100:.0f}%)"
            )

        # Check per-team cost
        if current_team and current_team in self.per_team_limits:
            team_limit = self.per_team_limits[current_team]
            # Use team-specific collector if provided, else estimate from global
            if team_collector is not None:
                team_summary = team_collector.get_summary()
                team_cost = team_summary.get("llm", {}).get("total_cost_usd", 0.0)
            else:
                team_cost = self._compute_team_cost(summary, current_team)
            if team_cost > team_limit:
                raise BudgetExceededError(
                    f"{current_team} cost ${team_cost:.4f} exceeds "
                    f"team limit ${team_limit:.2f}",
                    budget_type="per_team",
                    current_value=team_cost,
                    limit=team_limit,
                )
            if team_cost > team_limit * self.WARNING_THRESHOLD:
                warnings.append(
                    f"{current_team} cost warning: ${team_cost:.4f} / "
                    f"${team_limit:.2f} ({team_cost / team_limit * 100:.0f}%)"
                )

        return BudgetStatus(
            ok=True,
            total_cost_usd=round(total_cost, 6),
            total_tokens=total_tokens,
            cost_remaining_usd=round(self.max_cost - total_cost, 6),
            tokens_remaining=self.max_tokens - total_tokens,
            warnings=warnings,
        )

    @staticmethod
    def _compute_team_cost(summary: Dict, team: str) -> float:
        """
        Compute total cost for a team from the global summary.

        Uses by_stage grouping to sum costs across all stages. Since
        MetricsCollector groups by stage name (e.g., "Sourcing",
        "Refinement"), this provides the best available per-team estimate
        from a shared collector.

        For accurate per-team tracking in the cross-team pipeline,
        prefer passing a team-specific collector via team_collector.
        """
        by_stage = summary.get("by_stage", {})
        if by_stage:
            return sum(
                stage_data.get("llm", {}).get("total_cost_usd", 0.0)
                for stage_data in by_stage.values()
            )
        # Fallback: return global total (best effort)
        return summary.get("llm", {}).get("total_cost_usd", 0.0)

    def record_retry(self, stage_name: str) -> None:
        """
        Record a retry attempt for a stage.

        Args:
            stage_name: Name of the stage being retried.

        Raises:
            BudgetExceededError: If per-stage or total retry limit is exceeded.
        """
        self._stage_retries[stage_name] = self._stage_retries.get(stage_name, 0) + 1
        self._total_retries += 1

        if self._stage_retries[stage_name] > self.max_retries_per_stage:
            raise BudgetExceededError(
                f"Stage '{stage_name}' exceeded max retries "
                f"({self._stage_retries[stage_name]}/{self.max_retries_per_stage})",
                budget_type="retry_per_stage",
                current_value=self._stage_retries[stage_name],
                limit=self.max_retries_per_stage,
            )

        if self._total_retries > self.max_total_retries:
            raise BudgetExceededError(
                f"Total retries ({self._total_retries}) exceeded limit ({self.max_total_retries})",
                budget_type="retry_total",
                current_value=self._total_retries,
                limit=self.max_total_retries,
            )

    def get_retry_counts(self) -> Dict[str, Any]:
        """Return current retry counts."""
        return {
            "stage_retries": dict(self._stage_retries),
            "total_retries": self._total_retries,
            "max_retries_per_stage": self.max_retries_per_stage,
            "max_total_retries": self.max_total_retries,
        }

    def get_limits(self) -> Dict[str, Any]:
        """Return current budget configuration."""
        return {
            "max_cost_usd": self.max_cost,
            "max_tokens": self.max_tokens,
            "per_team_limits": self.per_team_limits,
            "max_retries_per_stage": self.max_retries_per_stage,
            "max_total_retries": self.max_total_retries,
        }
