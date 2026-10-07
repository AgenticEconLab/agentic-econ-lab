# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Adaptive Orchestrator — Dynamic topology selection (V0.6 Phase 5).

Selects optimal execution topology (sequential, parallel, hierarchical, hybrid)
based on team dependencies and world model state. Inspired by AdaptOrch
(arXiv 2602.16873).

Usage:
    from shared.orchestration.adaptive import AdaptiveOrchestrator

    orch = AdaptiveOrchestrator()
    topology = orch.select_topology(["IdeationTeam", "LiteratureTeam", "DataTeam"])
    plan = orch.create_execution_plan(["IdeationTeam", "LiteratureTeam", "DataTeam"])
"""

from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field


# Team dependency graph — which teams depend on which
TEAM_DEPENDENCIES: Dict[str, List[str]] = {
    "IdeationTeam": [],
    "LiteratureTeam": ["IdeationTeam"],
    "DataTeam": [],
    "ModelTeam": ["IdeationTeam", "LiteratureTeam", "DataTeam"],
}


class ExecutionStep(BaseModel):
    """A single step in an execution plan."""

    step_number: int
    teams: List[str]
    mode: str = "sequential"  # "sequential" or "parallel"
    depends_on: List[int] = Field(default_factory=list)


class ExecutionPlan(BaseModel):
    """A complete execution plan for a set of teams."""

    topology: str
    steps: List[ExecutionStep]
    total_steps: int = 0
    parallel_groups: int = 0


class AdaptiveOrchestrator:
    """Dynamic topology selection for multi-team pipelines.

    Analyzes team dependencies to determine optimal execution order
    and identifies opportunities for parallel execution.

    Args:
        dependencies: Custom dependency graph. Defaults to TEAM_DEPENDENCIES.
    """

    TOPOLOGIES = ["sequential", "parallel", "hierarchical", "hybrid"]

    def __init__(
        self,
        dependencies: Optional[Dict[str, List[str]]] = None,
    ):
        self.dependencies = dependencies or dict(TEAM_DEPENDENCIES)

    def select_topology(self, teams: List[str]) -> str:
        """Select optimal execution topology based on team dependencies.

        Args:
            teams: List of teams to execute.

        Returns:
            Topology name: "sequential", "parallel", "hierarchical", or "hybrid".
        """
        if len(teams) <= 1:
            return "sequential"

        # Check if any teams have dependencies on each other
        has_deps = False
        independent_count = 0
        for team in teams:
            deps = self.dependencies.get(team, [])
            relevant_deps = [d for d in deps if d in teams]
            if relevant_deps:
                has_deps = True
            else:
                independent_count += 1

        # All independent → parallel
        if not has_deps:
            return "parallel"

        # All have dependencies → sequential or hierarchical
        if independent_count == 0:
            return "sequential"

        # Some independent, some dependent → check structure
        if independent_count >= 2 and has_deps:
            # Mixed: some can run in parallel, others need to wait
            return "hybrid"

        # Single root + dependents → hierarchical
        if independent_count == 1 and has_deps:
            return "hierarchical"

        return "sequential"

    def create_execution_plan(self, teams: List[str]) -> ExecutionPlan:
        """Create a detailed execution plan with step ordering.

        Args:
            teams: List of teams to execute.

        Returns:
            ExecutionPlan with ordered steps.
        """
        topology = self.select_topology(teams)
        steps = self._build_steps(teams)

        parallel_groups = sum(1 for s in steps if len(s.teams) > 1)

        return ExecutionPlan(
            topology=topology,
            steps=steps,
            total_steps=len(steps),
            parallel_groups=parallel_groups,
        )

    def get_independent_teams(self, teams: List[str]) -> List[str]:
        """Identify teams that can run without waiting for others.

        Args:
            teams: Teams to check.

        Returns:
            List of teams with no dependencies within the given set.
        """
        independent = []
        for team in teams:
            deps = self.dependencies.get(team, [])
            relevant_deps = [d for d in deps if d in teams]
            if not relevant_deps:
                independent.append(team)
        return independent

    def get_execution_order(self, teams: List[str]) -> List[List[str]]:
        """Get topological ordering of teams as parallelizable groups.

        Args:
            teams: Teams to order.

        Returns:
            List of groups, where teams within a group can run in parallel.
        """
        remaining = set(teams)
        completed: Set[str] = set()
        order: List[List[str]] = []

        while remaining:
            # Find teams whose deps are all completed
            ready = []
            for team in remaining:
                deps = self.dependencies.get(team, [])
                relevant_deps = [d for d in deps if d in set(teams)]
                if all(d in completed for d in relevant_deps):
                    ready.append(team)

            if not ready:
                # Circular dependency or missing dep — add remaining sequentially
                order.append(sorted(remaining))
                break

            order.append(sorted(ready))
            completed.update(ready)
            remaining -= set(ready)

        return order

    def _build_steps(self, teams: List[str]) -> List[ExecutionStep]:
        """Build execution steps from topological ordering."""
        groups = self.get_execution_order(teams)
        steps = []
        for i, group in enumerate(groups):
            mode = "parallel" if len(group) > 1 else "sequential"
            depends_on = [i - 1] if i > 0 else []
            steps.append(ExecutionStep(
                step_number=i + 1,
                teams=group,
                mode=mode,
                depends_on=depends_on,
            ))
        return steps
