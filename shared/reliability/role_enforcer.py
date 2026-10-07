# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Role Enforcer — Role boundary validation for agent outputs.

Prevents LLM agents from drifting outside their designated roles
(e.g., a data-fetching agent making analytical claims). Checks
agent outputs for role boundary violations.

Usage:
    from shared.reliability.role_enforcer import RoleEnforcer

    enforcer = RoleEnforcer({
        "CorpusScout": {
            "allowed_output_types": ["literature_items", "search_results"],
            "forbidden_patterns": ["I recommend", "In my opinion"],
        },
    })

    violations = enforcer.check_output("CorpusScout", output_text)
    if violations:
        logger.warning(f"Role violations: {violations}")
"""

import re
from typing import Any, Dict, List, Optional


class RoleViolation:
    """A single role boundary violation."""

    __slots__ = ("agent", "violation_type", "message", "severity")

    def __init__(
        self,
        agent: str,
        violation_type: str,
        message: str,
        severity: str = "warning",
    ):
        self.agent = agent
        self.violation_type = violation_type
        self.message = message
        self.severity = severity  # "warning" or "error"

    def __repr__(self):
        return f"RoleViolation({self.agent}, {self.violation_type}: {self.message})"


# Default role constraints for AEL agents
DEFAULT_ROLE_CONSTRAINTS: Dict[str, Dict[str, Any]] = {
    # IdeationTeam agents
    "TrendSurfer": {
        "allowed_output_types": ["trends", "topics"],
        "forbidden_patterns": [r"(?i)\bI recommend\b", r"(?i)\bIn my opinion\b"],
        "max_output_length": 50000,
    },
    "CorpusScout": {
        "allowed_output_types": ["literature_items", "search_results", "papers"],
        "forbidden_patterns": [r"(?i)\bI believe\b"],
        "max_output_length": 100000,
    },
    # LiteratureTeam agents
    "InsightSummarizer": {
        "allowed_output_types": ["summaries", "insights"],
        "forbidden_patterns": [],
        "max_output_length": 100000,
    },
    # DataTeam agents
    "DataCollector": {
        "allowed_output_types": ["data", "datasets", "series"],
        "forbidden_patterns": [
            r"(?i)\bthe data suggests\b",
            r"(?i)\bwe can conclude\b",
        ],
        "max_output_length": 500000,
    },
}


class RoleEnforcer:
    """
    Role boundary validation for agent outputs.

    Checks agent outputs against configurable constraints:
    - Forbidden text patterns (regex)
    - Output length limits
    - (Future: output type validation)
    """

    def __init__(self, constraints: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        Args:
            constraints: Agent name → constraint dict mapping.
                Each constraint dict may include:
                - forbidden_patterns: List of regex patterns
                - max_output_length: Maximum output string length
                - allowed_output_types: List of allowed output types
        """
        self.constraints = constraints or DEFAULT_ROLE_CONSTRAINTS

    def check_output(
        self,
        agent_name: str,
        output: Any,
    ) -> List[RoleViolation]:
        """
        Check an agent's output for role boundary violations.

        Args:
            agent_name: Name of the agent.
            output: Agent output (str, dict, or any serializable).

        Returns:
            List of RoleViolation objects (empty = no violations).
        """
        violations: List[RoleViolation] = []
        constraint = self.constraints.get(agent_name)
        if constraint is None:
            return violations  # No constraints defined → no violations

        # Convert output to string for pattern matching
        if isinstance(output, dict):
            output_str = str(output)
        elif isinstance(output, str):
            output_str = output
        else:
            output_str = str(output)

        # Check forbidden patterns
        for pattern in constraint.get("forbidden_patterns", []):
            if re.search(pattern, output_str):
                violations.append(RoleViolation(
                    agent=agent_name,
                    violation_type="forbidden_pattern",
                    message=f"Output contains forbidden pattern: {pattern}",
                    severity="warning",
                ))

        # Check output length
        max_len = constraint.get("max_output_length")
        if max_len and len(output_str) > max_len:
            violations.append(RoleViolation(
                agent=agent_name,
                violation_type="output_too_long",
                message=f"Output length {len(output_str)} exceeds limit {max_len}",
                severity="warning",
            ))

        return violations

    def add_constraint(self, agent_name: str, constraint: Dict[str, Any]):
        """Add or update constraints for an agent."""
        self.constraints[agent_name] = constraint

    def get_constraint(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Get constraints for an agent."""
        return self.constraints.get(agent_name)
