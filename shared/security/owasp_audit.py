# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
OWASP Top 10 for Agentic Applications (2026) — final ASI01–ASI10 taxonomy.

Reference: https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/
Published: 2025-12-10.

The V0.6 audit referenced an earlier preview taxonomy. V0.7 Pre-Phase remaps to
the final labels and adds the *Least Agency* design principle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional


class ASICategory(str, Enum):
    """OWASP Agentic Security Index (final 2026 taxonomy)."""
    AGENT_GOAL_HIJACK                  = "ASI01"
    TOOL_MISUSE                        = "ASI02"
    AGENT_IDENTITY_PRIVILEGE_ABUSE     = "ASI03"
    SUPPLY_CHAIN                       = "ASI04"
    UNEXPECTED_CODE_EXECUTION          = "ASI05"
    MEMORY_CONTEXT_POISONING           = "ASI06"
    INSECURE_INTER_AGENT_COMMUNICATION = "ASI07"
    CASCADING_FAILURES                 = "ASI08"
    HUMAN_AGENT_TRUST                  = "ASI09"
    ROGUE_AGENTS                       = "ASI10"


# Deprecated aliases — V0.6 labels kept for one-release back-compat.
DEPRECATED_ASI_ALIASES: Dict[str, ASICategory] = {
    "goal_hijacking":            ASICategory.AGENT_GOAL_HIJACK,
    "tool_hijacking":            ASICategory.TOOL_MISUSE,
    "privilege_escalation":      ASICategory.AGENT_IDENTITY_PRIVILEGE_ABUSE,
    "dependency_risk":           ASICategory.SUPPLY_CHAIN,
    "code_injection":            ASICategory.UNEXPECTED_CODE_EXECUTION,
    "memory_poisoning":          ASICategory.MEMORY_CONTEXT_POISONING,
    "inter_agent_mitm":          ASICategory.INSECURE_INTER_AGENT_COMMUNICATION,
    "cascading_agent_failure":   ASICategory.CASCADING_FAILURES,
    "trust_boundary":            ASICategory.HUMAN_AGENT_TRUST,
    "rogue_agent":               ASICategory.ROGUE_AGENTS,
}


@dataclass
class ASIFinding:
    category: ASICategory
    severity: str                   # "info" | "low" | "medium" | "high" | "critical"
    component: str
    description: str
    remediation: str = ""
    least_agency_violation: bool = False


@dataclass
class ASIAuditReport:
    findings: List[ASIFinding] = field(default_factory=list)
    checks_run: List[ASICategory] = field(default_factory=list)

    @property
    def coverage(self) -> Dict[ASICategory, bool]:
        """Map of ASI category → whether at least one check ran for it."""
        return {c: (c in self.checks_run) for c in ASICategory}

    @property
    def coverage_ratio(self) -> float:
        return sum(1 for v in self.coverage.values() if v) / len(ASICategory)

    def by_category(self, category: ASICategory) -> List[ASIFinding]:
        return [f for f in self.findings if f.category == category]

    def summary(self) -> Dict[str, int]:
        out = {c.value: 0 for c in ASICategory}
        for f in self.findings:
            out[f.category.value] += 1
        return out


# Check = callable returning a list of findings.
ASICheck = Callable[[], List[ASIFinding]]


class OWASPASIAuditor:
    """Runs registered checks and aggregates findings by ASI category."""

    def __init__(self) -> None:
        self._checks: Dict[ASICategory, List[ASICheck]] = {c: [] for c in ASICategory}

    def register(self, category: ASICategory | str, check: ASICheck) -> None:
        cat = resolve_asi_category(category)
        self._checks[cat].append(check)

    def run(self, categories: Optional[List[ASICategory]] = None) -> ASIAuditReport:
        targets = categories or list(ASICategory)
        report = ASIAuditReport()
        for cat in targets:
            for check in self._checks.get(cat, []):
                report.checks_run.append(cat)
                for finding in check():
                    if finding.category != cat:
                        # Preserve the declared category on the finding but
                        # still record the check as belonging to `cat`.
                        pass
                    report.findings.append(finding)
        # Dedupe checks_run preserving order
        seen: set[ASICategory] = set()
        unique: List[ASICategory] = []
        for c in report.checks_run:
            if c not in seen:
                seen.add(c)
                unique.append(c)
        report.checks_run = unique
        return report


def resolve_asi_category(category: ASICategory | str) -> ASICategory:
    """Accept either an ASICategory, an ASI## string, or a deprecated alias."""
    if isinstance(category, ASICategory):
        return category
    if category in DEPRECATED_ASI_ALIASES:
        return DEPRECATED_ASI_ALIASES[category]
    # Try enum value (e.g. "ASI07")
    for member in ASICategory:
        if member.value == category:
            return member
    raise ValueError(f"Unknown ASI category: {category!r}")


# ---------------------------------------------------------------------------
# Least Agency principle
# ---------------------------------------------------------------------------

@dataclass
class LeastAgencyContext:
    """Declarative boundary for an agent/tool capability."""
    actor: str
    allowed_tools: List[str] = field(default_factory=list)
    allowed_scopes: List[str] = field(default_factory=list)
    allowed_actions: List[str] = field(default_factory=list)
    allowed_data_classes: List[str] = field(default_factory=list)


def assert_least_agency(
    ctx: LeastAgencyContext,
    *,
    tool: Optional[str] = None,
    scope: Optional[str] = None,
    action: Optional[str] = None,
    data_class: Optional[str] = None,
) -> None:
    """Raise PermissionError if the requested operation exceeds the declared envelope.

    Empty allow-lists are interpreted as "unrestricted" — callers must declare
    explicit restrictions to enforce least-agency. This is additive: existing
    callers that don't adopt LeastAgencyContext continue to work.
    """
    if tool is not None and ctx.allowed_tools and tool not in ctx.allowed_tools:
        raise PermissionError(
            f"least_agency: actor={ctx.actor!r} attempted tool={tool!r} "
            f"outside allowed set {ctx.allowed_tools}"
        )
    if scope is not None and ctx.allowed_scopes and scope not in ctx.allowed_scopes:
        raise PermissionError(
            f"least_agency: actor={ctx.actor!r} attempted scope={scope!r} "
            f"outside allowed set {ctx.allowed_scopes}"
        )
    if action is not None and ctx.allowed_actions and action not in ctx.allowed_actions:
        raise PermissionError(
            f"least_agency: actor={ctx.actor!r} attempted action={action!r} "
            f"outside allowed set {ctx.allowed_actions}"
        )
    if (
        data_class is not None
        and ctx.allowed_data_classes
        and data_class not in ctx.allowed_data_classes
    ):
        raise PermissionError(
            f"least_agency: actor={ctx.actor!r} attempted data_class={data_class!r} "
            f"outside allowed set {ctx.allowed_data_classes}"
        )
