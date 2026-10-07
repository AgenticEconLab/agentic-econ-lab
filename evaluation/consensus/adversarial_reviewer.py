# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Adversarial Reviewer — Critic agent for probing workflow output weaknesses.

Generates targeted adversarial critiques to identify:
- Hallucinated citations or facts
- Logical inconsistencies between stages
- Missing critical information
- Overstatements or unsupported claims
- Methodology-theory mismatches

Returns an AdversarialReport with specific issues found and a penalty score.

Usage:
    from evaluation.consensus.adversarial_reviewer import AdversarialReviewer

    reviewer = AdversarialReviewer(llm_client=client)
    report = reviewer.review(team, mode, outputs, execution_log)
    print(f"Issues found: {report.total_issues}, Penalty: {report.penalty:.3f}")
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field


class AdversarialIssue(BaseModel):
    """A single issue identified by adversarial review."""

    category: str = Field(description="Issue category (e.g., hallucination, inconsistency)")
    severity: str = Field(description="Severity: critical, major, minor")
    description: str = Field(description="Description of the issue")
    evidence: str = Field(default="", description="Evidence from the outputs")
    affected_stage: str = Field(default="", description="Stage affected")


class AdversarialReport(BaseModel):
    """Aggregated adversarial review report."""

    team: str
    mode: str
    issues: List[AdversarialIssue] = Field(default_factory=list)
    total_issues: int = 0
    critical_count: int = 0
    major_count: int = 0
    minor_count: int = 0
    penalty: float = Field(
        default=0.0,
        description="Score penalty from adversarial findings (0.0 = no penalty, 1.0 = maximum)",
    )
    reviewer_model: str = ""
    raw_response: str = Field(default="", exclude=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "team": self.team,
            "mode": self.mode,
            "total_issues": self.total_issues,
            "critical": self.critical_count,
            "major": self.major_count,
            "minor": self.minor_count,
            "penalty": round(self.penalty, 4),
            "reviewer_model": self.reviewer_model,
            "issues": [issue.model_dump() for issue in self.issues],
        }


# Severity weights for penalty calculation
_SEVERITY_WEIGHTS = {"critical": 0.15, "major": 0.08, "minor": 0.03}


class AdversarialReviewer:
    """
    Critic agent that probes workflow outputs for weaknesses.

    Uses an LLM to adversarially examine outputs and identify issues
    that standard evaluation might miss.
    """

    def __init__(
        self,
        llm_invoke_fn: Optional[Callable[[str], str]] = None,
        model: str = "vllm/mistral-small-3.2-24b-fp8",
    ):
        """
        Args:
            llm_invoke_fn: Function that takes a prompt string and returns
                the LLM response. If None, creates a default LLMClient.
            model: Model name (used for reporting; actual model determined
                by llm_invoke_fn).
        """
        self._invoke = llm_invoke_fn
        self._model = model

    def review(
        self,
        team: str,
        mode: str,
        outputs: Dict[str, Any],
        execution_log: Optional[Dict[str, Any]] = None,
    ) -> AdversarialReport:
        """
        Run adversarial review on workflow outputs.

        Args:
            team: Team name.
            mode: Mode name.
            outputs: Dict of output filename -> content.
            execution_log: Optional execution log for context.

        Returns:
            AdversarialReport with identified issues and penalty.
        """
        if self._invoke is None:
            return AdversarialReport(
                team=team,
                mode=mode,
                reviewer_model=self._model,
            )

        prompt = self._build_prompt(team, mode, outputs, execution_log)
        response = self._invoke(prompt)
        issues = self._parse_response(response)

        # Compute penalty
        critical = sum(1 for i in issues if i.severity == "critical")
        major = sum(1 for i in issues if i.severity == "major")
        minor = sum(1 for i in issues if i.severity == "minor")

        penalty = min(
            1.0,
            critical * _SEVERITY_WEIGHTS["critical"]
            + major * _SEVERITY_WEIGHTS["major"]
            + minor * _SEVERITY_WEIGHTS["minor"],
        )

        return AdversarialReport(
            team=team,
            mode=mode,
            issues=issues,
            total_issues=len(issues),
            critical_count=critical,
            major_count=major,
            minor_count=minor,
            penalty=penalty,
            reviewer_model=self._model,
            raw_response=response,
        )

    def _build_prompt(
        self,
        team: str,
        mode: str,
        outputs: Dict[str, Any],
        execution_log: Optional[Dict[str, Any]],
    ) -> str:
        """Build the adversarial review prompt."""
        output_text = self._format_outputs(outputs)

        context_text = ""
        if execution_log:
            meta = execution_log.get("metadata", {})
            topic = meta.get(
                "research_topic", meta.get("research_question", "Not specified")
            )
            context_text = f"Research Topic: {topic}\n"
            stages = execution_log.get("stages", [])
            if stages:
                stage_names = [s.get("name", "?") for s in stages]
                context_text += f"Stages executed: {', '.join(stage_names)}\n"

        return f"""You are an adversarial reviewer tasked with finding weaknesses, errors, and problems in AI-generated economics research outputs.

Your job is to be CRITICAL and THOROUGH. Look for:
1. **Hallucinated citations**: Papers, authors, or facts that appear fabricated
2. **Logical inconsistencies**: Contradictions between different parts of the output
3. **Missing critical information**: Important aspects that should be addressed but aren't
4. **Unsupported claims**: Assertions without evidence or proper reasoning
5. **Methodology-theory mismatches**: Methods that don't align with stated frameworks
6. **Terminology misuse**: Economics terms used incorrectly

## Workflow Context
Team: {team}
Mode: {mode}
{context_text}

## Outputs to Review
{output_text}

## Required JSON Output
Return a JSON object with this structure:
{{
  "issues": [
    {{
      "category": "hallucination|inconsistency|missing_info|unsupported_claim|methodology_mismatch|terminology_misuse",
      "severity": "critical|major|minor",
      "description": "Clear description of the issue",
      "evidence": "Quote or reference from the outputs",
      "affected_stage": "Which stage is affected (if identifiable)"
    }}
  ]
}}

If no issues are found (unlikely), return {{"issues": []}}.
Be thorough but fair. Only report genuine issues, not style preferences."""

    def _format_outputs(self, outputs: Dict[str, Any], max_chars: int = 20000) -> str:
        """Format outputs for the prompt."""
        sections = []
        total_chars = 0
        for filename, content in outputs.items():
            if total_chars >= max_chars:
                sections.append("[Additional files truncated]")
                break
            text = json.dumps(content, indent=2, default=str) if not isinstance(content, str) else content
            remaining = max_chars - total_chars
            if len(text) > remaining:
                text = text[:remaining] + "\n... [truncated]"
            sections.append(f"### {filename}\n```json\n{text}\n```")
            total_chars += len(text)
        return "\n\n".join(sections)

    def _parse_response(self, response: str) -> List[AdversarialIssue]:
        """Parse LLM response into AdversarialIssue objects."""
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            # Try extracting JSON from markdown
            import re

            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                except json.JSONDecodeError:
                    return []
            else:
                return []

        issues = []
        for item in data.get("issues", []):
            severity = item.get("severity", "minor").lower()
            if severity not in ("critical", "major", "minor"):
                severity = "minor"
            category = item.get("category", "other")

            issues.append(
                AdversarialIssue(
                    category=category,
                    severity=severity,
                    description=item.get("description", ""),
                    evidence=item.get("evidence", ""),
                    affected_stage=item.get("affected_stage", ""),
                )
            )

        return issues
