# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Prompt Registry — Externalized prompt management (V0.6 Phase 4, OWASP ASI01).

Centralizes prompt templates for governance, versioning, and goal alignment
validation. Defends against agent goal hijacking by validating prompts
against expected agent goals.

Usage:
    from shared.security.prompt_registry import PromptRegistry

    registry = PromptRegistry()
    registry.register("ideator_system", "You are Ideator...", version="1.0")
    prompt = registry.get("ideator_system", topic="AI economics")
    score = registry.validate_goal_alignment(prompt, "generate research ideas")
"""

import re
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PromptTemplate(BaseModel):
    """A versioned prompt template."""

    prompt_id: str
    template: str
    version: str = "1.0"
    description: str = ""
    expected_goal: str = ""
    created_at: str = ""
    variables: List[str] = Field(default_factory=list)


class PromptRegistry:
    """Externalized prompt management for governance and A/B testing.

    Args:
        prompts: Optional dict of pre-loaded prompts.
    """

    def __init__(self, prompts: Optional[Dict[str, PromptTemplate]] = None):
        self._prompts: Dict[str, PromptTemplate] = prompts or {}
        self._history: List[Dict[str, str]] = []

    def register(
        self,
        prompt_id: str,
        template: str,
        version: str = "1.0",
        description: str = "",
        expected_goal: str = "",
    ) -> None:
        """Register a new prompt template.

        Args:
            prompt_id: Unique identifier for the prompt.
            template: Template string with {variable} placeholders.
            version: Version string.
            description: Human-readable description.
            expected_goal: Expected agent goal for alignment checking.
        """
        # Extract variables from template
        variables = re.findall(r"\{(\w+)\}", template)

        self._prompts[prompt_id] = PromptTemplate(
            prompt_id=prompt_id,
            template=template,
            version=version,
            description=description,
            expected_goal=expected_goal,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            variables=variables,
        )

    def get(self, prompt_id: str, version: str = "latest", **variables) -> str:
        """Get prompt by ID with variable substitution.

        Args:
            prompt_id: Prompt identifier.
            version: Version to retrieve ("latest" for most recent).
            **variables: Template variable values.

        Returns:
            Rendered prompt string.

        Raises:
            KeyError: If prompt_id not found.
        """
        if prompt_id not in self._prompts:
            raise KeyError(f"Prompt not found: {prompt_id}")

        template = self._prompts[prompt_id]
        result = template.template

        # Substitute variables
        for key, value in variables.items():
            result = result.replace(f"{{{key}}}", str(value))

        self._history.append({
            "prompt_id": prompt_id,
            "version": template.version,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })

        return result

    def validate_goal_alignment(self, prompt: str, expected_goal: str) -> float:
        """Check if prompt aligns with expected agent goal (ASI01 defense).

        Uses keyword overlap heuristic to detect goal drift.
        In production, this could use an LLM for deeper analysis.

        Args:
            prompt: The prompt text to validate.
            expected_goal: Description of the expected goal.

        Returns:
            Alignment score 0.0-1.0 (higher = better aligned).
        """
        if not prompt or not expected_goal:
            return 0.0

        # Keyword overlap heuristic
        prompt_words = set(prompt.lower().split())
        goal_words = set(expected_goal.lower().split())

        if not goal_words:
            return 0.0

        overlap = prompt_words & goal_words
        return len(overlap) / len(goal_words)

    def list_prompts(self) -> List[str]:
        """List all registered prompt IDs."""
        return list(self._prompts.keys())

    def get_template(self, prompt_id: str) -> Optional[PromptTemplate]:
        """Get the raw template object."""
        return self._prompts.get(prompt_id)

    def get_usage_history(self) -> List[Dict[str, str]]:
        """Get prompt usage history for auditing."""
        return list(self._history)
