# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""DataTeam HITL checkpoints.

The printed question IS the prompt (a reviewer, human or LLM-Economist committee, never
votes on a blank prompt), the material under review is passed as
context, and the answer is classified so the stage can act on a non-approval (one bounded
revision where a revision is possible) and record it as a limitation that flows downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from shared.auto_input import auto_input

_APPROVALS = {"approved", "approve", "yes", "y", "ok", "okay", "proceed", "accept", "accepted",
              "continue", "confirm", "true", "a", "lgtm", "looks good", "sufficient"}


def is_approval(response: str) -> bool:
    """True for an approving answer. Anything else (a negative token, an alternative, or
    feedback text) is an objection the stage must act on or record."""
    r = (response or "").strip().lower().rstrip(".!")
    if not r:
        return True                     # empty -> the checkpoint's default ("approved")
    if r in _APPROVALS:
        return True
    first = r.split()[0].strip(",.;:!")
    return first in ("approved", "approve", "yes") and not any(
        w in r for w in (" but ", " however", " not ", "n't", " except"))


@dataclass
class CheckpointResult:
    checkpoint: str
    question: str
    response: str
    approved: bool
    revised: bool = False
    final_response: str = ""
    notes: List[str] = field(default_factory=list)

    def to_record(self) -> Dict:
        return {"checkpoint": self.checkpoint, "approved": self.approved,
                "response": self.response, "revised": self.revised,
                "final_response": self.final_response or self.response,
                "notes": self.notes}

    def limitation(self) -> Optional[str]:
        """Disclosure line when the reviewer did not approve (None when approved)."""
        if self.approved:
            return None
        tail = (" after one bounded revision" if self.revised else "")
        return (f"HITL {self.checkpoint}: the reviewer did not approve{tail} "
                f"(answer: '{(self.final_response or self.response)[:200]}'); the stage "
                "proceeded with this objection recorded.")


def ask(checkpoint: str, lines: List[str], decision: List[str], default: str,
        context: Optional[Dict] = None) -> CheckpointResult:
    """Pose one checkpoint. The full printed question is the prompt the reviewer (human or
    committee) answers; ``context`` carries the material under review."""
    question = "\n".join(
        [f"HITL CHECKPOINT — {checkpoint}", *lines, "", "Decision needed:",
         *[f"- {d}" for d in decision],
         f"Answer '{default}' to proceed as is, or state the objection."])
    print("\n" + "=" * 70)
    print(f"🛑 HITL CHECKPOINT: {checkpoint}")
    print("=" * 70)
    response = (auto_input(question, default=default, context=context or {}) or "").strip()
    response = response or default
    return CheckpointResult(checkpoint=checkpoint, question=question, response=response,
                            approved=is_approval(response))
