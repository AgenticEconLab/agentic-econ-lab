# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
AdversarialFalsifier — DASES-style counterexample generator (V0.7).

Reference: "Let the Abyss Stare Back: Adaptive Falsification for Autonomous
Scientific Discovery" (arXiv 2603.29045, Mar 30 2026). Pairs with the
existing Theorist agent in ModelTeam's TheoryStage — the Falsifier co-evolves
a set of admissible counterexamples that would invalidate the Theorist's
claim.

The V0.7 reference implementation is rule-based and deterministic: it looks
for claims of the form "X implies Y", "X causes Y", "monotone", "always",
"never", etc., and produces canned counterexample templates. This keeps the
Falsifier testable and CI-safe without requiring an LLM round-trip. When an
LLM client is passed, the Falsifier extends its templates with LLM-generated
counterexamples, capped by ``max_rounds``.

Admissibility gate (the part that makes the verdict meaningful)
---------------------------------------------------------------
An LLM "counterexample" is only allowed to *falsify* a claim when it is
**admissible**: it must (a) be a concrete, non-refusal scenario, (b) be
self-reported by the model as directly contradicting the claim *within the
claim's stated scope*, with non-trivial confidence, AND (c) target a claim that
a single counterexample can actually refute — i.e. a universal / absolute claim
rather than a hedged / probabilistic one ("often", "tends to", "on average").
This independent, code-side check is what stops the Falsifier from rubber-
stamping every claim as ``falsified`` just because the LLM always emits *some*
text. When no admissible counterexample is found the verdict is
``corroborated`` (an adversary actively tried and failed) or ``survives`` (no
adversary ran).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

# Reuse the shared LLM-output robustness helpers. These live behind the heavy
# ``shared`` package (which pulls httpx via shared.llm); since this module is
# only ever imported through ``shared.verification`` — which already depends on
# that chain — the import is free in production. The ImportError fallback keeps
# the module importable in stripped-down contexts (and for isolated unit tests).
try:  # pragma: no cover - exercised via the package import path
    from shared.json_repair import safe_json_loads
except ImportError:  # pragma: no cover
    import json as _json

    def safe_json_loads(text: str, fallback: Any = None) -> Any:  # type: ignore[misc]
        try:
            return _json.loads(text)
        except Exception:
            return {} if fallback is None else fallback

try:  # pragma: no cover - exercised via the package import path
    from shared.schema_coerce import LLMCoercedModel
except ImportError:  # pragma: no cover
    from pydantic import BaseModel as LLMCoercedModel  # type: ignore[assignment]


FalsifierVerdict = Literal[
    "survives", "corroborated", "needs_revision", "falsified"
]


@dataclass
class FalsifierReport:
    claim: str
    admissible_counterexamples: List[Dict[str, Any]] = field(default_factory=list)
    rejected_counterexamples: List[Dict[str, Any]] = field(default_factory=list)
    mechanistic_critique: str = ""
    suggested_revisions: List[str] = field(default_factory=list)
    verdict: FalsifierVerdict = "survives"
    rounds: int = 0


class _CounterexampleProbe(LLMCoercedModel):
    """Structured shape we ask the LLM to return for one counterexample probe.

    Built on :class:`LLMCoercedModel` so a slightly-off LLM JSON (a dict where a
    str is expected, a numeric string for ``confidence``, etc.) is coerced
    rather than raising. Defaults make every field optional so a partial object
    still parses — a missing/empty counterexample simply fails the admissibility
    gate (the safe direction: do not falsify)."""

    counterexample: str = ""
    claim_scope: str = ""
    contradicts_within_scope: bool = False
    is_universal_claim: bool = False
    confidence: float = 0.0
    reasoning: str = ""


# Universal / absolute markers — a single counterexample CAN refute these.
_UNIVERSAL_RE = re.compile(
    r"\b(always|never|all|every|each|any|none|necessarily|invariably|"
    r"guarantee[sd]?|ensure[sd]?|must|cannot|impossible|for all|in every case|"
    r"whenever|no\s+\w+\s+(?:can|ever))\b",
    re.IGNORECASE,
)
# Hedged / probabilistic markers — a single counterexample does NOT refute these
# (one exception is consistent with the claim), so they are not admissibly
# falsifiable by a lone instance.
_HEDGE_RE = re.compile(
    r"\b(often|usually|typically|generally|frequently|sometimes|occasionally|"
    r"tends?\s+to|tend\s+to|in\s+most\s+cases|on\s+average|approximately|"
    r"roughly|mostly|largely|in\s+general|by\s+and\s+large)\b",
    re.IGNORECASE,
)
# Non-answers / refusals the model may emit instead of a real counterexample.
_REFUSAL_RE = re.compile(
    r"^\s*(n/?a|none|no\s+counterexample|there\s+is\s+no\s+counterexample|"
    r"i\s+(?:cannot|can'?t|am\s+unable)|cannot\s+(?:provide|find)|"
    r"unable\s+to|not\s+applicable|as\s+an\s+ai)\b",
    re.IGNORECASE,
)

_MIN_COUNTEREXAMPLE_LEN = 12


def _claim_admits_single_counterexample(claim: str, llm_says_universal: bool) -> bool:
    """Can a *single* counterexample falsify this claim?

    True only for universal / absolute claims that are not hedged. A claim that
    is explicitly probabilistic ("often", "on average") is NOT refuted by one
    exception, so a lone counterexample against it is inadmissible. The claim is
    treated as universal if either our marker regex fires or the LLM (which read
    the full claim) reports it as universal."""
    if _HEDGE_RE.search(claim):
        return False
    return bool(llm_says_universal) or bool(_UNIVERSAL_RE.search(claim))


# Rule patterns: (claim-pattern, counterexample-template, revision-suggestion)
_RULE_PATTERNS: List[tuple[re.Pattern[str], str, str]] = [
    (
        re.compile(r"\b(always|every|all)\b", re.IGNORECASE),
        "universal quantifier \\1 — supply a counterexample",
        "qualify the claim with 'in most cases' or specify the support",
    ),
    (
        re.compile(r"\bnever\b", re.IGNORECASE),
        "negation universal — produce one instance where the opposite holds",
        "bound the claim temporally or conditionally",
    ),
    (
        re.compile(r"\bmonotone(?:ally)?\s+(increas|decreas)", re.IGNORECASE),
        "monotonicity — construct non-monotone regime via parameter shift",
        "identify the regime boundaries where monotonicity breaks",
    ),
    (
        re.compile(r"\b(cause|causes|causal)\b", re.IGNORECASE),
        "unmeasured confounder — omitted variable bias invalidates causal reading",
        "state the adjustment set; use CausalFMClient or DoWhy for identification",
    ),
    (
        re.compile(r"\b(converges?|converg(?:ence|ent))\b", re.IGNORECASE),
        "convergence — instability under small perturbation of initial conditions",
        "document the basin of attraction and Lyapunov condition",
    ),
    (
        re.compile(r"\b(proves|proof|proven)\b", re.IGNORECASE),
        "appeal to proof without derivation — requires formal derivation",
        "supply the derivation or cite a verified source",
    ),
    (
        re.compile(r"\brational(?:ity| agent)?\b", re.IGNORECASE),
        "rational-agent assumption — bounded rationality counterexample (DRL agents)",
        "acknowledge the bounded-rationality alternative (BoE SWP 1142)",
    ),
]


class AdversarialFalsifier:
    """Co-evolving Falsifier paired with Theorist.

    Parameters
    ----------
    max_rounds
        Hard cap on iteration rounds (BudgetController-friendly). Default 2.
    llm_client
        Optional LLMClient; when provided, the Falsifier performs up to
        ``max_rounds`` LLM-assisted counterexample-generation passes.
    collector
        Optional MetricsCollector for observability.
    min_confidence
        Minimum self-reported confidence (0-1) an LLM counterexample must carry
        before it is eligible to be admissible. Default 0.5.
    """

    def __init__(
        self,
        *,
        max_rounds: int = 2,
        llm_client: Any = None,
        collector: Any = None,
        min_confidence: float = 0.5,
    ) -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds must be >= 1")
        self.max_rounds = max_rounds
        self.llm_client = llm_client
        self.collector = collector
        self.min_confidence = min_confidence

    def run(self, claim: str, context: Optional[Dict[str, Any]] = None) -> FalsifierReport:
        report = FalsifierReport(claim=claim)
        ctx = context or {}
        llm_ran = self.llm_client is not None

        # Round 1 — rule-based counterexample scan
        for pattern, tmpl, revision in _RULE_PATTERNS:
            if pattern.search(claim):
                report.admissible_counterexamples.append({
                    "source": "rule",
                    "trigger": pattern.pattern,
                    "counterexample": pattern.sub(tmpl, claim, count=1),
                })
                if revision and revision not in report.suggested_revisions:
                    report.suggested_revisions.append(revision)

        report.rounds = 1

        # Optional LLM-assisted rounds — propose, then VALIDATE admissibility
        # before letting a counterexample drive the verdict.
        if llm_ran:
            for _ in range(self.max_rounds - 1):
                report.rounds += 1
                extra = self._llm_counterexample(claim, ctx)
                if not extra:
                    continue
                if extra.get("admissible"):
                    report.admissible_counterexamples.append(extra)
                    # Feed the accepted counterexample back as a concrete
                    # Theorist revision note.
                    ce = extra.get("counterexample", "").strip()
                    if ce:
                        note = (
                            "restrict the claim's scope to exclude this admissible "
                            f"counterexample, or weaken its universal quantifier: {ce}"
                        )
                        if note not in report.suggested_revisions:
                            report.suggested_revisions.append(note)
                else:
                    report.rejected_counterexamples.append(extra)

        # Compose mechanistic critique
        n_adm = len(report.admissible_counterexamples)
        n_rej = len(report.rejected_counterexamples)
        if n_adm:
            report.mechanistic_critique = (
                f"Claim challenged by {n_adm} admissible counterexample(s) across "
                f"{report.rounds} round(s); see suggested_revisions."
            )
        elif n_rej:
            report.mechanistic_critique = (
                f"No admissible counterexample after {report.rounds} round(s); "
                f"{n_rej} candidate(s) rejected as out-of-scope or non-contradicting."
            )
        else:
            report.mechanistic_critique = (
                "No counterexample found; claim withstood the probe."
            )

        report.verdict = self._decide_verdict(report, llm_ran=llm_ran)
        return report

    # ------------------------------------------------------------------

    def _decide_verdict(self, report: FalsifierReport, *, llm_ran: bool) -> FalsifierVerdict:
        """Map (admissible counterexamples, rule matches, adversary-ran) → verdict.

        An *admissible* counterexample is a validated in-scope contradiction and
        is sufficient to falsify. Absent that:
        - LLM source present  -> single admissible LLM counterexample falsifies.
        - rule matches only   -> the claim's wording is loose but no real
          contradiction was found: ``needs_revision`` when an LLM adversary
          actually probed it; otherwise fall back to the rule-count heuristic
          (>=2 distinct structural objections -> ``falsified``).
        - nothing at all      -> ``corroborated`` if an adversary ran and failed,
          else ``survives`` (only a rule probe ran)."""
        has_admissible_llm = any(
            c.get("source") == "llm" for c in report.admissible_counterexamples
        )
        if has_admissible_llm:
            return "falsified"

        rule_ces = [
            c for c in report.admissible_counterexamples if c.get("source") == "rule"
        ]
        if not report.admissible_counterexamples:
            return "corroborated" if llm_ran else "survives"

        # Only rule-based structural objections remain.
        if llm_ran:
            return "needs_revision"
        return "falsified" if len(rule_ces) >= 2 else "needs_revision"

    def _llm_counterexample(self, claim: str, ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Probe the LLM for a counterexample and validate its admissibility.

        Returns a dict with an ``admissible`` flag (True only when the proposed
        counterexample is a concrete, in-scope, contradicting scenario for a
        single-instance-falsifiable claim), or ``None`` on call failure / empty
        response. Non-admissible candidates are still returned (admissible=False)
        so the caller can record them for transparency."""
        try:
            raw = self.llm_client.invoke(self._build_prompt(claim))  # type: ignore[attr-defined]
        except Exception:
            return None
        if not isinstance(raw, str) or not raw.strip():
            return None

        data = safe_json_loads(raw, fallback={})
        probe: Optional[_CounterexampleProbe] = None
        if isinstance(data, dict) and data:
            try:
                probe = _CounterexampleProbe(**data)
            except Exception:
                probe = None

        if probe is None:
            # Unstructured / unparseable response: record the raw text but do NOT
            # treat it as a validated contradiction.
            return {
                "source": "llm",
                "counterexample": raw.strip()[:600],
                "admissible": False,
                "rejection_reason": "unparseable_probe",
            }

        ce_text = (probe.counterexample or "").strip()
        admissible, reason = self._validate_counterexample(claim, ce_text, probe)
        entry: Dict[str, Any] = {
            "source": "llm",
            "counterexample": ce_text or "(no in-scope counterexample proposed)",
            "claim_scope": (probe.claim_scope or "").strip(),
            "confidence": float(probe.confidence or 0.0),
            "admissible": admissible,
        }
        if not admissible:
            entry["rejection_reason"] = reason
        return entry

    def _validate_counterexample(
        self, claim: str, ce_text: str, probe: "_CounterexampleProbe"
    ) -> tuple[bool, str]:
        """Real accept/reject criterion for an LLM-proposed counterexample.

        Accept only when ALL hold:
        1. a concrete, non-refusal counterexample of meaningful length exists;
        2. the model reports it directly contradicts the claim within its scope;
        3. confidence clears ``min_confidence``;
        4. the claim is one a single counterexample can falsify (universal /
           absolute, not hedged) — checked independently of the model."""
        if not ce_text or len(ce_text) < _MIN_COUNTEREXAMPLE_LEN:
            return False, "no_counterexample_proposed"
        if _REFUSAL_RE.search(ce_text):
            return False, "refusal_or_non_answer"
        if not probe.contradicts_within_scope:
            return False, "does_not_contradict_within_scope"
        if float(probe.confidence or 0.0) < self.min_confidence:
            return False, "low_confidence"
        if not _claim_admits_single_counterexample(claim, probe.is_universal_claim):
            return False, "claim_not_falsifiable_by_single_instance"
        return True, ""

    @staticmethod
    def _build_prompt(claim: str) -> str:
        """Strict-falsifier prompt asking for a *structured* admissibility verdict."""
        return (
            "You are a STRICT adversarial falsifier in the Popperian tradition. "
            "Decide whether a SINGLE concrete counterexample can refute the claim "
            "below, and if so, produce one.\n\n"
            f"CLAIM: {claim}\n\n"
            "Reason carefully:\n"
            "1. Identify the claim's SCOPE and whether it is UNIVERSAL/absolute "
            "(words like 'all', 'always', 'never', 'necessarily', 'guarantees') "
            "or HEDGED/probabilistic ('often', 'tends to', 'in most cases', 'on "
            "average'). A single counterexample can refute a universal claim but "
            "CANNOT refute a hedged one.\n"
            "2. Propose exactly ONE concrete scenario that lies WITHIN the claim's "
            "stated scope and in which the claim is FALSE. It must DIRECTLY "
            "contradict the claim, not merely raise a caveat, a measurement issue, "
            "or an out-of-scope edge case.\n"
            "3. If no such in-scope contradicting scenario exists — because the "
            "claim is hedged, tautological, definitional, or genuinely robust — "
            "you MUST return an empty counterexample and set "
            "contradicts_within_scope to false. Do NOT invent a weak or "
            "out-of-scope objection just to have something.\n\n"
            "Return ONLY a JSON object with exactly these keys:\n"
            '{"claim_scope": str, "is_universal_claim": bool, '
            '"counterexample": str (empty if none), '
            '"contradicts_within_scope": bool, "confidence": number 0.0-1.0, '
            '"reasoning": str}'
        )
