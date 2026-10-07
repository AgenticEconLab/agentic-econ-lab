# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
AvailabilityScout — grade each DRS requirement against what a tier supplies.

The scout consumes the requirements and the series a tier can offer (in the real
pipeline, the series discovered/retrieved by the DataSourceStage) and returns a
verdict per requirement: FEASIBLE (a good direct match), PROXY_ONLY (only an
approximate series — recorded as a proxy candidate with a bias caveat), or
UNAVAILABLE.

Matching is deterministic (token-overlap similarity) so the scout runs offline and
is unit-testable. An optional ``assess_fn`` seam lets a later version delegate the
verdict + proxy/bias reasoning to an LLM without changing callers.
"""

from __future__ import annotations

import re
from typing import Callable, List, Optional

from pydantic import BaseModel, Field

from DataTeam.ael.feasibility.contracts import (
    AvailabilityFinding,
    DataRequirementsSpec,
    ProxyCandidate,
    RequirementVerdict,
)

_WORD = re.compile(r"[a-z0-9]+")
# Generic tokens that shouldn't drive a match (they appear in almost every name).
_STOP = {"rate", "index", "total", "real", "nominal", "us", "usa", "annual", "data", "of", "the"}


def _tokens(text: str) -> set:
    return {t for t in _WORD.findall((text or "").lower()) if t not in _STOP and len(t) > 1}


# Canonical economic concepts. Token-Jaccard alone scores "Annualized Inflation Rate" vs
# "Consumer Price Index for All Urban Consumers" at 0.0, so canonical series would never match
# their concepts and the loop would churn narrow/MRR on satisfiable requirements.
# A shared concept sets a PROXY-grade floor (0.5) — deliberately below the direct-match
# threshold (0.6): inflation from CPI is a derivation, a close proxy, not the exact variable.
_CONCEPTS = [
    ("inflation", ("inflation", "cpi", "consumer price", "price level", "deflator")),
    ("policy_rate", ("federal funds", "policy rate", "fedfunds", "central bank rate", "effective funds")),
    ("unemployment", ("unemployment", "jobless", "unrate")),
    ("output", ("gdp", "gross domestic product", "aggregate output", "national output")),
    ("wage", ("wage", "earnings", "labor compensation", "hourly compensation")),
    ("exchange_rate", ("exchange rate", "fx rate", "currency value")),
    ("gov_spending", ("government spending", "government expenditure", "public expenditure", "fiscal expenditure")),
    ("temperature", ("temperature anomaly", "temperature", "climate anomaly")),
    # staples that would otherwise stay unmet-High for lack of a concept
    ("labor_share", ("labor share", "labour share", "labor income share")),
    ("real_rate", ("real interest rate", "real rate")),
    ("risk_free", ("risk-free rate", "risk free rate", "treasury bill", "t-bill")),
    ("wealth_share", ("wealth share", "net worth held", "wealth held by")),
    ("poverty", ("poverty headcount", "poverty gap", "poverty rate")),
    ("income_share", ("income share", "income quintile", "income decile")),
]
_CONCEPT_FLOOR = 0.5


def _concepts(text: str) -> set:
    lowered = (text or "").lower()
    return {name for name, phrases in _CONCEPTS if any(p in lowered for p in phrases)}


def similarity(a: str, b: str) -> float:
    """Token Jaccard similarity in [0, 1], with a proxy-grade floor when both names map to
    the same canonical economic concept (inflation<->CPI, policy rate<->federal funds, ...)."""
    ta, tb = _tokens(a), _tokens(b)
    jac = len(ta & tb) / len(ta | tb) if ta and tb else 0.0
    if _concepts(a) & _concepts(b):
        return max(jac, _CONCEPT_FLOOR)
    return jac


class AvailableSeries(BaseModel):
    """A series a tier can supply (adapter target for DataSourceStage output)."""

    variable_name: str
    source_name: str
    series_id: str = ""
    tier: str = "open_source_api"
    coverage_note: str = ""
    quality: float = Field(default=1.0, ge=0.0, le=1.0, description="0..1 fitness of the series")
    source_title: str = Field(
        default="", description="The source's own title of the retrieved series; empty "
                                "when the series was not retrieved or the source gave none")
    retrieved: bool = Field(
        default=False, description="True only when a successful, non-simulated retrieval "
                                   "record with observations exists for this series")


def _concept_check(requested: str, title: Optional[str], series_id: str = ""):
    from shared.tools.econ_connectors import title_concept_check
    return title_concept_check(requested, title, series_id=series_id)


def _pool_title_check(requested: str, s: "AvailableSeries"):
    """(status, reason) of a pool series' own title vs the requirement; ('', '') when the
    pool row carries no title (selected-but-unretrieved rows) — the name match then stands."""
    if not s.source_title:
        return "", ""
    return _concept_check(requested, s.source_title, s.series_id)


def _hit_concept(requested: str, hit):
    """(status, title, reason) for a fetchable-lookup hit. The pipeline's resolver returns a
    FetchableHit carrying its concept check; a 3-tuple (source, id, title) is checked here; a
    bare (source, id) from an injected lookup is taken as pre-verified (legacy seam)."""
    status = getattr(hit, "concept_status", None)
    if status:
        return status, getattr(hit, "title", None), getattr(hit, "reason", "")
    if len(hit) >= 3:
        s, r = _concept_check(requested, hit[2], str(hit[1]))
        return s, hit[2], r
    return "match", None, ""


# Callable seam: (requirement, available_in_tier, tier) -> AvailabilityFinding
AssessFn = Callable[[object, List[AvailableSeries], str], AvailabilityFinding]

# Seam: human series/variable name -> (source_name, series_id) or None. In the real
# pipeline this is econ_connectors.lookup_known_series — deterministic, offline, curated.
FetchableLookup = Callable[[str], Optional[tuple]]


class AvailabilityScout:
    """Grades requirements against a tier's available series.

    Parameters
    ----------
    match_threshold:
        similarity >= this (with adequate quality) -> FEASIBLE.
    proxy_threshold:
        proxy_threshold <= similarity < match_threshold -> a proxy candidate.
    min_quality:
        a direct match below this quality is downgraded to a proxy, not FEASIBLE.
    assess_fn:
        optional LLM-backed override; when supplied it decides the finding.
    """

    def __init__(
        self,
        match_threshold: float = 0.6,
        proxy_threshold: float = 0.3,
        min_quality: float = 0.5,
        assess_fn: Optional[AssessFn] = None,
        fetchable_lookup: Optional[FetchableLookup] = None,
    ):
        self.match_threshold = match_threshold
        self.proxy_threshold = proxy_threshold
        self.min_quality = min_quality
        self._assess_fn = assess_fn
        self._fetchable_lookup = fetchable_lookup

    def grade(
        self, drs: DataRequirementsSpec, available: List[AvailableSeries], tier: str = "open_source_api"
    ) -> List[AvailabilityFinding]:
        return [self._grade_one(req, available, tier) for req in drs.requirements]

    def _grade_one(self, req, available: List[AvailableSeries], tier: str) -> AvailabilityFinding:
        if self._assess_fn is not None:
            return self._assess_fn(req, available, tier)

        in_tier = [s for s in available if s.tier == tier]
        scored = sorted(
            ((s, similarity(req.variable_name, s.variable_name)) for s in in_tier),
            key=lambda x: x[1],
            reverse=True,
        )

        base = dict(requirement_id=req.requirement_id, variable_name=req.variable_name, priority=req.priority)

        # Direct match: strong similarity AND adequate quality AND a successful retrieval
        # record AND the retrieved series' own title passes the concept check -> FEASIBLE.
        # A selected-but-unretrieved series, or one whose title is missing/unverified, is not
        # a verified match (a 'Wealth Gini' with no retrieval record must not be graded
        # FEASIBLE with no fetch scheduled): it falls through to the verified connector
        # lookup, else stays a proxy with that caveat.
        if (scored and scored[0][1] >= self.match_threshold
                and scored[0][0].quality >= self.min_quality
                and scored[0][0].retrieved
                and _pool_title_check(req.variable_name, scored[0][0])[0] == "match"):
            best = scored[0][0]
            return AvailabilityFinding(
                verdict=RequirementVerdict.FEASIBLE,
                matched_source=best.source_name,
                matched_series=best.series_id,
                coverage_note=best.coverage_note,
                notes=f"direct match (similarity {scored[0][1]:.2f}, quality {best.quality:.2f})",
                **base,
            )

        # The pool is only what the retrieval plan happened to fetch.
        # A requirement absent from the pool but covered by a curated connector id is
        # FEASIBLE — fetchable on demand — not grounds to revise the model (e.g. var:G_t,
        # Government Spending, with FRED/GCE one call away).
        # An exact curated match outranks approximate pool proxies.
        connector_proxy: Optional[ProxyCandidate] = None
        if self._fetchable_lookup is not None:
            try:
                hit = self._fetchable_lookup(req.variable_name)
            except Exception:
                hit = None
            if hit:
                f_source, f_id = hit[0], hit[1]
                status, title, reason = _hit_concept(req.variable_name, hit)
                if status == "match":
                    return AvailabilityFinding(
                        verdict=RequirementVerdict.FEASIBLE,
                        matched_source=f_source,
                        matched_series=f_id,
                        fetchable=True,
                        notes=f"not in retrieved dataset; resolved in the open-connector "
                              f"universe as {f_source}/{f_id}"
                              + (f" ('{title}')" if title else "")
                              + " — supplemental fetch scheduled",
                        **base,
                    )
                # A name hit whose official title is a different (or unverifiable)
                # concept is at best a proxy — 'Tax-to-GDP' is not GDP, 'Wealth Gini' is not
                # the income Gini. Never FEASIBLE, never a supplemental fetch.
                connector_proxy = ProxyCandidate(
                    variable_name=title or f_id, source_name=f_source, series_id=f_id,
                    closeness=0.3 if status == "mismatch" else 0.4,
                    bias_caveat=(
                        f"connector series {f_source}/{f_id} ('{title}') is a different "
                        f"construct: {reason}" if status == "mismatch" else
                        f"connector series {f_source}/{f_id}: its official title could not "
                        f"be verified against '{req.variable_name}' ({reason or 'no title'})"),
                )

        # Otherwise gather proxies: anything above the proxy threshold, plus a
        # strong-but-low-quality match (downgraded from FEASIBLE).
        proxies: List[ProxyCandidate] = [connector_proxy] if connector_proxy else []
        for s, sim in scored:
            if sim < self.proxy_threshold:
                continue
            low_q = sim >= self.match_threshold and s.quality < self.min_quality
            t_status, t_reason = _pool_title_check(req.variable_name, s)
            if t_status == "mismatch":
                caveat = (f"the retrieved series' own title is '{s.source_title}', a different "
                          f"construct: {t_reason}")
            elif not s.retrieved:
                caveat = ("selected but no successful retrieval record — availability and "
                          "concept unverified")
            elif t_status != "match" and sim >= self.match_threshold:
                caveat = (f"retrieved, but its source title could not be verified against "
                          f"'{req.variable_name}' ({t_reason or 'no title'})")
            elif low_q:
                caveat = f"low series fitness (quality {s.quality:.2f})"
            else:
                caveat = f"approximate construct (similarity {sim:.2f}) — not the exact variable"
            proxies.append(ProxyCandidate(
                variable_name=s.variable_name, source_name=s.source_name,
                series_id=s.series_id, closeness=round(sim, 3), bias_caveat=caveat,
            ))

        if proxies:
            return AvailabilityFinding(
                verdict=RequirementVerdict.PROXY_ONLY,
                proxy_candidates=proxies,
                notes=f"{len(proxies)} proxy candidate(s); no exact match in tier '{tier}'",
                **base,
            )

        return AvailabilityFinding(
            verdict=RequirementVerdict.UNAVAILABLE,
            notes=f"no matching or proxy series in tier '{tier}'",
            **base,
        )
