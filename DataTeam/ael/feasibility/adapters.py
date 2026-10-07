# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Adapters between the live ModelTeam/DataTeam artifacts and the feasibility loop.

  * ``drs_from_model_artifacts`` — build the Data Requirements Spec (DRS) from a
    ModelTeam design output (exogenous/observable variables) plus its calibration
    output (every empirical target). This is decision (A): the model's *inputs* and
    the moments it must *match* are exactly what the study needs data for.
  * ``available_series_from_source`` — map a DataTeam DataSourceStage output into the
    ``AvailableSeries`` the scout grades against.

Both take plain dicts (the on-disk JSON artifacts) and tolerate missing keys, so
they never raise on a partial upstream output.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from DataTeam.ael.schemas.stage_outputs import DataRequirement
from DataTeam.ael.feasibility.contracts import DataRequirementsSpec
from DataTeam.ael.feasibility.scout import AvailableSeries

# Model variables that require external data (model inputs / observables), as
# opposed to endogenous/solved variables and calibrated parameters.
_DATA_BEARING_TYPES = {"exogenous", "observable"}

# Unobservable model PRIMITIVES are not data requirements. LLM designs type shocks as
# "exogenous", so name/description matching is needed: a Demand Shock / Signal Noise /
# measurement-error innovation exists in no dataset, and demanding it would make the
# feasibility loop structurally non-convergent (every MRR would ask the model to drop its own
# innovations).
_LATENT_RE = re.compile(
    r"\bshocks?\b|\bnoise\b|\binnovations?\b|\bdisturbances?\b|\bmeasurement error\b|"
    r"\bwhite noise\b|\brandom (walk|term|error)\b|\bi\.?i\.?d\.?\b|\bstochastic (term|error)\b",
    re.IGNORECASE,
)


def _is_latent_innovation(var: dict) -> bool:
    """True for unobservable innovations (shocks/noise/disturbances) — model primitives that
    can never be satisfied by a dataset and must not become data requirements."""
    text = " ".join(str(var.get(k, "") or "") for k in ("variable_name", "description"))
    return bool(_LATENT_RE.search(text))


# Inputs the PIPELINE ITSELF constructs (LLM-extracted features, text-corpus embeddings,
# NLP-derived vectors) are not retrievable series — demanding them as High-essential would make
# the review return MRRs asking the model to drop its own derived features. They stay in the DRS (they ARE real inputs) but as non-essential, clearly labeled, so
# the loop resolves them as a documented construction step instead of churning.
# Conservative match: requires an LLM/NLP/text-processing marker — a plain "consumer sentiment
# index" (a retrievable survey series, e.g. UMich) must NOT match.
_CONSTRUCTED_RE = re.compile(
    r"\bllm\b|large language model|embedding|text corp(us|ora)|"
    r"nlp[- ](derived|extracted|based|pipeline|processed)|"
    r"(derived|extracted|constructed) (from|via) (text|news|documents?|transcripts?|llm)|"
    r"llm[- ](derived|extracted|generated|based)|narrative (vector|representation|extraction)",
    re.IGNORECASE,
)


def _is_pipeline_constructed(var: dict) -> bool:
    """True for LLM/text-derived inputs the pipeline constructs internally (not retrievable)."""
    text = " ".join(str(var.get(k, "") or "") for k in ("variable_name", "description"))
    return bool(_CONSTRUCTED_RE.search(text))


# DERIVED STATISTICS demanded as series. "Output Volatility", "Evasion Gini", "return
# autocorrelation", "Frisch elasticity" are moments/statistics no data catalog carries as a
# series — the scout can never match them, so MRRs would ask models to delete their own
# calibration targets. They are COMPUTABLE by the pipeline (estimation/calibration)
# from base series, so they stay in the DRS as non-essential, clearly labeled. Statistic
# words only — a plain "rate"/"ratio"/"index" is often a genuine retrievable series and must
# NOT match ("Effective Tax Rate", "Tax Revenue to GDP Ratio").
_DERIVED_MOMENT_RE = re.compile(
    r"\bvolatilit(y|ies)\b|\bvariances?\b|\bstandard deviations?\b|\bstd\.? dev\b|"
    r"\bauto-?correlations?\b|\bcorrelations?\b|\bcovariances?\b|\belasticit(y|ies)\b|"
    r"\bgini\b|\bskewness\b|\bkurtosis\b|\bpersistence\b|\bhalf-?li(fe|ves)\b|\bfrisch\b",
    re.IGNORECASE,
)

_DERIVED_MOMENT_TAG = ("[derived statistic: computed by the pipeline from base series "
                       "(estimation/calibration), not directly retrievable]")


def _is_derived_moment(name: str, description: str = "") -> bool:
    """True for moment/statistic requirements (volatility, Gini, elasticities, ...)."""
    return bool(_DERIVED_MOMENT_RE.search(f"{name or ''} {description or ''}"))


# LATENT BEHAVIORAL/PERCEPTION parameters (Perceived Detection Probability, Trust Decay,
# Policy Entropy, Risk-Aversion Premium) are preference/belief primitives — calibration
# parameters, not observables. The innovation filter above only catches shock/noise
# innovations, so these are filtered separately (otherwise they would enter the DRS as
# High-essential data requirements and churn the loop). Conservative
# match: belief/perception/preference markers; a survey series like "Consumer Sentiment
# Index" (retrievable, e.g. UMich) must NOT match.
_BEHAVIORAL_LATENT_RE = re.compile(
    r"\bperceived\b|\bperceptions?\b|\bbeliefs?\b|\bsubjective\b|\btrust (level|decay|dynamics)\b|"
    r"\baggregate trust\b|\bfairness (index|perception)\b|\bperceived fairness\b|"
    r"\brisk[- ]aversion\b|\bentropy\b|\blearning (convergence|rate)\b|\bsocial contagion\b|"
    r"\bdiscount factor\b|\butility (loss|weight|cost)\b",
    re.IGNORECASE,
)

_BEHAVIORAL_LATENT_TAG = ("[latent behavioral parameter: a preference/belief primitive to be "
                          "calibrated or assumed (documented), not a retrievable series]")


def _is_behavioral_latent(name: str, description: str = "") -> bool:
    """True for preference/belief/perception primitives — calibration parameters."""
    return bool(_BEHAVIORAL_LATENT_RE.search(f"{name or ''} {description or ''}"))


# Model artifacts carry raw LaTeX in symbols/names (\mathcal{D}_{text}, \mathbf{z}_t^{LLM})
# — unreadable requirement ids and noise for availability matching. Strip to plain text.
_LATEX_CMD_RE = re.compile(r"\\(mathcal|mathbf|mathrm|mathit|text|hat|bar|tilde|vec)\s*\{([^{}]*)\}")


def _strip_latex(s: str) -> str:
    out = str(s or "")
    for _ in range(3):                                   # nested commands
        out = _LATEX_CMD_RE.sub(r"\2", out)
    out = re.sub(r"[\\${}]", "", out)
    return out.strip()


def _matched_book_moments(calibration: Optional[Dict[str, Any]]) -> List[str]:
    """Moment names the calibration harness matched against the CITED target book — i.e.
    model_moments rows carrying an empirical value (the book's number, never LLM-invented)."""
    out: List[str] = []
    for cm in _iter((calibration or {}).get("calibrated_models")):
        for m in _iter(cm.get("model_moments")):
            name = m.get("moment_name")
            if name and m.get("empirical_value") is not None and name not in out:
                out.append(str(name))
    return out


def _book_moment_for(target_name: str, book_moments: List[str]) -> Optional[str]:
    """The book moment satisfying ``target_name``, by token containment/overlap
    ('capital_output_ratio' <-> 'Steady-State Capital-Output Ratio'), else None."""
    t_tokens = {w for w in re.findall(r"[a-z0-9]+", str(target_name).lower()) if len(w) > 1}
    if not t_tokens:
        return None
    for mk in book_moments:
        m_tokens = {w for w in mk.lower().split("_") if len(w) > 1}
        if not m_tokens:
            continue
        if m_tokens <= t_tokens or (len(m_tokens & t_tokens) / len(m_tokens | t_tokens)) >= 0.5:
            return mk
    return None


def _iter(dicts: Any) -> List[dict]:
    return [d for d in dicts if isinstance(d, dict)] if isinstance(dicts, list) else []


def _target_priority(importance: Any) -> str:
    return {"high": "High", "medium": "Medium", "low": "Low"}.get(
        str(importance or "").strip().lower(), "High"
    )


def _requirement_from_variable(var: dict) -> DataRequirement:
    constructed = _is_pipeline_constructed(var)
    name = _strip_latex(var.get("variable_name", "") or var.get("variable_symbol", ""))
    desc = _strip_latex(var.get("description", ""))
    derived = (not constructed) and _is_derived_moment(name, desc)
    if constructed:
        desc = ("[pipeline-constructible: derived via internal text/LLM processing — "
                "not a retrievable series] " + desc).strip()
    elif derived:
        desc = (_DERIVED_MOMENT_TAG + " " + desc).strip()
    non_essential = constructed or derived
    return DataRequirement(
        requirement_id=f"var:{_strip_latex(var.get('variable_symbol') or var.get('variable_id') or var.get('variable_name', ''))}",
        variable_name=name,
        description=desc,
        frequency="unspecified",
        time_period="unspecified",
        geographic_coverage="unspecified",
        unit_of_measurement=var.get("domain", "") or "unspecified",
        # Constructed/derived inputs are resolved by an internal pipeline step, not
        # by retrieval — non-essential so the loop cannot churn on them.
        priority="Medium" if non_essential else "High",
        suggested_sources=(["pipeline-constructed"] if constructed
                           else ["pipeline-computed"] if derived else []),
    )


def _requirement_from_target(t: dict) -> DataRequirement:
    return DataRequirement(
        requirement_id=f"target:{t.get('target_id') or t.get('target_name', '')}",
        variable_name=t.get("target_name", ""),
        description=t.get("description", ""),
        frequency="unspecified",
        time_period="unspecified",
        geographic_coverage="unspecified",
        unit_of_measurement=t.get("target_type", "") or "unspecified",
        priority=_target_priority(t.get("importance")),
        suggested_sources=[t.get("empirical_source")] if t.get("empirical_source") else [],
    )


def drs_from_model_artifacts(
    research_question: str,
    model_design: Optional[Dict[str, Any]] = None,
    calibration: Optional[Dict[str, Any]] = None,
    cycle: int = 0,
) -> DataRequirementsSpec:
    """DRS = exogenous/observable model variables + every empirical target."""
    requirements: List[DataRequirement] = []
    seen = set()

    def _add(req: DataRequirement) -> None:
        key = (req.variable_name or "").strip().lower()
        if not key or key in seen:
            return
        seen.add(key)
        requirements.append(req)

    source_model = ""
    for fm in _iter((model_design or {}).get("formal_models")):
        source_model = source_model or fm.get("model_title", "")
        for var in _iter(fm.get("variables")):
            _var_text = " ".join(str(var.get(k, "") or "")
                                 for k in ("variable_name", "description"))
            if (str(var.get("variable_type", "")).strip().lower() in _DATA_BEARING_TYPES
                    and not _is_latent_innovation(var)     # shocks/noise are not observables
                    and not _is_behavioral_latent(_var_text)):  # beliefs/preferences either
                _add(_requirement_from_variable(var))

    # A calibration target the deterministic harness ALREADY matched against the cited
    # target book (macro_targets.yaml) is satisfied — it is not a data-retrieval need (an MRR
    # must not ask the model to remove e.g. K/Y or the capital share when the book provides
    # them and the harness matched them). Such targets stay in the DRS for
    # provenance but as non-essential, clearly attributed to the book.
    book_moments = _matched_book_moments(calibration)
    for cm in _iter((calibration or {}).get("calibrated_models")):
        for target in _iter(cm.get("empirical_targets")):
            req = _requirement_from_target(target)
            mk = _book_moment_for(req.variable_name, book_moments)
            if mk:
                req = req.model_copy(update={
                    "priority": "Medium",
                    "description": (f"[satisfied by the cited calibration target book "
                                    f"(moment: {mk})] " + (req.description or "")).strip(),
                })
            # A target that is a derived statistic (computable from base series) or a
            # latent behavioral parameter (calibrated/assumed, documented) is not a retrieval
            # need — non-essential, clearly labeled, so the MRR never demands its removal.
            elif _is_derived_moment(req.variable_name, req.description):
                req = req.model_copy(update={
                    "priority": "Medium",
                    "description": (_DERIVED_MOMENT_TAG + " " + (req.description or "")).strip(),
                })
            elif _is_behavioral_latent(req.variable_name, req.description):
                req = req.model_copy(update={
                    "priority": "Medium",
                    "description": (_BEHAVIORAL_LATENT_TAG + " " + (req.description or "")).strip(),
                })
            _add(req)

    return DataRequirementsSpec(
        research_question=research_question,
        requirements=requirements,
        source_model=source_model,
        cycle=cycle,
    )


def _artifact_payload(obj: Any) -> Any:
    """Unwrap a pipeline artifact envelope ({name, data, ...}) if present."""
    if isinstance(obj, dict) and "data" in obj and "name" in obj:
        return obj.get("data")
    return obj


def requirements_for_retrieval(
    upstream_artifacts: Optional[Dict[str, Any]], research_question: str = "",
) -> List[DataRequirement]:
    """The data requirements the DataTeam's retrieval should serve.

    ``model_specification["data_requirements"]`` alone is not produced by any team, so the
    requirements are resolved from these sources, in order: the ``data_requirements_spec`` artifact (feasibility loop), else the DRS derived
    from the ``model_design`` + ``model_specification`` artifacts (the same deterministic
    adapter the loop uses), else a legacy ``model_specification.data_requirements`` list.
    Requirements tagged as not retrievable (pipeline-constructed inputs, derived statistics,
    latent behavioral parameters, targets satisfied by the cited target book — their
    descriptions start with a ``[...]`` tag) are left out: retrieval cannot serve them."""
    ua = upstream_artifacts or {}
    rows: List[Any] = []
    drs = _artifact_payload(ua.get("data_requirements_spec"))
    if isinstance(drs, dict) and _iter(drs.get("requirements")):
        rows = _iter(drs.get("requirements"))
    else:
        design = _artifact_payload(ua.get("model_design"))
        spec = _artifact_payload(ua.get("model_specification"))
        if isinstance(design, dict) or isinstance(spec, dict):
            derived = drs_from_model_artifacts(
                research_question, design if isinstance(design, dict) else None,
                spec if isinstance(spec, dict) else None)
            rows = [r.model_dump() for r in derived.requirements]
        if not rows and isinstance(spec, dict):
            rows = _iter(spec.get("data_requirements"))
    out: List[DataRequirement] = []
    for r in rows:
        try:
            req = r if isinstance(r, DataRequirement) else DataRequirement(**{
                "requirement_id": str(r.get("requirement_id") or r.get("variable_name") or ""),
                "variable_name": str(r.get("variable_name") or ""),
                "description": str(r.get("description") or ""),
                "frequency": str(r.get("frequency") or "unspecified"),
                "time_period": str(r.get("time_period") or "unspecified"),
                "geographic_coverage": str(r.get("geographic_coverage") or "unspecified"),
                "unit_of_measurement": str(r.get("unit_of_measurement") or "unspecified"),
                "priority": str(r.get("priority") or "Medium"),
                "suggested_sources": [str(x) for x in (r.get("suggested_sources") or [])],
            })
        except Exception:
            continue
        if not req.variable_name or str(req.description).lstrip().startswith("["):
            continue
        out.append(req)
    return out


def format_revision_directive(revision_request: Optional[Dict[str, Any]]) -> str:
    """Render a Model Revision Request into a directive block for the ModelDesign prompt.

    Returns "" when there is no revision, so an unrevised design prompt is byte-identical
    to before. ModelTeam consumes this on a feasibility revision cycle.
    """
    if not revision_request:
        return ""
    reason = str(revision_request.get("reason", "")).strip()
    change = str(revision_request.get("requested_change", "")).strip()
    lines = ["REVISION DIRECTIVE (a prior data-feasibility review requires this model to change):"]
    if reason:
        lines.append(f"- Reason: {reason}")
    if change:
        lines.append(f"- Required change: {change}")
    lines.append(
        "Act on this: remove the named variable(s), or replace each with a construct the "
        "available data can support (e.g. a close proxy), and keep the rest of the model "
        "coherent. Do not reintroduce the flagged variable(s)."
    )
    return "\n".join(lines)


def available_series_from_source(
    data_source_output: Optional[Dict[str, Any]], tier: str = "open_source_api"
) -> List[AvailableSeries]:
    """Map a DataSourceStage output (selected series, else retrieved data) to supply."""
    out = data_source_output or {}
    rows = _iter(out.get("selected_series")) or _iter(out.get("retrieved_data"))
    # What was ACTUALLY retrieved for each id — its own title (concept check) and whether
    # it is a simulated placeholder (a simulation never satisfies a requirement)
    retrieved = {str(r.get("series_id", "")): r for r in _iter(out.get("retrieved_data"))}
    series: List[AvailableSeries] = []
    for s in rows:
        name = s.get("series_name") or s.get("variable_role") or s.get("series_id", "")
        if not name:
            continue
        got = retrieved.get(str(s.get("series_id", ""))) or {}
        simulated = _truthy(got.get("data_simulated"))
        ok = _retrieval_succeeded(got)
        series.append(AvailableSeries(
            variable_name=name,
            source_name=s.get("source_name", ""),
            series_id=s.get("series_id", ""),
            tier=tier,
            coverage_note=s.get("units", "") or s.get("frequency", ""),
            # Quality 1.0 only for a successful retrieval; a selected id with no
            # retrieval record is not evidence of availability
            quality=1.0 if ok else (0.2 if simulated else 0.0),
            source_title=str(got.get("source_title") or ""),
            retrieved=ok,
        ))
    return series


def _truthy(v: Any) -> bool:
    return v is True or str(v).strip().lower() in ("true", "1", "yes")


def _retrieval_succeeded(row: Dict[str, Any]) -> bool:
    """A retrieval record that is real data: present, not simulated, status success (when
    stated) and a positive observation count (when stated)."""
    if not row or _truthy(row.get("data_simulated")):
        return False
    status = str(row.get("retrieval_status") or "success").strip().lower()
    if not status.startswith("success"):
        return False
    n = row.get("num_observations")
    if n not in (None, ""):
        try:
            return float(n) > 0
        except (TypeError, ValueError):
            return False
    return True
