# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Deterministic report assembly (§4.7 Reporter's structural core).

Every numeric cell in the markdown skeleton is FORMATTED FROM the source artifacts; the LLM
contributes only clearly delimited narrative blocks (intro, discussion), which the Stage-3
consistency checker then verifies number-by-number. The Limitations section is
auto-collected from the pipeline's own disclosures — simulated series, repair notes,
calibration refusals, estimation caveats, feasibility non-convergence — so a reader sees
every shortcut without trusting anyone's narrative."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .types import InterpretationResult

_NARRATIVE_MARK = "<!-- narrative:{slot} -->"


def _fmt(x: Any, nd: int = 4) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}g}"
    return str(x)


def _first_question(research_questions: Dict) -> str:
    for key in ("final_questions", "questions", "prioritized", "prioritized_questions"):
        lst = (research_questions or {}).get(key)
        if isinstance(lst, list) and lst:
            q = lst[0]
            return q.get("question", str(q)) if isinstance(q, dict) else str(q)
    return "(research question unavailable)"


def _iter(x) -> List[dict]:
    return [d for d in x if isinstance(d, dict)] if isinstance(x, list) else []


def _short_title(question: str, limit: int = 120) -> str:
    """A long question is
    shortened at the last word boundary before ``limit`` with an ellipsis; the full question
    is printed in Section 1."""
    q = " ".join(str(question).split())
    if len(q) <= limit:
        return q
    cut = q[:limit].rsplit(" ", 1)[0].rstrip(" ,;:-")
    return (cut or q[:limit]) + "…"


def _tf(transform: Optional[str], ref: str) -> str:
    t = transform or "level"
    return f"{ref}" if t == "level" else f"{t}({ref})"


def _term_core(v: Dict) -> str:
    """The column the harness builds for one variable: transform(series) [− subtrahend]
    [× partner], then the lag (estim_harness.harness.build_design)."""
    core = _tf(v.get("transform"), str(v.get("series_ref", "?")))
    if v.get("subtract_ref"):
        core = (f"[{core} − "
                f"{_tf(v.get('subtract_transform') or v.get('transform'), str(v['subtract_ref']))}]")
    if v.get("interact_with"):
        core = f"{core} × {_tf(v.get('transform'), str(v['interact_with']))}"
    if v.get("lag"):
        core += f", lag {v.get('lag')}"
    return core


def render_spec_terms(spec: Dict) -> List[str]:
    """Names alone hide what a term is (e.g. 'Network Connectivity' as IT.CEL.SETS.P2 ×
    Gini). One line per term: its series, transform, lag,
    subtraction and interaction."""
    spec = spec or {}
    lines = []
    dep = spec.get("dependent") or {}
    if dep:
        lines.append(f"dependent `{dep.get('name')}` = {_term_core(dep)}")
    for v in spec.get("regressors") or []:
        lines.append(f"regressor `{v.get('name')}` = {_term_core(v)}")
    for v in spec.get("instruments") or []:
        lines.append(f"instrument `{v.get('name')}` = {_term_core(v)}")
    if spec.get("include_trend"):
        lines.append("linear trend")
    return lines


def _compact_spec(spec: Dict) -> str:
    spec = spec or {}
    dep = spec.get("dependent") or {}
    rhs = [_term_core(v) for v in spec.get("regressors") or []]
    if spec.get("include_trend"):
        rhs.append("trend")
    return f"{_term_core(dep) if dep else '?'} ~ {' + '.join(rhs) or '?'}"


def _estimation_meta(estimation_results: Dict, key: str):
    """Specification-review records live at the top level of estimation_results (pipeline runner) or in the
    final stage artifact's metadata (standalone runs)."""
    er = estimation_results or {}
    if key in er:
        return er.get(key)
    return (er.get("metadata") or {}).get(key)


def derived_numbers(code_generation: Optional[Dict],
                    estimation_results: Optional[Dict] = None) -> Dict[str, Dict]:
    """Numbers the template DERIVES from artifacts (a sum of refused equations) appear in
    no artifact, so the checker could only match them by coincidence.
    Each derived number is computed here once, with its derivation, stored in the drafting
    artifact, and added to the checker's source pool under the provenance 'derived'."""
    out: Dict[str, Dict] = {}
    gens = (code_generation or {}).get("results") or [] if code_generation else []
    if gens:
        out["code_generation.refused_total"] = {
            "value": sum(len(g.get("refused") or []) for g in gens),
            "derivation": "sum over generated models of len(refused) in code_generation",
        }
    rob = (((estimation_results or {}).get("inference") or {}).get("robustness")) or []
    if rob:
        n_est = sum(1 for c in rob if c.get("variant_estimate") is not None)
        out["robustness.estimable_variants"] = {
            "value": n_est, "derivation": "count of robustness variants with an estimate"}
        out["robustness.not_estimable_variants"] = {
            "value": len(rob) - n_est,
            "derivation": "count of robustness variants without an estimate"}
    return out


def _clip(text: str, n: int = 600) -> str:
    t = " ".join(str(text).split())
    return t if len(t) <= n else t[:n].rsplit(" ", 1)[0] + "…"


def collect_limitations(
    data_source: Dict, model_specification: Dict, estimation_results: Dict,
    feasibility_report: Optional[Dict] = None,
    report_objections: Optional[List[str]] = None,
) -> List[str]:
    """Auto-collect every honest disclosure the pipeline made along the way."""
    lims: List[str] = []
    for item in _iter((data_source or {}).get("retrieved_data")):
        sid = item.get("series_id", "?")
        if item.get("data_simulated"):
            lims.append(f"Series `{sid}` is DISCLOSED SIMULATION (no live retrieval "
                        "succeeded); results touching it are illustrative only.")
        elif "repaired" in str(item.get("quality_notes", "")):
            lims.append(f"Series `{sid}` was retrieved via a disclosed id repair: "
                        f"{str(item.get('quality_notes'))[:160]}")
        if item.get("proxy_for"):
            lims.append(f"Series `{sid}` is a proxy: the source's own series is "
                        f"'{item.get('source_title') or sid}', used for the requested variable "
                        f"'{item.get('proxy_for')}'.")
        elif item.get("title_check") == "unverified" and not item.get("data_simulated"):
            lims.append(f"Series `{sid}` could not be checked against the requested variable "
                        f"'{item.get('series_name', sid)}': the source supplied no informative title.")
        if item.get("discontinued"):
            lims.append(f"Series `{sid}` is discontinued: its last observation is "
                        f"{item.get('end_date', '?')}.")
    for obj in _iter(((data_source or {}).get("metadata") or {}).get("hitl_objections")):
        q = str(obj.get("checkpoint") or "data checkpoint")[:120]
        why = str(obj.get("final_response") or obj.get("response") or "")[:200]
        if obj.get("revised"):
            q += ", after one revision"
        lims.append(f"Data checkpoint objection ({q}): {why}" if why else
                    f"Data checkpoint objection recorded at: {q}")
    for cm in _iter((model_specification or {}).get("calibrated_models")):
        meta = cm.get("metadata") or {}
        status = meta.get("calibration_status")
        if status and status not in ("calibrated",):
            reason = meta.get("uncalibratable_reason")
            title = str(cm.get("model_title", "?"))[:60]
            est = meta.get("harness_estimated_params") or {}
            scope = ""
            if status == "archetype_calibrated":
                scope = (f" — the harness simulated the canonical stand-in "
                         f"'{meta.get('simulator') or '?'}', not this model's equations, and "
                         f"estimated only {', '.join(est) or 'no parameter'}; every other "
                         "parameter keeps its proposed value")
            elif status == "point_calibrated":
                scope = (f" — just-identified: {', '.join(est) or 'no parameter'} estimated to "
                         "reproduce the cited target(s); no over-identifying test")
            lims.append(f"Model '{title}' calibration verdict: `{status}`" + scope
                        + (f" — {str(reason)[:140]}" if reason else ""))
    outcome = (estimation_results or {}).get("outcome") or {}
    if outcome.get("verdict") == "inestimable":
        lims.append(f"Estimation honestly declined: `{outcome.get('reason')}` — no empirical "
                    "coefficients are reported.")
    elif outcome.get("verdict") == "fragile":
        lims.append("Estimation verdict is `fragile`: severe diagnostic failures — "
                    "interpret coefficients with caution.")
    for note in outcome.get("notes") or []:
        if "downgraded" in str(note) or "preview" in str(note):
            lims.append(f"Estimation note: {str(note)[:180]}")
    # Theory-derived predictions the data did not support: an inconclusive or
    # contradicted hypothesis is a finding the reader needs next to the limitations.
    for h in ((estimation_results or {}).get("inference") or {}).get("hypotheses") or []:
        oc = h.get("outcome") or ("supported" if h.get("supported") else
                                  "not supported" if h.get("supported") is False else None)
        if oc in (None, "supported", "consistent"):
            continue
        lims.append(f"Hypothesis '{h.get('name')}' (`{h.get('param')} {h.get('restriction')}`): "
                    f"`{oc}` — estimate {_fmt(h.get('estimate'))}, p = {_fmt(h.get('p_value'))}")
    # Committee objections that were not resolved within the one bounded revision
    for obj in _estimation_meta(estimation_results, "committee_objections") or []:
        lims.append(f"Estimation-specification review (before estimation): {_clip(obj)}")
    for obj in report_objections or []:
        lims.append(f"Report-draft review (published after one revision): {_clip(obj)}")
    if feasibility_report and feasibility_report.get("status") not in (None, "feasible", "resolved"):
        lims.append(f"Model–data feasibility loop terminated `{feasibility_report.get('status')}` "
                    f"after {feasibility_report.get('cycles')} cycle(s); unresolved requirements "
                    "are documented in the Data Availability Report.")
    return lims


def _format_reference(item: Dict) -> str:
    """One APA-ish reference line from a literature-batch item. Only artifact fields."""
    authors = item.get("authors") or []
    if isinstance(authors, list):
        cleaned = [str(a).strip() for a in authors if str(a).strip()]
        names = ", ".join(cleaned[:6]) + (" et al." if len(cleaned) > 6 else "")
    else:
        names = str(authors).strip()
    year = item.get("year")
    # The venue is the journal/repository; the search provider ('source': Crossref,
    # OpenAlex, arXiv, ...) is never printed as one. Unknown venue -> left empty.
    from shared.tools.scholarly_search import clean_venue
    venue = clean_venue(item.get("venue")) or ""
    url = str(item.get("url") or "").strip()
    parts = [f"{names or 'Unknown'} ({year if year else 'n.d.'})."]
    parts.append(f"{str(item.get('title', '')).strip()}.")
    if venue:
        parts.append(f"*{venue}*.")
    if url:
        parts.append(url)
    return " ".join(p for p in parts if p and p != ".")


def assemble_report(
    research_questions: Dict,
    literature_review: Dict,
    model_specification: Dict,
    data_source: Dict,
    estimation_results: Dict,
    interpretation: InterpretationResult,
    narratives: Optional[Dict[str, str]] = None,
    feasibility_report: Optional[Dict] = None,
    pipeline_run_id: str = "",
    literature_batch: Optional[Dict] = None,
    code_generation: Optional[Dict] = None,
    code_validation: Optional[Dict] = None,
    report_objections: Optional[List[str]] = None,
) -> str:
    """Build the full markdown report. Deterministic skeleton + narrative slots."""
    narratives = narratives or {}
    lines: List[str] = []
    add = lines.append
    question = _first_question(research_questions)

    add(f"# Research Report: {_short_title(question)}")
    add("")
    add("*Produced end-to-end by the AEL research pipeline; every number below is computed "
        "from, and cross-checked against, the pipeline's own artifacts.*")

    add("\n## 1. Research Question")
    add(f"> {question}")
    if narratives.get("intro"):
        add("\n" + _NARRATIVE_MARK.format(slot="intro"))
        add(narratives["intro"].strip())

    # Literature — the synthesis artifact is a structured dict (title/abstract/sections/
    # synthesis); render its parts instead of dumping the dict repr.
    add("\n## 2. Literature Context")
    review = literature_review.get("literature_review") if isinstance(literature_review, dict) else None
    if isinstance(review, dict):
        if review.get("title"):
            add(f"**{str(review['title']).strip()}**\n")
        for key in ("abstract", "synthesis"):
            text = str(review.get(key) or "").strip()
            if text:
                add(text[:1500] + ("…" if len(text) > 1500 else ""))
                add("")
    else:
        review_text = ""
        if isinstance(literature_review, dict):
            review_text = str(literature_review.get("review_text") or review or "")
        if review_text and review_text != "None":
            add(review_text.strip()[:1800] + ("…" if len(review_text) > 1800 else ""))
        else:
            add("*(no literature review artifact available)*")

    # Model
    add("\n## 3. Model")
    models = _iter((model_specification or {}).get("calibrated_models"))
    if models:
        for cm in models[:3]:
            meta = cm.get("metadata") or {}
            add(f"\n### {cm.get('model_title', 'Model')}")
            status = meta.get("calibration_status", "unknown")
            add(f"Calibration verdict (deterministic harness): `{status}`")
            est = meta.get("harness_estimated_params") or {}
            if meta.get("simulator"):
                add(f"Simulator: `{meta.get('simulator')}`"
                    + (" (canonical stand-in, not this model's equations)"
                       if meta.get("simulator_is_archetype") else ""))
            params = _iter(cm.get("calibrated_parameters"))
            if params:
                # The language model's proposal and the harness's estimate are distinct
                # columns; only the latter is a calibration result.
                add("\n| Parameter | Proposed value (LLM) | Harness estimate | Basis |")
                add("|---|---|---|---|")
                for p in params[:12]:
                    sym = p.get('parameter_symbol', p.get('parameter_name', '?'))
                    add(f"| {sym} | {_fmt(p.get('calibrated_value'))} "
                        f"| {_fmt(est[sym]) if sym in est else '—'} "
                        f"| {str(p.get('calibration_basis', p.get('justification', '')))[:60]} |")
                for sym in [k for k in est if k not in {p.get('parameter_symbol') for p in params}]:
                    add(f"| {sym} | — | {_fmt(est[sym])} | harness-estimated |")
    else:
        add("*(no model specification artifact available)*")

    # Data
    add("\n## 4. Data")
    retrieved = _iter((data_source or {}).get("retrieved_data"))
    if retrieved:
        add("| Series | Source | Obs | Range | Status |")
        add("|---|---|---|---|---|")
        for item in retrieved:
            status = "SIMULATED (disclosed)" if item.get("data_simulated") else "real"
            if "repaired" in str(item.get("quality_notes", "")):
                status += ", id-repaired (disclosed)"
            if item.get("proxy_for"):
                status += f", proxy for '{str(item.get('proxy_for'))[:40]}'"
            elif item.get("title_check") == "unverified" and not item.get("data_simulated"):
                status += ", title unverified"
            if item.get("discontinued"):
                status += ", discontinued"
            if item.get("projections_excluded"):
                status += f", {item.get('projections_excluded')} projected obs. excluded"
            # The source's own title names the series; the requested name only if absent
            label = str(item.get("source_title") or item.get("series_name", ""))[:40]
            add(f"| `{item.get('series_id', '?')}` ({label}) "
                f"| {item.get('source_name', '?')} | {item.get('num_observations', '?')} "
                f"| {item.get('start_date', '?')}–{item.get('end_date', '?')} | {status} |")
    else:
        add("*(no retrieved-data artifact available)*")

    # Estimation
    add("\n## 5. Estimation Results")
    outcome = (estimation_results or {}).get("outcome") or {}
    verdict = outcome.get("verdict", "missing")
    add(f"Harness verdict: `{verdict}`"
        + (f" ({outcome.get('reason')})" if outcome.get("reason") else ""))
    if verdict in ("estimated", "fragile"):
        add(f"\nSpecification: `{outcome.get('dependent_name')}` "
            f"~ {' + '.join(c['name'] for c in outcome.get('coefficients', []) if c.get('name') != 'const')} "
            f"({str(outcome.get('method') or 'ols').upper()}, {outcome.get('cov_type')} SEs), "
            f"n = {outcome.get('n_obs')}, "
            f"R² = {_fmt(outcome.get('r_squared'))}, "
            f"sample {outcome.get('sample_start')}–{outcome.get('sample_end')} "
            f"({outcome.get('frequency')}).")
        terms = render_spec_terms(outcome.get("spec") or {})
        if terms:
            add("\nConstruction of each column (as built by the harness):")
            for t in terms:
                add(f"- {t}")
        add("\n| Coefficient | Estimate | Std. error | p-value | 95% CI |")
        add("|---|---|---|---|---|")
        for c in outcome.get("coefficients") or []:
            add(f"| {c['name']} | {_fmt(c['estimate'])} | {_fmt(c['std_error'])} "
                f"| {_fmt(c['p_value'])} | [{_fmt(c['ci_low'])}, {_fmt(c['ci_high'])}] |")
        if interpretation.effects:
            add("\n**Economic magnitudes** (computed from the analysis panel):")
            add("\n| Regressor | Kind | One-SD effect | Standardized β | Elasticity at means |")
            add("|---|---|---|---|---|")
            na = lambda x: "n/a" if x is None else _fmt(x)   # noqa: E731
            for e in interpretation.effects:
                add(f"| {e.param} | {e.interpretation_kind} | {na(e.one_sd_effect)} "
                    f"| {na(e.standardized_beta)} | {na(e.elasticity_at_means)} |")
            for e in interpretation.effects:
                if getattr(e, "note", ""):
                    add(f"\n*{e.param}: {e.note}*")
        inference = (estimation_results or {}).get("inference") or {}
        hyps = inference.get("hypotheses") or []
        if hyps:
            add("\n**Theory-derived hypotheses:**")
            add("\n| Hypothesis | Restriction | Estimate | p-value | Outcome |")
            add("|---|---|---|---|---|")
            for h in hyps:
                # outcome distinguishes 'inconclusive' from 'contradicted'; older
                # artifacts only carry the boolean 'supported'.
                oc = h.get("outcome") or ("supported" if h.get("supported") else
                                          "not supported" if h.get("supported") is False else "n/a")
                add(f"| {h.get('name')} | `{h.get('param')} {h.get('restriction')}` "
                    f"| {_fmt(h.get('estimate'))} | {_fmt(h.get('p_value'))} | {oc} |")
        rob = inference.get("robustness") or []
        if rob:
            # Only variants the harness could estimate count as robustness checks
            n_est = sum(1 for c in rob if c.get("variant_estimate") is not None)
            n_not = len(rob) - n_est
            add(f"\nRobustness: {n_est} estimable variant(s)"
                + (f", sign-stability {_fmt(inference.get('stability_score'))}"
                   if inference.get("stability_score") is not None else "")
                + (f"; {n_not} variant(s) could not be estimated" if n_not else "") + ".")
        # Identification + proxy statements travel from the harness artifact into
        # the report verbatim — the reader sees the design's limits without opening JSON.
        for n in outcome.get("notes") or []:
            if str(n).startswith(("identification:", "proxy disclosure:",
                                  "interaction without main effects:")):
                add(f"\n*{n}*")
        # Which variables of the CodeTeam's executable model the specification measured
        use = (estimation_results or {}).get("executable_model_use") or {}
        if use.get("available"):
            linked = use.get("linked_variables") or {}
            ss = use.get("steady_state_values") or {}
            parts = [f"`{r}` -> `{s}`" + (f" (steady state {_fmt(ss[s])})" if s in ss else "")
                     for r, s in linked.items()]
            add(f"\nExecutable model: module for '{str(use.get('model_title', ''))[:80]}' "
                f"(`{use.get('generation_verdict')}` / `{use.get('validation_verdict')}`); "
                + ("series linked to module variables: " + ", ".join(parts) + "."
                   if parts else "no series in the specification is linked to a module "
                                 "variable."))
    # Every specification the harness fitted, its verdict, and which one is reported
    fitted = _iter(_estimation_meta(estimation_results, "fitted_specifications"))
    reviews = _iter(_estimation_meta(estimation_results, "spec_review"))
    if reviews:
        add("\nCommittee review of the specification before estimation: "
            + "; ".join(f"round {r.get('round')} ({str(r.get('candidate', '')).replace('_', ' ')})"
                        f" {'approved' if r.get('approved') else 'rejected'}" for r in reviews)
            + ".")
    if fitted:
        add("\n**Specifications fitted by the harness** (all fits, in order; a repair "
            "rejected before fitting is listed as `not_fitted`):")
        add("\n| # | Origin | Committee | Specification | Verdict | n | Reported |")
        add("|---|---|---|---|---|---|---|")
        for i, f in enumerate(fitted, 1):
            add(f"| {i} | {str(f.get('label', '')).replace('_', ' ')} "
                f"| {str(f.get('committee', '')).replace('_', ' ')} "
                f"| {_compact_spec(f.get('spec') or {})} "
                f"| `{f.get('verdict')}`" + (f" ({f.get('reason')})" if f.get("reason") else "")
                + f" | {f.get('n_obs') if f.get('fitted', True) else '-'} "
                f"| {'yes' if f.get('reported') else 'no'} |")
        for i, f in enumerate(fitted, 1):
            if f.get("reported") and f.get("selection"):
                add(f"\nReported specification: #{i} — {f.get('selection')}.")
    if interpretation.figure_files:
        add("\nFigures: " + ", ".join(f"`{f}`" for f in interpretation.figure_files))

    # Code implementation: deterministic inventory of the generated modules, verdicts
    # verbatim, storage pointer
    if code_generation:
        add("\n## 5b. Code Implementation")
        gens = (code_generation or {}).get("results") or []
        vals = (code_validation or {}).get("results") or [] if code_validation else []
        add("Executable steady-state modules derived deterministically from each formal "
            "model's parsed equation system (sympy → numpy/scipy; no free-hand LLM code). "
            "Modules and a run manifest are published under `5_code/generated_models/` "
            "in the run's output folder; each runs standalone (`python model_N.py`).")
        add("\n| Model | Generation | Equations parsed | Validation | Checks passed |")
        add("|---|---|---|---|---|")
        for i, g in enumerate(gens):
            v = vals[i] if i < len(vals) else {}
            checks = v.get("checks") or []
            n_pass = sum(1 for c in checks if c.get("passed"))
            add(f"| {str(g.get('model_title', ''))[:60]} | `{g.get('verdict')}` "
                f"| {g.get('n_parseable', 0)}/{g.get('n_equations', 0)} "
                f"| `{v.get('verdict', 'n/a')}` | {n_pass}/{len(checks)} |")
        refused_total = derived_numbers(code_generation).get(
            "code_generation.refused_total", {}).get("value", 0)
        if refused_total:
            # name the refusal kinds that actually occurred (the parser's status labels)
            _kinds = {"refused_nonalgebraic": "non-algebraic constructs",
                      "refused_differential": "differentials",
                      "refused_difference_operator": "difference operators",
                      "refused_symbol_collision": "symbol collisions",
                      "refused_superscript_label": "superscript labels",
                      "refused_undefined_function": "undefined functions",
                      "refused_unparsed_subscript": "unparsed subscripts",
                      "refused_prose": "prose"}
            seen = []
            for g in gens:
                for r in g.get("refused") or []:
                    st = str(r.get("status", ""))
                    lbl = _kinds.get(st, "parse failures" if st.startswith("parse_fail") else None)
                    if lbl and lbl not in seen:
                        seen.append(lbl)
            add(f"\nThe parser refused {refused_total} equation(s) across the models"
                + (f" ({', '.join(seen)})" if seen else "")
                + "; refused equations are left out of the module, not approximated.")

    # Discussion (LLM narrative)
    add("\n## 6. Discussion")
    if narratives.get("discussion"):
        add(_NARRATIVE_MARK.format(slot="discussion"))
        add(narratives["discussion"].strip())
    else:
        add("*(no narrative discussion)*")

    # Limitations
    add("\n## 7. Limitations")
    lims = collect_limitations(data_source, model_specification, estimation_results,
                               feasibility_report, report_objections=report_objections)
    if lims:
        for lim in lims:
            add(f"- {lim}")
    else:
        add("- No disclosed limitations were recorded by the pipeline for this run.")

    # References — the reviewed corpus, straight from the literature artifact (never
    # LLM-invented; the consistency checker verifies years/URLs against the artifact).
    items = [i for i in _iter((literature_batch or {}).get("literature_items"))
             if i.get("title")]
    if items:
        add("\n## 8. References")
        add("*The literature corpus gathered and reviewed by the pipeline (Section 2 "
            "synthesizes these sources).*\n")
        items = sorted(items, key=lambda i: (str((i.get("authors") or ["ZZ"])[0]
                                                 if isinstance(i.get("authors"), list)
                                                 else i.get("authors", "ZZ")).lower(),
                                             str(i.get("year") or "")))
        # bullets, not ordinals: enumeration numbers are template-generated and would be
        # flagged by the number-consistency check as data-less numbers
        for item in items:
            add(f"- {_format_reference(item)}")

    # Provenance
    add(f"\n## {'9' if items else '8'}. Provenance")
    add(f"- Pipeline run: `{pipeline_run_id or 'standalone'}`")
    add("- Source artifacts: `research_questions`, `literature_review`, "
        "`model_specification`, `data_source`, `estimation_results`"
        + (", `literature_batch`" if items else ""))
    add("- Every table above is generated directly from these artifacts; narrative sections "
        "are LLM-written and verified by the Stage 3 number-consistency check.")

    return "\n".join(lines) + "\n"
