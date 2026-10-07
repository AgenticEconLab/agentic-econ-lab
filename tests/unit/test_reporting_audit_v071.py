# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Reporting fixes found by auditing the v0.7.1 runs.

- report-draft gate: a revision is re-voted; a rejected revision is published with the
  objections in Limitations; the estimation section lists every fitted specification
- consistency check: DOI/hash fragments are not numbers; non-finite values never verify
- template-derived numbers come from a 'derived' source; percent scaling only for numbers
  printed as percentages
- effect kinds per (dependent, regressor) transform pair; n/a for trend and interaction rows
- the specification is rendered term by term; interactions without main effects are flagged
- the quality stage receives data_source/literature_batch/question and writes into the run dir
- robustness count of estimable variants; titles shortened at a word boundary
"""

import json

import numpy as np
import pytest

from ReportingTeam.ael.report_harness import (
    assemble_report,
    check_consistency,
    derived_numbers,
    interpret_estimation,
    render_spec_terms,
)
from ReportingTeam.ael.report_harness.consistency import extract_numbers


# ---------------------------------------------------------------- consistency check
def test_doi_and_hash_fragments_are_not_numbers():
    text = ("see https://doi.org/10.1016/j.jmoneco.2020.4e903 and commit 439e62711; "
            "version v0.7.1; series x1; the estimate is 0.42.")
    values = [v for v, _ in extract_numbers(text)]
    assert values == [0.42]


def test_inf_in_pool_cannot_verify_anything():
    artifacts = {"lit": {"doi": "10.1016/j.x.2020.4e903", "weird": "1e999", "x": float("inf")}}
    rep = check_consistency("The effect is 0.4321 and the count is 4698.78.", artifacts)
    assert rep.verdict == "has_unverified" and rep.verified == 0
    assert {u.value for u in rep.unverified} == {0.4321, 4698.78}


# ---------------------------------------------------------------- derived numbers, percent scaling
def test_percent_scaling_only_for_printed_percentages():
    pool = {"a": {"x": 4698.78}}
    assert check_consistency("We count 47 cases.", pool).verdict == "has_unverified"
    assert check_consistency("A rise of 47% in the index.", {"a": {"x": 0.47}}).verdict == "consistent"
    assert check_consistency("about 12.8 percent", [{"share": 0.128}]).verdict == "consistent"


def test_refused_total_is_verified_from_the_derived_source():
    gens = {"results": [{"model_title": "m1", "verdict": "partial", "n_parseable": 3,
                         "n_equations": 20, "refused": ["e"] * 9},
                        {"model_title": "m2", "verdict": "partial", "n_parseable": 2,
                         "n_equations": 18, "refused": ["e"] * 8}]}
    derived = derived_numbers(gens)
    assert derived["code_generation.refused_total"]["value"] == 17
    text = "The parser refused 17 equation(s) across the models."
    assert check_consistency(text, {"code_generation": gens}).verdict == "has_unverified"
    rep = check_consistency(text, {"code_generation": gens, "derived": derived})
    assert rep.verdict == "consistent" and rep.verified_numbers[0].sources == ["derived"]


def _estimation(spec, coefs, rows):
    return {"outcome": {
        "verdict": "estimated", "dependent_name": spec["dependent"]["name"], "n_obs": len(rows),
        "r_squared": 0.5, "cov_type": "HAC", "method": "ols",
        "sample_start": "1960-01-01", "sample_end": "2020-01-01", "frequency": "annual",
        "coefficients": [{"name": n, "estimate": b, "std_error": 0.1, "t_stat": 1.0,
                          "p_value": 0.3, "ci_low": b - 0.2, "ci_high": b + 0.2}
                         for n, b in coefs],
        "analysis_data": rows, "notes": [], "spec": spec},
        "inference": {"hypotheses": [], "robustness": [], "stability_score": None}}


def _rows(n=40, seed=1, cols=("y", "x", "z", "xz")):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        r = {"date": f"{1960 + i}-01-01"}
        for c in cols:
            r[c] = float(rng.normal(5, 1))
        out.append(r)
    return out


# ---------------------------------------------------------------- effect kinds
def test_growth_dependent_gets_no_elasticity_at_means():
    spec = {"dependent": {"name": "y", "series_ref": "CPI", "transform": "pct_change"},
            "regressors": [{"name": "x", "series_ref": "FF", "transform": "level"}]}
    res = interpret_estimation(_estimation(spec, [("const", 1.0), ("x", 0.5)],
                                           _rows(cols=("y", "x"))))
    e = res.effects[0]
    assert e.interpretation_kind == "growth_effect" and e.elasticity_at_means is None
    assert e.one_sd_effect is not None


def test_level_level_keeps_elasticity_at_means():
    spec = {"dependent": {"name": "y", "series_ref": "Y"},
            "regressors": [{"name": "x", "series_ref": "X"}]}
    res = interpret_estimation(_estimation(spec, [("x", 0.5)], _rows(cols=("y", "x"))))
    assert res.effects[0].interpretation_kind == "marginal_effect"
    assert res.effects[0].elasticity_at_means is not None


def test_trend_and_interaction_rows_get_no_one_sd_or_elasticity():
    spec = {"dependent": {"name": "y", "series_ref": "Y"},
            "regressors": [{"name": "x", "series_ref": "X"}, {"name": "z", "series_ref": "Z"},
                           {"name": "xz", "series_ref": "X", "interact_with": "Z"}],
            "include_trend": True}
    rows = _rows(cols=("y", "x", "z", "xz", "trend"))
    res = interpret_estimation(_estimation(
        spec, [("const", 1.0), ("x", 0.5), ("z", 0.2), ("xz", 0.1), ("trend", 0.01)], rows))
    by = {e.param: e for e in res.effects}
    for name in ("xz", "trend"):
        assert by[name].one_sd_effect is None and by[name].elasticity_at_means is None
        assert by[name].standardized_beta is None and by[name].note
    mean_z = np.mean([r["z"] for r in rows])
    assert by["xz"].conditional_marginal_effect == pytest.approx(0.5 + 0.1 * mean_z)


def test_interaction_without_main_effects_reports_na():
    spec = {"dependent": {"name": "y", "series_ref": "Y"},
            "regressors": [{"name": "xz", "series_ref": "X", "interact_with": "Z"}]}
    res = interpret_estimation(_estimation(spec, [("xz", 0.1)], _rows(cols=("y", "xz"))))
    e = res.effects[0]
    assert e.conditional_marginal_effect is None and "main effect" in e.note


# ---------------------------------------------------------------- term-by-term specification
def test_spec_rendered_term_by_term():
    spec = {"dependent": {"name": "Inflation", "series_ref": "CPIAUCSL",
                          "transform": "yoy_pct_change"},
            "regressors": [{"name": "Real Rate x Gini", "series_ref": "FEDFUNDS",
                            "transform": "level", "subtract_ref": "CPIAUCSL",
                            "subtract_transform": "yoy_pct_change",
                            "interact_with": "SI.POV.GINI", "lag": 1}]}
    lines = render_spec_terms(spec)
    assert lines[0] == "dependent `Inflation` = yoy_pct_change(CPIAUCSL)"
    assert lines[1] == ("regressor `Real Rate x Gini` = [FEDFUNDS − yoy_pct_change(CPIAUCSL)] "
                        "× SI.POV.GINI, lag 1")


def test_harness_flags_interaction_without_main_effects():
    import pandas as pd
    from EstimationTeam.ael.estim_harness import EstimationSpec, run_estimation
    rng = np.random.default_rng(0)
    idx = pd.date_range("1950-01-01", periods=80, freq="YS")
    panel = {k: pd.Series(rng.normal(5, 1, 80), index=idx) for k in ("Y", "X", "Z")}
    spec = EstimationSpec(**{"dependent": {"name": "y", "series_ref": "Y"},
                             "regressors": [{"name": "x", "series_ref": "X"},
                                            {"name": "xz", "series_ref": "X",
                                             "interact_with": "Z"}]})
    out = run_estimation(spec, panel)
    flagged = [n for n in out.notes if n.startswith("interaction without main effects:")]
    assert len(flagged) == 1 and "'Z'" in flagged[0] and "'X'" not in flagged[0]


# ---------------------------------------------------------------- assembly: draft gate, specification, robustness
def _assemble(est, question="Does X drive Y?", **kw):
    interp = interpret_estimation(est)
    return assemble_report({"final_questions": [{"question": question}]}, {}, {}, {},
                           est, interp, **kw)


def test_report_lists_fitted_specs_objections_and_construction():
    spec = {"dependent": {"name": "y", "series_ref": "Y"},
            "regressors": [{"name": "x", "series_ref": "X"}]}
    est = _estimation(spec, [("const", 1.0), ("x", 0.5)], _rows(cols=("y", "x")))
    big = {"dependent": {"name": "y", "series_ref": "Y"},
           "regressors": [{"name": f"r{i}", "series_ref": "X", "lag": i} for i in range(6)]}
    est["fitted_specifications"] = [
        {"label": "proposal", "committee": "rejected", "spec": big, "verdict": "inestimable",
         "reason": "insufficient_observations", "n_obs": 0, "reported": False, "selection": ""},
        {"label": "refusal_repair", "committee": "not_reviewed", "spec": spec,
         "verdict": "estimated", "reason": None, "n_obs": 40, "reported": True,
         "selection": "repair after the harness refused the proposal"}]
    est["spec_review"] = [{"round": 1, "candidate": "proposal", "approved": False},
                          {"round": 2, "candidate": "committee_revision", "approved": False}]
    est["committee_objections"] = ["Committee objections to the proposed specification: "
                                   "omits the policy rate"]
    md = _assemble(est, report_objections=["objections to the revised draft: too long"])
    assert "Specifications fitted by the harness" in md
    assert "| 1 | proposal | rejected |" in md and "| 2 | refusal repair | not reviewed |" in md
    assert "Reported specification: #2" in md
    assert "round 1 (proposal) rejected; round 2 (committee revision) rejected" in md
    lims = md.split("## 7. Limitations")[1]
    assert "omits the policy rate" in lims and "too long" in lims
    assert "- regressor `x` = X" in md
    rep = check_consistency(md, {"est": est, "interp": interpret_estimation(est).model_dump()})
    assert rep.verdict == "consistent", rep.unverified


def test_robustness_counts_estimable_variants_only():
    spec = {"dependent": {"name": "y", "series_ref": "Y"},
            "regressors": [{"name": "x", "series_ref": "X"}]}
    est = _estimation(spec, [("x", 0.5)], _rows(cols=("y", "x")))
    est["inference"]["robustness"] = [
        {"name": "a", "variant_estimate": 0.4, "sign_stable": True},
        {"name": "b", "variant_estimate": None, "sign_stable": None},
        {"name": "c", "variant_estimate": None, "sign_stable": None}]
    est["inference"]["stability_score"] = 1.0
    md = _assemble(est)
    assert "Robustness: 1 estimable variant(s), sign-stability 1; 2 variant(s) could not be " \
           "estimated." in md


def test_long_title_cut_at_word_boundary():
    q = ("How does the interaction between monetary policy tightening and household "
         "wealth inequality shape aggregate consumption dynamics in heterogeneous-agent "
         "economies with borrowing constraints?")
    est = {"outcome": {"verdict": "inestimable", "reason": "no_data"}}
    md = _assemble(est, question=q)
    title = md.splitlines()[0][len("# Research Report: "):]
    assert title.endswith("…") and len(title) <= 121
    assert q.startswith(title[:-1]) and q[len(title) - 1] == " "
    assert f"> {q}" in md


# ---------------------------------------------------------------- draft gate and quality-stage inputs via the runner
def _runner_upstream():
    spec = {"dependent": {"name": "outcome", "series_ref": "Y"},
            "regressors": [{"name": "driver", "series_ref": "X"}]}
    est = _estimation(spec, [("const", 1.0), ("driver", 2.0)],
                      _rows(n=60, cols=("outcome", "driver")))
    return {
        "research_questions": {"final_questions": [{"question": "Does X drive Y?"}]},
        "literature_review": {"review_text": "Prior evidence is mixed."},
        "model_specification": {"calibrated_models": []},
        "data_source": {"retrieved_data": [
            {"series_id": "XSER", "series_name": "Driver", "source_name": "FRED",
             "num_observations": 60, "start_date": "1990-01-01", "end_date": "1994-12-01",
             "data_simulated": False, "quality_notes": "ok"}]},
        "estimation_results": est,
    }


def test_quality_stage_gets_data_source_and_writes_into_run_dir(monkeypatch, tmp_path):
    from shared.llm import LLMClient
    from pipeline.team_runners import run_reporting_team
    monkeypatch.setenv("AEL_HITL_MODE", "auto")
    monkeypatch.setattr(LLMClient, "invoke", lambda self, messages, **kw: "Plain prose.")
    code_tree = tmp_path / "code"
    code_tree.mkdir()
    run_dir = tmp_path / "run" / "ReportingTeam"
    run_dir.mkdir(parents=True)
    monkeypatch.chdir(code_tree)
    run_reporting_team(_runner_upstream(), "ModeNoWcNoHITL", str(run_dir))
    q = json.loads((run_dir / "quality_output.json").read_text())["quality"]
    assert "No retrieved-data artifact" not in q["availability_statement"]
    assert "1 of 1 series retrieved" in q["availability_statement"]
    assert (run_dir / "research_report_formatted.md").exists()
    assert not list(code_tree.iterdir()), "nothing may be written outside the run directory"


def test_rejected_draft_revision_is_revoted_then_published_with_objections(monkeypatch, tmp_path):
    from shared.llm import LLMClient
    import pipeline.team_runners as tr
    monkeypatch.setenv("AEL_HITL_MODE", "llm_economist")
    monkeypatch.setattr(LLMClient, "invoke", lambda self, messages, **kw: "Plain prose.")
    votes = []
    monkeypatch.setattr(tr, "_stage_approved",
                        lambda r, ctx, q, auto_rounds=2: votes.append(r) or False)
    fbs = iter(["first draft overstates causality", "revision still overstates"])
    monkeypatch.setattr(tr, "_collect_feedback", lambda method, ctx, **kw: next(fbs))
    monkeypatch.chdir(tmp_path)
    out = tr.run_reporting_team(_runner_upstream(), "ModeWithWcWithHITL", str(tmp_path))
    assert votes == [1, 2]
    md = out["research_report"]["report_markdown"]
    lims = md.split("## 7. Limitations")[1]
    assert "first draft overstates causality" in lims and "revision still overstates" in lims
    draft = json.loads((tmp_path / "drafting_output.json").read_text())
    assert len(draft["drafting"]["metadata"]["report_objections"]) == 2


def test_drafting_feedback_file_goes_to_output_dir(monkeypatch, tmp_path):
    import importlib.util
    from pathlib import Path
    path = (Path(__file__).resolve().parents[2] / "ReportingTeam" / "ael" / "ModeNoWcWithHITL"
            / "2-DraftingStage.py")
    spec = importlib.util.spec_from_file_location("_draft_fb", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "auto_input", lambda *a, **k: "concern")
    out_dir, cwd = tmp_path / "run", tmp_path / "code"
    out_dir.mkdir()
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    mod.DraftingOrchestrator(quiet=True, output_dir=str(out_dir)).collect_human_feedback(1, {})
    assert (out_dir / "round1_reporting_feedback.json").exists()
    assert not list(cwd.iterdir())


def test_data_disclosures_reach_table_and_limitations():
    from ReportingTeam.ael.report_harness.assemble import collect_limitations
    ds = {"retrieved_data": [
        {"series_id": "X1", "series_name": "Thing", "title_check": "unverified", "end_date": "2024-01-01"},
        {"series_id": "TEDRATE", "series_name": "TED Spread", "source_title": "TED Spread",
         "discontinued": True, "end_date": "2022-01-21"}],
        "metadata": {"hitl_objections": [{"checkpoint": "series selection", "approved": False,
                                          "response": "frequency mismatch", "revised": True}]}}
    lims = collect_limitations(ds, {}, {})
    assert any("`X1` could not be checked" in l for l in lims)
    assert any("`TEDRATE` is discontinued" in l and "2022-01-21" in l for l in lims)
    assert any("series selection, after one revision" in l and "frequency mismatch" in l for l in lims)
